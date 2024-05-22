import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import argparse
from tensorboardX import SummaryWriter
import torchvision.transforms as transforms
import torchvision
from models.resnet import ResNet18, ResNet34
from models.dla import DLA
from models.resnet_orig import ResNet18_orig, ResNet34_orig
from models.dla_orig import DLA_orig
from models.wide_resnet_orig import wideResnet_orig
from models.simple_conv import simpleConv
from models.simple_conv_orig import simpleConv_orig

import random
import numpy as np
import sys
import os
import pandas as pd

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.append(parentdir)

from utils.Empirical.architectures import ARCHITECTURES
from utils.Empirical.datasets import DATASETS
from utils.Empirical.utils_ensemble import AverageMeter, accuracy, test, copy_code, requires_grad_, evaltrans, evaltrans_correct
from utils.Empirical.datasets import get_dataset
from utils.Empirical.architectures import get_architecture
from train.Empirical.trainer import Naive_Trainer, cos_Trainer, Naive_Trainer_ortho, Naive_Trainer_ortho_catclip, Naive_Trainer_ortho_catclip_2, cos_Trainer_ortho, TRS_Trainer
from utils_cifar100 import get_training_dataloader, get_test_dataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--dataset', default='cifar')
parser.add_argument('--arch', default='ResNet18', type=str)
parser.add_argument('--workers', default=2, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=201, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--batch', default=128, type=int, metavar='N', help='batchsize (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, help='initial learning rate', dest='lr')
parser.add_argument('--lr_step_size', type=int, default=40, help='How often to decrease learning by gamma.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--num-models', default=3, type=int)
parser.add_argument('--adv-training', action='store_true')
parser.add_argument('--epsilon', default=512, type=float)
parser.add_argument('--num-steps', default=4, type=int)
parser.add_argument('--adv-eps', default=0.04, type=float)
parser.add_argument('--widen_factor', default=1, type=int, help='widen factor for WideResNet')
parser.add_argument('--tech', default='vanilla', help='vanilla for no prior method and trs for TRS')

############# FastClip:
parser.add_argument('--method', default='orig', help='clipping method (use orig for no clipping)')
parser.add_argument('--mode', default='wBN', help='what to do with BN layers (leave empty for keeping it as it is)')
parser.add_argument('--seed', default=1, type=int, help='seed value')
parser.add_argument('--convsn', default=1., type=float, help='clip value for conv (and concatenations if the flag activated)')
########################################################################

############# TRS:
parser.add_argument('--plus-adv', action='store_true')
parser.add_argument('--coeff', default=40.0, type=float)
parser.add_argument('--lamda', default=2.5, type=float)
parser.add_argument('--scale', default=1.0, type=float)
parser.add_argument('--trs_adv_eps', default=8./255, type=float)
parser.add_argument('--init-eps', default=0.005, type=float)
########################################################################

############# LOTOS:
parser.add_argument('--bottom_clip', default=1.0, type=float, help='lower bound for smallest singular value of the conv layer')
parser.add_argument('--cat_bottom_clip', default=1.0, type=float, help='lower bound for smallest singular value of the concatenation')
parser.add_argument('--conv_factor', default=0.00, type=float, help='factor of layerwise bad length in loss for the conv layer')
parser.add_argument('--cat_factor', default=0.00, type=float, help='factor of layerwise bad length in loss for the concatenation')
parser.add_argument('--decrement', default=0.01, type=float, help='decrement for layerwise bad lengths when k > 1')
parser.add_argument('--efe', default=0, type=int, help='widen factor for WideResNet')
parser.add_argument('--cat', action='store_true', help='determines if the clipping method is applied to the concatenation of the conv and batch norm')
parser.add_argument('--fBN', action='store_true', help='determines if the batch norm layer exists for the first layer')
########################################################################


args = parser.parse_args()
div2_flag = False

freq = 50
conv_freq = 1
effective_epoch = args.efe
conv_only = False
conv_1st_only = False

trs_flag = False
if args.tech == 'trs':
    trs_flag = True
    conv_freq = 5
elif args.tech == 'vanilla':
    trs_flag = False

print('trs flag: ', trs_flag)

if args.adv_training:
    mode = f"adv_{args.epsilon}_{args.num_steps}"
else:
    mode = args.tech + "_"
    if args.method == 'orig':
        mode += "orig_"

        if not args.fBN and args.mode == 'wBN':
            mode += "1stnoBN_"
        mode += args.mode

    else:
        if args.cat:
            mode += "catclip" + str(args.convsn) + "_"
        else:
            mode += "clip" + str(args.convsn) + "_"

        if not args.fBN and args.mode == 'wBN':
            mode += "1stnoBN_"

        mode += args.mode


        if args.conv_factor > 0.00001 or args.cat_factor > 0.00001:
            mode = args.tech + "_" + f"ortho_convonly{conv_only}_1stconv{conv_1st_only}_efe{effective_epoch}_cat{freq}_conv{conv_freq}_catclip{args.convsn}_mal{args.bottom_clip}_mcat{args.cat_bottom_clip}_convfac{args.conv_factor}_catfac{args.cat_factor}_{args.mode}"



args.outdir = f"/{args.dataset}_{args.arch}/{mode}_{args.seed}/"
args.epsilon /= 256.0

args.outdir = "logs/" + args.outdir

print(args.outdir)
print('learning rate: ', args.lr)
print('dataset: ', args.dataset)
print('catclip: ', args.cat)
print('1stBN: ', args.fBN)
print('mode: ', args.mode)
print('bottom clip: ', args.bottom_clip)
print('cat bottom clip: ', args.cat_bottom_clip)
print('conv factor: ', args.conv_factor)
print('cat factor: ', args.cat_factor)

def main():
    clip_flag    = False
    orig_flag    = False

    seed_val = args.seed
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    np.random.seed(seed_val)
    random.seed(seed_val)

    mode = args.mode
    bn_flag = True
    opt_iter = 5
    clip_steps = 50
    if mode == 'wBN':
        mode = ''
        bn_flag = True
    elif mode == 'noBN':
        bn_flag = False
        opt_iter = 1
        clip_steps = 100
        args.epochs = 121

    if args.method == 'orig':
        orig_flag    = True
    else:
        clip_flag    = True

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    copy_code(args.outdir)

    if args.dataset == 'cifar':
        print('cifar!')
        in_chan = 3
        num_classes = 10
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = torchvision.datasets.CIFAR10( root='./data', train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader( trainset, batch_size=128, shuffle=True, num_workers=1)

        testset = torchvision.datasets.CIFAR10( root='./data', train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader( testset, batch_size=128, shuffle=False, num_workers=1)

    elif args.dataset == 'cifar100':
        print('cifar100!')
        in_chan = 3
        num_classes = 100
        train_loader = get_training_dataloader(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761), batch_size=128, num_workers=1)
        test_loader = get_test_dataloader(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761), batch_size=128, num_workers=1)

    else:
        print('dataset not recognized!')
        exit(0)

    model_path = os.path.join(args.outdir, 'checkpoint.pth.tar')
    writer = SummaryWriter(args.outdir)

    model = []
    for i in range(args.num_models):
        print('arch: ', args.arch)
        if clip_flag:
            if args.arch == 'ResNet18':
                submodel = ResNet18(concat_sv=args.cat, in_chan=in_chan, num_classes=num_classes, device=device, clip=args.convsn, clip_concat=args.convsn, clip_flag=True, bn=bn_flag, first_bn=args.fBN, clip_steps=clip_steps, clip_outer=args.cat, clip_opt_iter=opt_iter, summary=True, writer=writer, save_info=False, identifier=1000*i)
            elif args.arch == 'ResNet34':
                submodel = ResNet34(concat_sv=args.cat, in_chan=in_chan, num_classes=num_classes, device=device, clip=args.convsn, clip_concat=args.convsn, clip_flag=True, bn=bn_flag, first_bn=args.fBN, clip_steps=clip_steps, clip_outer=args.cat, clip_opt_iter=opt_iter, summary=True, writer=writer, save_info=False, identifier=1000*i)
            elif args.arch == 'DLA':
                submodel = DLA(concat_sv=args.cat, in_chan=in_chan, num_classes=num_classes, device=device, clip=args.convsn, clip_concat=args.convsn, clip_flag=True, bn=bn_flag, first_bn=args.fBN, clip_steps=clip_steps, clip_outer=args.cat, clip_opt_iter=opt_iter, summary=True, writer=writer, save_info=False, identifier=1000*i)
        elif orig_flag:
            print('adding orig model!')
            if args.arch == 'ResNet18':
                submodel = ResNet18_orig(in_chan=in_chan, num_classes=num_classes, bn=bn_flag, device=device, first_bn=args.fBN)
            elif args.arch == 'ResNet34':
                submodel = ResNet34_orig(in_chan=in_chan, num_classes=num_classes, bn=bn_flag, device=device, first_bn=args.fBN)
            elif args.arch == 'DLA':
                submodel = DLA_orig(in_chan=in_chan, num_classes=num_classes, bn=bn_flag, device=device, first_bn=args.fBN)
        submodel = nn.DataParallel(submodel)
        model.append(submodel)
    print("Model loaded")

    criterion = nn.CrossEntropyLoss().cuda()

    param = list(model[0].parameters())
    for i in range(1, args.num_models):
        param.extend(list(model[i].parameters()))

    optimizer = optim.SGD(param, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.gamma)

    trans_list = []
    loss_acc_list = []
    best_acc = 0.0
    batch_counter = 0
    lsv_list_dict = {}
    lsv_list_dict_conv = {}
    for epoch in range(args.epochs):
        print('catclip ortho')
        train_loss, batch_counter, lsv_list_dict, lsv_list_dict_conv = Naive_Trainer_ortho_catclip_2(args, train_loader, model, criterion, optimizer, epoch, device, writer, catclip=True, no_effect_epochs=effective_epoch, batch_counter=batch_counter, mal_freq=freq, conv_freq=conv_freq, layer_1_only=False, conv_1st_only=conv_1st_only, lsv_list_dict=lsv_list_dict, lsv_list_dict_conv=lsv_list_dict_conv, conv_only=conv_only, trs_flag=trs_flag)

        if True:
            test_acc, test_loss = test(test_loader, model, criterion, epoch, device, writer)
            loss_acc_list.append((epoch, train_loss, test_loss, test_acc))
            if test_acc >= best_acc:
                for i in range(args.num_models):
                    model_path_i = model_path + '_best_'+ ".%d" % (i)
                    torch.save({
                            'epoch': epoch,
                            'arch': args.arch,
                            'state_dict': model[i].state_dict(),
                        }, model_path_i)
                best_acc = test_acc
            writer.add_scalar('test/best_acc', best_acc, epoch)

        if args.mode == 'wBN':
            ensemble_cond = epoch % 20 == 0 and epoch >= 120
        elif args.mode == 'noBN':
            ensemble_cond = epoch % 20 == 0 and epoch >= 60 and epoch <=140
        else:
            print('mode not recognized!')
            exit(0)
        
        if ensemble_cond:
            trans_list_new = evaltrans_correct(args, test_loader, model, criterion, epoch, device, writer)
            trans_list += trans_list_new
            for i in range(args.num_models):
                model_path_i = model_path + ".%d" % (i) + "_%d" % (epoch)
                torch.save({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model[i].state_dict(),
                }, model_path_i)
        scheduler.step()

    trans_res_path = os.path.join(args.outdir, 'trans_e' + str(args.adv_eps) + '_s' + str(args.seed) + '.csv')

    col_names = ['epoch']
    for k in range(args.num_models):
        col_names.append('t' + str(k))
    col_names.append('acc')
    for k in range(args.num_models):
        col_names.append('conds' + str(k))
    for k in range(args.num_models):
        col_names.append('trans' + str(k))

    df = pd.DataFrame(trans_list, columns=col_names) 
    df.to_csv(trans_res_path)

    loss_acc_res_path = os.path.join(args.outdir, 'loss_acc_e' + str(args.adv_eps) + '_s' + str(args.seed) + '.csv')
    df = pd.DataFrame(loss_acc_list, columns=['epoch', 'train_loss', 'test_loss', 'test_acc'])
    df.to_csv(loss_acc_res_path)

if __name__ == "__main__":
    main()


