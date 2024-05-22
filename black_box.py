import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import sys
import os
import torchvision.transforms as transforms
import random
import numpy as np
import pandas as pd
from models.resnet import ResNet18
from models.resnet_orig import ResNet18_orig
from models.dla import DLA
from models.dla_orig import DLA_orig

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.append(parentdir)

from advertorch.attacks import GradientSignAttack, LinfBasicIterativeAttack, L2PGDAttack, LinfPGDAttack, LinfMomentumIterativeAttack, \
    CarliniWagnerL2Attack, JacobianSaliencyMapAttack, ElasticNetL1Attack
from advertorch.attacks.utils import attack_whole_dataset
from utils.Empirical.utils_ensemble import evaltrans_correct
from utils.Empirical.datasets import DATASETS, get_dataset
from utils_cifar100 import get_training_dataloader, get_test_dataloader
from utils.Empirical.architectures import get_architecture
from models.ensemble import Ensemble
import matplotlib
matplotlib.use('Agg')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument("base_classifier", type=str, help="path to the source ensemble model")
parser.add_argument("--base_classifier_2", default='', type=str, help="path to the target ensemble model")
parser.add_argument('--attack_type', default='pgd')
parser.add_argument('--dataset', default='cifar', help='dataset')
parser.add_argument('--num-models', default=3, type=int)
parser.add_argument('--model', default='ResNet18', help='Deep Learning model to train')
parser.add_argument('--method', default='orig', help='clipping method (use orig for no clipping)')
parser.add_argument('--mode', default='wBN')
parser.add_argument('--adv-eps', default=0.04, type=float)
parser.add_argument('--adv-steps', default=50, type=int)
parser.add_argument('--random-start', default=1, type=int)
parser.add_argument('--coeff', default=0.1, type=float) # for jsma, cw, ela
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--choice', default='last', help='use last (for using the model saved at the lost epoch) or best (the model with the best test accuracy)')
parser.add_argument('--cat', action='store_true', help='whether the clipping has been used on the concatenated conv and batch norm layers')
parser.add_argument('--fBN', action='store_true', help='whether the batch norm for the first convolution layers exists')

args = parser.parse_args()

source_model_count = 3

def main():
    clip_flag    = False
    orig_flag    = False
    num_class    = 10
    
    print('catclip: ', args.cat)

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
        if args.choice == 'last':
            args.choice = 'exp'

    if args.method == 'orig':
        orig_flag    = True
    else:
        clip_flag    = True

    if args.dataset == 'cifar':
        print('cifar!')
        in_chan = 3
        num_classes = 10
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        testset = torchvision.datasets.CIFAR10( root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader( testset, batch_size=128, shuffle=False, num_workers=1)

    elif args.dataset == 'cifar100':
        print('cifar100!')
        in_chan = 3
        num_classes = 100
        testloader = get_test_dataloader(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761), batch_size=128, num_workers=1, shuffle=False)


    elif args.dataset == 'mnist':
        in_chan = 1
        num_classes = 10
        test_dataset = get_dataset(args.dataset, 'test')
        pin_memory = (args.dataset == "imagenet")
        testloader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch, num_workers=args.workers, pin_memory=pin_memory)

    else:
        print('not the correct dataset!')
        exit(0)

    ##############################################################################################################
    address = args.base_classifier + '/checkpoint.pth.tar'
    print('source models: ', address)
    the_epoch = 0

    s_models = []
    for i in range(source_model_count):
        if args.choice == 'last':
            checkpoint = torch.load(address + ".%d_200" % (i)) # last epoch
        elif args.choice == 'best':
            checkpoint = torch.load(address + "_best_.%d" % (i)) # last epoch
        elif args.choice == 'exp':
            if not torch.cuda.is_available():
                checkpoint = torch.load(address + ".%d_120" % (i), map_location=torch.device('cpu'))
            else:
                checkpoint = torch.load(address + ".%d_120" % (i)) # last epoch
            print(address + ".%d_120" % (i))
        else:
            print('not the correct choice!')
            exit(0) 

        the_epoch = checkpoint['epoch'] -1
        print('the epoch: ', the_epoch)

        if clip_flag:
            print('clip', bn_flag)
            if args.model == 'ResNet18':
                model = ResNet18(concat_sv=args.cat, num_classes=num_classes, in_chan=in_chan, device=device, clip=1.0, clip_flag=True, bn=bn_flag, first_bn=bn_flag and args.fBN, clip_steps=clip_steps, clip_outer=args.cat, clip_opt_iter=opt_iter, summary=False, writer=None, save_info=False)
            elif args.model == 'DLA':
                model = DLA(concat_sv=args.cat, in_chan=in_chan, num_classes=num_classes, device=device, clip=1.0,  clip_flag=True, bn=bn_flag, first_bn=bn_flag and args.fBN, clip_steps=clip_steps, clip_outer=args.cat, clip_opt_iter=opt_iter, summary=False, writer=None, save_info=False)
        elif orig_flag:
            print('orig', bn_flag)
            if args.model == 'ResNet18':
                model = ResNet18_orig(in_chan=in_chan, num_classes=num_classes, bn=bn_flag, first_bn=args.fBN, device=device)
            elif args.model == 'DLA':
                model = DLA_orig(in_chan=in_chan, num_classes=num_classes, bn=bn_flag, device=device, first_bn=args.fBN)

        # model = nn.DataParallel(model).cuda()
        model = nn.DataParallel(model).to(device)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        print('model loaded %d' % (i))
        model.eval()
        s_models.append(model)

    print ('Model loaded')

    ##############################################################################################################
    ##############################################################################################################

    t_address = args.base_classifier_2 + '/checkpoint.pth.tar'
    print('target models: ', t_address)
    the_epoch = 0

    t_clip_flag = True
    t_cat = True 

    if '_orig_' in t_address:
        t_clip_flag = False
        t_cat = False 

    if args.mode == 'noBN':
        t_cat = False
    t_attack = args.attack_type

    t_models = []
    for i in range(args.num_models):
        if args.choice == 'last':
            t_checkpoint = torch.load(t_address + ".%d_200" % (i)) # last epoch
        elif args.choice == 'best':
            t_checkpoint = torch.load(t_address + "_best_.%d" % (i)) # last epoch
        elif args.choice == 'exp':
            if not torch.cuda.is_available():
                t_checkpoint = torch.load(t_address + ".%d_120" % (i), map_location=torch.device('cpu'))
            else:
                t_checkpoint = torch.load(t_address + ".%d_120" % (i)) # last epoch
            print(t_address + ".%d_120" % (i))
        else:
            print('not the correct choice!')
            exit(0) 

        the_epoch = checkpoint['epoch'] -1
        print('the epoch: ', the_epoch)

        if t_clip_flag:
            print('clip', bn_flag)
            if args.model == 'ResNet18':
                t_model = ResNet18(concat_sv=t_cat, num_classes=num_classes, in_chan=in_chan, device=device, clip=1.0, clip_flag=True, bn=bn_flag, first_bn=bn_flag and args.fBN, clip_steps=clip_steps, clip_outer=t_cat, clip_opt_iter=opt_iter, summary=False, writer=None, save_info=False)
            elif args.model == 'DLA':
                t_model = DLA(concat_sv=args.cat, in_chan=in_chan, num_classes=num_classes, device=device, clip=1.0, clip_flag=True, bn=bn_flag, first_bn=bn_flag and args.fBN, clip_steps=clip_steps, clip_outer=t_cat, clip_opt_iter=opt_iter, summary=False, writer=None, save_info=False)
        else:
            print('orig', bn_flag)
            if args.model == 'ResNet18':
                t_model = ResNet18_orig(in_chan=in_chan, num_classes=num_classes, bn=bn_flag, first_bn=args.fBN, device=device)
            elif args.model == 'DLA':
                t_model = DLA_orig(in_chan=in_chan, num_classes=num_classes, bn=bn_flag, device=device, first_bn=args.fBN)

        # model = nn.DataParallel(model).cuda()
        t_model = nn.DataParallel(t_model).to(device)
        t_model.load_state_dict(t_checkpoint['state_dict'], strict=False)
        print('model loaded %d' % (i))
        t_model.eval()
        t_models.append(t_model)

    t_ensemble = Ensemble(t_models)
    t_ensemble.eval()

    print ('Target Model loaded')
    ##############################################################################################################
    loss_fn = nn.CrossEntropyLoss()

    ##############################################################################################################
    #### whitebox attack on target ensemble:

    test_iter = tqdm(testloader, desc='Batch', leave=False, position=2)
    if (t_attack == "pgd"):
        t_adversary = LinfPGDAttack(
            t_ensemble, loss_fn=loss_fn, eps=args.adv_eps,
            nb_iter=args.adv_steps, eps_iter=args.adv_eps / 10, rand_init=True, clip_min=0., clip_max=1.,
            targeted=False)
    elif (t_attack == "fgsm"):
        t_adversary = GradientSignAttack(
            t_ensemble, loss_fn=loss_fn, eps=args.adv_eps,
            clip_min=0., clip_max=1., targeted=False)
    elif (t_attack == "mim"):
        t_adversary = LinfMomentumIterativeAttack(
            t_ensemble, loss_fn=loss_fn, eps=args.adv_eps,
            nb_iter=args.adv_steps, eps_iter=args.adv_eps / 10, clip_min=0., clip_max=1.,
            targeted=False)
    elif (t_attack == "cw"):
        t_adversary = CarliniWagnerL2Attack(
            t_ensemble, confidence=0.1, max_iterations=1000, clip_min=0., clip_max=1.,
            targeted=False, num_classes=num_classes, binary_search_steps=1, initial_const=args.coeff)

    # t_advsamples, label, t_pred, t_advpred = attack_whole_dataset(t_adversary, test_iter, device="cuda")
    t_advsamples, label, t_pred, t_advpred = attack_whole_dataset(t_adversary, test_iter, device=device)

    both = label[(label == t_pred) & (t_advpred == label)]
    correct_pred = label[label == t_pred]
    robust_acc = 100. * both.size(0) / correct_pred.size(0)

    adv_acc = 100. * (label == t_advpred).sum().item() / len(label)

    print("Target model: %s (eps = %.2f): %.2f (Before), %.2f (After),  %.2f (Raw adv acc)" % (args.attack_type.upper(), args.adv_eps,
        100. * (label == t_pred).sum().item() / len(label),
        np.mean(robust_acc), np.mean(adv_acc)))

    ##############################################################################################################

    # test_dataset = get_dataset(args.dataset, 'test')
    # pin_memory = (args.dataset == "imagenet")
    # testloader = DataLoader(test_dataset, shuffle=False, batch_size=128, num_workers=4, pin_memory=pin_memory)


    robust_acc = []
    advsamples_lst = []
    pred_lst = []
    advpred_lst = []
    adv = []
    for i in range(source_model_count):
        ensemble = s_models[i]
        test_iter = tqdm(testloader, desc='Batch', leave=False, position=2)

        if (args.attack_type == "pgd"):
            adversary = LinfPGDAttack(
                ensemble, loss_fn=loss_fn, eps=args.adv_eps,
                nb_iter=args.adv_steps, eps_iter=args.adv_eps / 10, rand_init=True, clip_min=0., clip_max=1.,
                targeted=False)
        elif (args.attack_type == "pgdl2"):
            adversary = L2PGDAttack(
                ensemble, loss_fn=loss_fn, eps=args.adv_eps,
                nb_iter=args.adv_steps, eps_iter=args.adv_eps / 10, rand_init=True, clip_min=0., clip_max=1.,
                targeted=False)
        elif (args.attack_type == "fgsm"):
            adversary = GradientSignAttack(
                ensemble, loss_fn=loss_fn, eps=args.adv_eps,
                clip_min=0., clip_max=1., targeted=False)
        elif (args.attack_type == "mim"):
            adversary = LinfMomentumIterativeAttack(
                ensemble, loss_fn=loss_fn, eps=args.adv_eps,
                nb_iter=args.adv_steps, eps_iter=args.adv_eps / 10, clip_min=0., clip_max=1.,
                targeted=False)
        elif (args.attack_type == "bim"):
            adversary = LinfBasicIterativeAttack(
                ensemble, loss_fn=loss_fn, eps=args.adv_eps,
                nb_iter=args.adv_steps, eps_iter=args.adv_eps / args.steps, clip_min=0., clip_max=1.,
                targeted=False)
        elif (args.attack_type == "cw"):
            adversary = CarliniWagnerL2Attack(
                ensemble, confidence=0.1, max_iterations=1000, clip_min=0., clip_max=1.,
                targeted=False, num_classes=num_classes, binary_search_steps=1, initial_const=args.coeff)

        elif (args.attack_type == "ela"):
            adversary = ElasticNetL1Attack(
                ensemble, initial_const=args.coeff, confidence=0.1, max_iterations=100, clip_min=0., clip_max=1.,
                targeted=False, num_classes=num_classes
            )
        elif (args.attack_type == "jsma"):
            adversary = JacobianSaliencyMapAttack(
                ensemble, clip_min=0., clip_max=1., num_classes=num_classes, gamma=args.coeff)

        adv.append(adversary)
        # advsamples, label, pred, advpred = attack_whole_dataset(adversary, test_iter, device="cuda")
        advsamples, label, pred, advpred = attack_whole_dataset(adversary, test_iter, device=device)
        advsamples_lst.append(advsamples)
        pred_lst.append(pred)
        advpred_lst.append(advpred)

        both = label[(label == pred) & (advpred == label)]
        correct_pred = label[label == pred]
        robust_acc = 100. * both.size(0) / correct_pred.size(0)

        print("source: %s (eps = %.2f): %.2f (Before), %.2f (After)" % (args.attack_type.upper(), args.adv_eps,
            100. * (label == pred).sum().item() / len(label),
            np.mean(robust_acc)))


    trans = np.zeros(source_model_count)
    acc = np.zeros(source_model_count)
    elig_count = np.zeros(source_model_count)
    both_correct_count = np.zeros(source_model_count)
    trans_count = np.zeros(source_model_count)
    robust_acc_list = []
    perturbation_robustness = np.zeros(source_model_count)

    for i in range(source_model_count):
        _, label, pred, advpred = advsamples_lst[i], label, pred_lst[i], advpred_lst[i]
        print('>>>>>>>>>> model: ', i, 'acc: ', (label == pred).sum().item() / len(label))

        y = label[label == pred]
        acc[i] = y.size(0) / label.size(0)
        y_wrong = label[(label == pred) & (advpred != label)]
        trans_self = y_wrong.size(0) / len(y)
        robust_acc_list.append(100. * (1.-trans_self))

        y_pr = label[advpred == pred]
        perturbation_robustness[i] = y_pr.size(0) / label.size(0)

        # for j in range(args.num_models):

        inputc = _[(label == pred) & (advpred != label) & (label==t_pred)]
        print('model: ', i, inputc.size(0), ' out of ', _.size(0))
        y = label[(label == pred) & (advpred != label) & (label==t_pred)]
        both_cor = label[(label == pred) & (label==t_pred)]

        elig_count[i] = y.size(0)
        both_correct_count[i] = both_cor.size(0)
    
        with torch.no_grad():
            if inputc.size(0) > 0:
                __ = t_adversary.predict(inputc)
                output = (__).max(1, keepdim=False)[1]
                trans[i] += (output != y).sum().item()
                trans_count[i] = trans[i]
                print('transfered count: ', trans[i], ' out of ', y.size(0))
                trans[i] /= len(y)

        print(i, trans[i])

    return trans, trans_count, elig_count, acc, both_correct_count


if __name__ == '__main__':
    choise = args.choice

    trans_list, trans_counts, elig_counts, source_acc, both_correct = main()
    df = pd.DataFrame({'trans': trans_list, 'count': trans_counts, 'eligible': elig_counts, 'both_correct': both_correct, 's_acc': source_acc})

    print(args.base_classifier.split('/')[-1], ' to ', args.base_classifier_2.split('/')[-1])
    print('mean: ', np.mean(trans_list), 'std: ', np.std(trans_list))

    outdir = args.base_classifier + args.base_classifier_2.split('/')[-1] + str(args.seed) + '_blackbox_' + choise
    df.to_csv(outdir + '_' + args.attack_type + '_' + str(args.adv_eps) + '_' + str(args.coeff) +  '.csv', index=False)
