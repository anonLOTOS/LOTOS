import shutil
import time
import numpy as np

import torch
import torch.nn.functional as F
import torch.autograd as autograd
import sys, os
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.append(parentdir)
from tqdm import tqdm
import PIL.Image
from torchvision.transforms import ToTensor
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from advertorch.attacks import GradientSignAttack, LinfBasicIterativeAttack, L2PGDAttack, LinfPGDAttack, LinfMomentumIterativeAttack, \
    CarliniWagnerL2Attack, JacobianSaliencyMapAttack
from advertorch.attacks.utils import attack_whole_dataset
from models.ensemble import Ensemble
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res



def init_logfile(filename: str, text: str):
    f = open(filename, 'w')
    f.write(text+"\n")
    f.close()


def log(filename: str, text: str):
    f = open(filename, 'a')
    f.write(text+"\n")
    f.close()


def init_logfile(filename: str, text: str):
    f = open(filename, 'w')
    f.write(text+"\n")
    f.close()

def log(filename: str, text: str):
    f = open(filename, 'a')
    f.write(text+"\n")
    f.close()


def requires_grad_(model:torch.nn.Module, requires_grad:bool) -> None:
    for param in model.parameters():
        param.requires_grad_(requires_grad)


def copy_code(outdir):
    """Copies files to the outdir to store complete script with each experiment"""
    # embed()
    code = []
    exclude = set([])
    for root, _, files in os.walk("./code", topdown=True):
        for f in files:
            if not f.endswith('.py'):
                continue
            code += [(root,f)]

    for r, f in code:
        codedir = os.path.join(outdir,r)
        if not os.path.exists(codedir):
            os.mkdir(codedir)
        shutil.copy2(os.path.join(r,f), os.path.join(codedir,f))
    print("Code copied to '{}'".format(outdir))


def requires_grad_(model:torch.nn.Module, requires_grad:bool) -> None:
    for param in model.parameters():
        param.requires_grad_(requires_grad)


def Cosine(g1, g2):
    return torch.abs(F.cosine_similarity(g1, g2)).mean()  # + (0.05 * torch.sum(g1**2+g2**2,1)).mean()

def Magnitude(g1):
    return (torch.sum(g1**2,1)).mean() * 2


def gen_plot(transmat):
    import itertools
    plt.figure(figsize=(6, 6))
    plt.yticks(np.arange(0, 3, step=1))
    plt.xticks(np.arange(0, 3, step=1))
    cmp = plt.get_cmap('Blues')
    plt.imshow(transmat, interpolation='nearest', cmap=cmp, vmin=0, vmax=100.0)
    plt.title("Transfer attack success rate")
    plt.colorbar()
    thresh = 50.0
    for i, j in itertools.product(range(transmat.shape[0]), range(transmat.shape[1])):
        plt.text(j, i, "{:0.2f}".format(transmat[i, j]),
                     horizontalalignment="center",
                     color="white" if transmat[i, j] > thresh else "black")

    plt.ylabel('Target model')
    plt.xlabel('Base model')
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    return buf

def evaltrans(args, loader, models, criterion, epoch, device, writer=None):

    for i in range(len(models)):
        models[i].eval()

    cos01_losses = AverageMeter()
    cos02_losses = AverageMeter()
    cos12_losses = AverageMeter()

    """
    for _, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        batch_size = inputs.size(0)
        inputs.requires_grad = True
        grads = []
        for j in range(args.num_models):
            logits = models[j](inputs)
            loss = criterion(logits, targets)
            grad = autograd.grad(loss, inputs, create_graph=True)[0]
            grad = grad.flatten(start_dim=1)
            grads.append(grad)

        cos01 = Cosine(grads[0], grads[1])
        cos02 = Cosine(grads[0], grads[2])
        cos12 = Cosine(grads[1], grads[2])
        cos01_losses.update(cos01.item(), batch_size)
        cos02_losses.update(cos02.item(), batch_size)
        cos12_losses.update(cos12.item(), batch_size)
    """

    adv = []
    for i in range(len(models)):
        curmodel = models[i]
        adversary = LinfPGDAttack(
            curmodel, loss_fn=criterion, eps=args.adv_eps,
            nb_iter=50, eps_iter=args.adv_eps / 10, rand_init=True, clip_min=0., clip_max=1.,
            targeted=False)
        adv.append(adversary)

    trans_list = []
    trans = np.zeros((3, 3))
    for i in range(len(models)):
        # test_iter = tqdm(loader, desc='Batch', leave=False, position=2)
        # _, label, pred, advpred = attack_whole_dataset(adv[i], test_iter, device="cuda")
        _, label, pred, advpred = attack_whole_dataset(adv[i], loader, device="cuda")
        for j in range(len(models)):
            for r in range((_.size(0) - 1) // 200 + 1):
                inputc = _[r * 200: min((r + 1) * 200, _.size(0))]
                y = label[r * 200: min((r + 1) * 200, _.size(0))]
                __ = adv[j].predict(inputc)
                output = (__).max(1, keepdim=False)[1]
                trans[i][j] += (output == y).sum().item()
            trans[i][j] /= len(label)
            print(i, j, trans[i][j])
        new_tup = (epoch, trans[i,0], trans[i,1], trans[i,2])
        trans_list.append(new_tup)

    plot_buf = gen_plot((1. - trans) * 100.)
    image = PIL.Image.open(plot_buf)
    image = ToTensor()(image)

    writer.add_image('TransferImage', image, epoch)
    writer.add_scalar('test/cos01', cos01_losses.avg, epoch)
    writer.add_scalar('test/cos02', cos02_losses.avg, epoch)
    writer.add_scalar('test/cos12', cos12_losses.avg, epoch)

    return trans_list


def evaltrans_correct(args, loader, models, criterion, epoch, device, writer=None, perturb_robust=False, l2=False):
    for i in range(len(models)):
        models[i].eval()

    adv = []
    advsamples_lst = []
    pred_lst = []
    advpred_lst = []   
    label = None
    for i in range(len(models)):
        curmodel = models[i]

        if l2:
            adversary = L2PGDAttack(
                curmodel, loss_fn=criterion, eps=args.adv_eps,
                nb_iter=50, eps_iter=args.adv_eps / 10, rand_init=True, clip_min=0., clip_max=1.,
                targeted=False)
        else:
            adversary = LinfPGDAttack(
                curmodel, loss_fn=criterion, eps=args.adv_eps,
                nb_iter=50, eps_iter=args.adv_eps / 10, rand_init=True, clip_min=0., clip_max=1.,
                targeted=False)

        adv.append(adversary)
        advsamples, label, pred, advpred = attack_whole_dataset(adversary, loader, device="cuda")
        advsamples_lst.append(advsamples)
        pred_lst.append(pred)
        advpred_lst.append(advpred)

    trans_list = []
    accs = np.zeros(args.num_models)
    trans = np.zeros((args.num_models, args.num_models))
    conds_count = np.zeros((args.num_models, args.num_models))
    trans_counts = np.zeros((args.num_models, args.num_models))
    perturbation_robustness = np.zeros(args.num_models)
    for i in range(len(models)):
        _, label, pred, advpred = advsamples_lst[i], label, pred_lst[i], advpred_lst[i]

        for j in range(len(models)):
            if j==i:
                if trans[i][j] > 0.00000001:
                    continue
                y = label[label == pred]
                accs[i] = y.size(0) / label.size(0)
                conds_count[i][j] = y.size(0)
                y_wrong = label[(label == pred) & (advpred != label)]
                trans_counts[i][j] = y_wrong.size(0)
                trans[i][j] = y_wrong.size(0) / len(y)

                if perturb_robust:
                    y_pr = label[advpred == pred]
                    perturbation_robustness[i] = y_pr.size(0) / label.size(0)

            else:
                inputc = _[(label == pred) & (advpred != label) & (label==pred_lst[j])]
                print('model: ', i, inputc.size(0), ' out of ', _.size(0))
                y = label[(label == pred) & (advpred != label) & (label==pred_lst[j])]
                conds_count[i][j] = y.size(0)
            
                with torch.no_grad():
                    # for r in range((inputc.size(0) - 1) // 100 + 1):
                    #     inputc_sub = inputc[r * 200: min((r + 1) * 200, inputc.size(0))]
                    #     y_sub = y[r * 200: min((r + 1) * 200, inputc.size(0))]
                    #     __ = adv[j].predict(inputc_sub)
                    #     output = (__).max(1, keepdim=False)[1]
                    #     trans[i][j] += (output == y_sub).sum().item()

                    if inputc.size(0) > 0:
                        __ = adv[j].predict(inputc)
                        output = (__).max(1, keepdim=False)[1]
                        trans[i][j] += (output != y).sum().item()
                        trans_counts[i][j] += trans[i][j]
                        print('transfered count: ', trans[i][j], ' out of ', y.size(0))
                        trans[i][j] /= len(y)

            print(i, j, trans[i][j])

        new_list = [epoch]
        for k in range(args.num_models):
            new_list.append(trans[i,k])
        new_list.append(accs[i])
        for k in range(args.num_models):
            new_list.append(conds_count[i,k])
        for k in range(args.num_models):
            new_list.append(trans_counts[i,k])


        if perturb_robust:
            # new_tup = (epoch, trans[i,0], trans[i,1], trans[i,2], accs[i], conds_count[i,0], conds_count[i,1], conds_count[i,2], trans_counts[i,0], trans_counts[i,1], trans_counts[i,2], perturbation_robustness[i])
            new_list.append(perturbation_robustness[i])
#         else:
#             new_tup = (epoch, trans[i,0], trans[i,1], trans[i,2], accs[i], conds_count[i,0], conds_count[i,1], conds_count[i,2], trans_counts[i,0], trans_counts[i,1], trans_counts[i,2])
# 
        new_tup = tuple(new_list)
        trans_list.append(new_tup)

    # plot_buf = gen_plot((trans) * 100.)
    # image = PIL.Image.open(plot_buf)
    # image = ToTensor()(image)

    # if writer is not None:
    #     writer.add_image('TransferImage', image, epoch)

    return trans_list


def test(loader, models, criterion, epoch, device, writer=None, print_freq=10):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    # switch to eval mode
    for i in range(len(models)):
        models[i].eval()

    ensemble = Ensemble(models)
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(loader):
            # measure data loading time
            data_time.update(time.time() - end)
            inputs, targets = inputs.to(device), targets.to(device)

            # compute output
            outputs = ensemble(inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1.item(), inputs.size(0))
            top5.update(acc5.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # if i % print_freq == 0:
            #     print('Test: [{0}/{1}]\t'
            #               'Time {batch_time.avg:.3f}\t'
            #               'Data {data_time.avg:.3f}\t'
            #               'Loss {loss.avg:.4f}\t'
            #               'Acc@1 {top1.avg:.3f}\t'
            #               'Acc@5 {top5.avg:.3f}'.format(
            #             i, len(loader), batch_time=batch_time, data_time=data_time,
            #             loss=losses, top1=top1, top5=top5))

        print('Test Loss: ', losses.avg, 'acc: ', top1.avg) 

        if writer is not None:
            writer.add_scalar('loss/test', losses.avg, epoch)
            writer.add_scalar('accuracy/test@1', top1.avg, epoch)
            writer.add_scalar('accuracy/test@5', top5.avg, epoch)

    return top1.avg, losses.avg


def simple_test(loader, targets, model, device):
    model.eval()
    top1 = AverageMeter()

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(loader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            top1.update(acc1.item(), inputs.size(0))

    return top1.avg
