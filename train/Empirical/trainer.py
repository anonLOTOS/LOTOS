import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import time
import torch.nn.functional as F
import torch.autograd as autograd
import sys
import os
import numpy as np
from copy import deepcopy
# from tqdm import tqdm

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.append(parentdir)


from utils.Empirical.utils_ensemble import AverageMeter, accuracy, test, copy_code, requires_grad_
from models.ensemble import Ensemble
from utils.Empirical.utils_ensemble import Cosine, Magnitude
from utils.Empirical.third_party.distillation import Linf_distillation


def PGD(models, inputs, labels, eps):
    steps = 6
    alpha = eps / 3.

    adv = inputs.detach() + torch.FloatTensor(inputs.shape).uniform_(-eps, eps).cuda()
    adv = torch.clamp(adv, 0, 1)
    criterion = nn.CrossEntropyLoss()

    adv.requires_grad = True
    for _ in range(steps):
        #adv.requires_grad_()
        grad_loss = 0
        for i, m in enumerate(models):
            loss = criterion(m(adv), labels)
            grad = autograd.grad(loss, adv, create_graph=True)[0]
            grad = grad.flatten(start_dim=1)
            grad_loss += Magnitude(grad)

        grad_loss /= 3
        grad_loss.backward()
        sign_grad = adv.grad.data.sign()
        with torch.no_grad():
            adv.data = adv.data + alpha * sign_grad
            adv.data = torch.max(torch.min(adv.data, inputs + eps), inputs - eps)
            adv.data = torch.clamp(adv.data, 0., 1.)

    adv.grad = None
    return adv.detach()


def cos_Trainer(args, loader: DataLoader, models, criterion, optimizer: Optimizer, epoch: int, device: torch.device, writer=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_std = AverageMeter()
    # top1 = AverageMeter()
    # top5 = AverageMeter()
    cos_losses = AverageMeter()
    # smooth_losses = AverageMeter()
    cos01_losses = AverageMeter()
    cos02_losses = AverageMeter()
    cos12_losses = AverageMeter()

    end = time.time()

    for i in range(args.num_models):
        models[i].train()
        requires_grad_(models[i], True)

    for i, (inputs, targets) in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)

        inputs, targets = inputs.to(device), targets.to(device)
        batch_size = inputs.size(0)
        inputs.requires_grad = True
        grads = []
        loss_std = 0
        for j in range(args.num_models):
            logits = models[j](inputs)
            loss = criterion(logits, targets)
            grad = autograd.grad(loss, inputs, create_graph=True)[0]
            grad = grad.flatten(start_dim=1)
            grads.append(grad)
            loss_std += loss

        cos_loss, smooth_loss = 0, 0

        cos01 = Cosine(grads[0], grads[1])
        cos02 = Cosine(grads[0], grads[2])
        cos12 = Cosine(grads[1], grads[2])

        cos_loss = (cos01 + cos02 + cos12) / 3.

        """
        N = inputs.shape[0] // 2
        cureps = (args.adv_eps - args.init_eps) * epoch / args.epochs + args.init_eps
        clean_inputs = inputs[:N].detach()	# PGD(self.models, inputs[:N], targets[:N])
        adv_inputs = PGD(models, inputs[N:], targets[N:], cureps).detach()

        adv_x = torch.cat([clean_inputs, adv_inputs])

        adv_x.requires_grad = True

        if (args.plus_adv):
            for j in range(args.num_models):
                outputs = models[j](adv_x)
                loss = criterion(outputs, targets)
                grad = autograd.grad(loss, adv_x, create_graph=True)[0]
                grad = grad.flatten(start_dim=1)
                smooth_loss += Magnitude(grad)

        else:
            # grads = []
            for j in range(args.num_models):
                outputs = models[j](inputs)
                loss = criterion(outputs, targets)
                grad = autograd.grad(loss, inputs, create_graph=True)[0]
                grad = grad.flatten(start_dim=1)
                smooth_loss += Magnitude(grad)

        smooth_loss /= 3
        """


        # loss = loss_std + args.scale * (args.coeff * cos_loss + args.lamda * smooth_loss)
        # loss = loss_std + args.scale * (args.coeff * cos_loss)
        loss = loss_std + 10.* cos_loss
        # loss = loss_std + 0.5* cos_loss
        # loss = loss_std + 5.0* cos_loss


        """
        ensemble = Ensemble(models)
        logits = ensemble(inputs)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(logits, targets, topk=(1, 5))
        top1.update(acc1.item(), batch_size)
        top5.update(acc5.item(), batch_size)
        smooth_losses.update(smooth_loss.item(), batch_size)
        """

        losses_std.update(loss_std.item(), batch_size)
        losses.update(loss.item(), batch_size)
        cos01_losses.update(cos01.item(), batch_size)
        cos02_losses.update(cos02.item(), batch_size)
        cos12_losses.update(cos12.item(), batch_size)
        cos_losses.update(cos_loss.item(), batch_size)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print('Epoch: ', epoch, 'Loss: ', losses.avg, 'STD loss: ', losses_std.avg, 'Cos loss: ', cos_losses.avg)

    writer.add_scalar('train/batch_time', batch_time.avg, epoch)
    # writer.add_scalar('train/acc@1', top1.avg, epoch)
    # writer.add_scalar('train/acc@5', top5.avg, epoch)
    writer.add_scalar('train/loss', losses.avg, epoch)
    writer.add_scalar('train/std_loss', losses_std.avg, epoch)
    writer.add_scalar('train/cos_loss', cos_losses.avg, epoch)
    # writer.add_scalar('train/smooth_loss', smooth_losses.avg, epoch)
    writer.add_scalar('train/cos01', cos01_losses.avg, epoch)
    writer.add_scalar('train/cos02', cos02_losses.avg, epoch)
    writer.add_scalar('train/cos12', cos12_losses.avg, epoch)

    return losses.avg


def cos_Trainer_ortho(args, loader: DataLoader, models, criterion, optimizer: Optimizer, epoch: int, device: torch.device, writer=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_std = AverageMeter()
    cos_losses = AverageMeter()
    cos01_losses = AverageMeter()
    cos02_losses = AverageMeter()
    cos12_losses = AverageMeter()
    ortho_losses = AverageMeter()

    end = time.time()
    weights = torch.from_numpy(np.array([1 - 0.01*i for i in range(100)])).to(device)

    lsv_list_dict = {}
    for j in range(args.num_models):
        lsv_list_dict[j] = None

    for i in range(args.num_models):
        models[i].train()
        requires_grad_(models[i], True)

    for i, (inputs, targets) in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)
        inputs, targets = inputs.to(device), targets.to(device)
        batch_size = inputs.size(0)
        inputs.requires_grad = True
        grads = []
        loss_std = 0
        ortho_loss = 0
        for j in range(args.num_models):
            logits = models[j](inputs)
            loss = criterion(logits, targets)
            grad = autograd.grad(loss, inputs, create_graph=True)[0]
            grad = grad.flatten(start_dim=1)
            grads.append(grad)
            loss_std += loss

            VT_list = []
            idx = 0
            for (m_name, m) in models[j].named_modules():
                # if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)):
                if isinstance(m, (torch.nn.Conv2d)):
                    # with torch.no_grad():
                    attrs = vars(m)
                    for item in attrs.items():
                        if item[0] == '_buffers':
                            VT = item[1]['weight_VT']
                            if i != 0:# and j != 0:
                                for k in range(args.num_models):
                                    if k == j:
                                        continue
                                    prev_VT_list = lsv_list_dict[k]
                                    bad_vector = prev_VT_list[idx]
                                    item[1]['weight_BVT'] = bad_vector
                                    bad_vector = torch.nn.parameter.Parameter(data=bad_vector, requires_grad=False)
                                    op_shape = [i for i in range(1, len(bad_vector.shape))]
                                    # print('op shape: ', op_shape)
                                    # print('in: ', bad_vector.shape)
                                    # print('out: ', m(bad_vector).shape)
                                    # print('sum squared: ', torch.sum(m(bad_vector) **2, axis=op_shape).shape)
                                    # print('sqrt: ', torch.sqrt(torch.sum(m(bad_vector) **2, axis=op_shape)).shape)

                                    # bad_vec_length = torch.sqrt(torch.sum(m(bad_vector) **2))/torch.sqrt(torch.sum(bad_vector **2))
                                    bad_vec_length = torch.sqrt(torch.sum(m(bad_vector) **2, axis=op_shape))/torch.sqrt(torch.sum(bad_vector **2, axis=op_shape)) ##### fix this shit for multiple vectors!                       
                                    bad_vec_length = torch.sum(torch.mul(bad_vec_length, weights[:len(bad_vec_length)]))/torch.sum(weights[:len(bad_vec_length)])
                                    # bad_vec_length = torch.sum(bad_vec_length)
                                    # bad_vec_length = torch.sqrt(torch.sum(m(bad_vector) **2))
                                    # loss += 0.01*bad_vec_length
                                    # loss_std += 0.01*bad_vec_length
                                    ortho_loss += 0.05*bad_vec_length

                    idx += 1

                    VT_list.append(VT)
            lsv_list_dict[j] = VT_list


        cos_loss, smooth_loss = 0, 0

        cos01 = Cosine(grads[0], grads[1])
        cos02 = Cosine(grads[0], grads[2])
        cos12 = Cosine(grads[1], grads[2])

        cos_loss = (cos01 + cos02 + cos12) / 3.

        if i != 0:
            ortho_losses.update(ortho_loss.item(), batch_size)

        loss = loss_std + cos_loss + ortho_loss
        # loss = loss_std + 0.5* cos_loss
        # loss = loss_std + 5.0* cos_loss

        losses_std.update(loss_std.item(), batch_size)
        losses.update(loss.item(), batch_size)
        cos01_losses.update(cos01.item(), batch_size)
        cos02_losses.update(cos02.item(), batch_size)
        cos12_losses.update(cos12.item(), batch_size)
        cos_losses.update(cos_loss.item(), batch_size)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print('Epoch: ', epoch, 'Loss: ', losses.avg, 'STD loss: ', losses_std.avg, 'Cos loss: ', cos_losses.avg, 'Ortho loss: ', ortho_losses.avg)

    writer.add_scalar('train/batch_time', batch_time.avg, epoch)
    writer.add_scalar('train/loss', losses.avg, epoch)
    writer.add_scalar('train/std_loss', losses_std.avg, epoch)
    writer.add_scalar('train/cos_loss', cos_losses.avg, epoch)
    writer.add_scalar('train/ortho_loss', ortho_losses.avg, epoch)
    writer.add_scalar('train/cos01', cos01_losses.avg, epoch)
    writer.add_scalar('train/cos02', cos02_losses.avg, epoch)
    writer.add_scalar('train/cos12', cos12_losses.avg, epoch)

    return losses.avg

def TRS_Trainer(args, loader: DataLoader, models, criterion, optimizer: Optimizer, epoch: int, device: torch.device, writer=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    # top1 = AverageMeter()
    # top5 = AverageMeter()
    cos_losses = AverageMeter()
    losses_std = AverageMeter()
    smooth_losses = AverageMeter()
    cos01_losses = AverageMeter()
    cos02_losses = AverageMeter()
    cos12_losses = AverageMeter()

    end = time.time()

    for i in range(args.num_models):
        models[i].train()
        requires_grad_(models[i], True)

    batch_count = 0
    for i, (inputs, targets) in enumerate(loader):
        batch_count += 1
        # measure data loading time
        data_time.update(time.time() - end)

        inputs, targets = inputs.to(device), targets.to(device)
        batch_size = inputs.size(0)
        inputs.requires_grad = True
        grads = []
        loss_std = 0
        for j in range(args.num_models):
            logits = models[j](inputs)
            loss = criterion(logits, targets)
            grad = autograd.grad(loss, inputs, create_graph=True)[0]
            grad = grad.flatten(start_dim=1)
            grads.append(grad)
            loss_std += loss

        cos_loss, smooth_loss = 0, 0

        cos01 = Cosine(grads[0], grads[1])
        cos02 = Cosine(grads[0], grads[2])
        cos12 = Cosine(grads[1], grads[2])

        cos_loss = (cos01 + cos02 + cos12) / 3.


        if (args.plus_adv):
            if epoch == 0 and i == 0:
                print('full TRS method!')

            N = inputs.shape[0] // 2
            cureps = (args.adv_eps - args.init_eps) * epoch / args.epochs + args.init_eps
            clean_inputs = inputs[:N].detach()	# PGD(self.models, inputs[:N], targets[:N])
            adv_inputs = PGD(models, inputs[N:], targets[N:], cureps).detach()
            adv_x = torch.cat([clean_inputs, adv_inputs])

            adv_x.requires_grad = True
            for j in range(args.num_models):
                outputs = models[j](adv_x)
                loss = criterion(outputs, targets)
                grad = autograd.grad(loss, adv_x, create_graph=True)[0]
                grad = grad.flatten(start_dim=1)
                smooth_loss += Magnitude(grad)

        else:
            if epoch == 0 and i == 0:
                print('TRS l2 method!')
            # grads = []
            for j in range(args.num_models):
                outputs = models[j](inputs)
                loss = criterion(outputs, targets)
                grad = autograd.grad(loss, inputs, create_graph=True)[0]
                grad = grad.flatten(start_dim=1)
                smooth_loss += Magnitude(grad)

        smooth_loss /= 3

        loss = loss_std + args.scale * (args.coeff * cos_loss + args.lamda * smooth_loss)

        """
        ensemble = Ensemble(models)
        logits = ensemble(inputs)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(logits, targets, topk=(1, 5))
        top1.update(acc1.item(), batch_size)
        top5.update(acc5.item(), batch_size)

        """
        losses.update(loss.item(), batch_size)
        losses_std.update(loss_std.item(), batch_size)
        cos_losses.update(cos_loss.item(), batch_size)
        smooth_losses.update(smooth_loss.item(), batch_size)
        cos01_losses.update(cos01.item(), batch_size)
        cos02_losses.update(cos02.item(), batch_size)
        cos12_losses.update(cos12.item(), batch_size)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        """
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                        'Time {batch_time.avg:.3f}\t'
                        'Data {data_time.avg:.3f}\t'
                        'Loss {loss.avg:.4f}\t'
                        'Acc@1 {top1.avg:.3f}\t'
                        'Acc@5 {top5.avg:.3f}'.format(
                epoch, i, len(loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))
        """

    print('Epoch: ', epoch, 'loss: ', losses.avg, 'loss std: ', losses_std.avg, 'cos_loss: ', cos_losses.avg, 'smooth_loss: ', smooth_losses.avg, 'time: ', batch_time.avg * batch_count)

    writer.add_scalar('train/batch_time', batch_time.avg, epoch)
    # writer.add_scalar('train/acc@1', top1.avg, epoch)
    # writer.add_scalar('train/acc@5', top5.avg, epoch)
    writer.add_scalar('train/loss', losses.avg, epoch)
    writer.add_scalar('train/std_loss', losses_std.avg, epoch)
    writer.add_scalar('train/cos_loss', cos_losses.avg, epoch)
    writer.add_scalar('train/smooth_loss', smooth_losses.avg, epoch)
    writer.add_scalar('train/cos01', cos01_losses.avg, epoch)
    writer.add_scalar('train/cos02', cos02_losses.avg, epoch)
    writer.add_scalar('train/cos12', cos12_losses.avg, epoch)

    return losses.avg


def Naive_Trainer_ortho_backup(args, loader: DataLoader, models, criterion, optimizer: Optimizer, epoch: int, device: torch.device, writer=None, batch_counter=0, layer_1_only=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    reg_losses = AverageMeter()
    ortho_losses = AverageMeter()

    end = time.time()

    factor = args.factor
    decrement = args.decrement
    weights = torch.from_numpy(np.array([1 - decrement*i for i in range(100)])).to(device)

    lsv_list_dict = {}
    for j in range(args.num_models):
        lsv_list_dict[j] = None

    for i in range(args.num_models):
        models[i].train()
        requires_grad_(models[i], True)


    for i, (inputs, targets) in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)
        inputs, targets = inputs.to(device), targets.to(device)
        batch_size = inputs.size(0)
        inputs.requires_grad = True
        loss_std = 0
        ortho_loss = 0
        for j in range(args.num_models):
            logits = models[j](inputs)
            loss = criterion(logits, targets)
            loss_std += loss

            VT_list = []
            idx = 0

            for (m_name, m) in models[j].named_modules():
                if isinstance(m, (torch.nn.Conv2d)):
                    # with torch.no_grad():
                    attrs = vars(m)
                    for item in attrs.items():
                        if item[0] == '_buffers':
                            VT = item[1]['weight_VT']
                            if i != 0:# and j != 0:
                                first_flag = True
                                for k in range(args.num_models):
                                    if k == j:
                                        if i == 1:
                                            prev_VT = lsv_list_dict[k]
                                            sing_vector = prev_VT[idx]
                                            sing_vector = torch.nn.parameter.Parameter(data=sing_vector, requires_grad=False)
                                            op_shape = [i for i in range(1, len(sing_vector.shape))]
                                            # lsv_check = torch.sqrt(torch.sum(m(sing_vector) **2, axis=op_shape))/torch.sqrt(torch.sum(sing_vector **2, axis=op_shape)) 
                                            # print('model: ', j, 'layer: ', idx, 'lsv check: ', lsv_check)
                                        continue

                                    prev_VT_list = lsv_list_dict[k]

                                    if layer_1_only and idx > 0:
                                        continue

                                    bad_vector = prev_VT_list[idx]
                                    if first_flag:
                                        first_flag = False
                                        item[1]['weight_BVT1'] = bad_vector
                                    else:
                                        item[1]['weight_BVT2'] = bad_vector


                                    bad_vector = torch.nn.parameter.Parameter(data=bad_vector, requires_grad=False)
                                    op_shape = [i for i in range(1, len(bad_vector.shape))]
                                    # print('op shape: ', op_shape)
                                    # print('in: ', bad_vector.shape)
                                    # print('out: ', m(bad_vector).shape)
                                    # print('sum squared: ', torch.sum(m(bad_vector) **2, axis=op_shape).shape)
                                    # print('sqrt: ', torch.sqrt(torch.sum(m(bad_vector) **2, axis=op_shape)).shape)

                                    # bad_vec_length = torch.sqrt(torch.sum(m(bad_vector) **2))/torch.sqrt(torch.sum(bad_vector **2))
                                    bad_vec_length = torch.sqrt(torch.sum(m(bad_vector) **2, axis=op_shape))/torch.sqrt(torch.sum(bad_vector **2, axis=op_shape)) ##### fix this shit for multiple vectors!                       
                                    # bad_vec_length = torch.nn.ReLU(bad_vec_length-args.bottom_clip)
                                    # bad_vec_length = torch.clamp(bad_vec_length-args.bottom_clip, min=0.)
                                    bad_vec_length = torch.nn.functional.relu(bad_vec_length-args.bottom_clip)

                                    # if i == 1:
                                    #     if epoch == 0:
                                    #         print('model: ', j, 'layer: ', idx, 'bad vec shape: ', bad_vector.shape)
                                    #         print('model: ', j, 'layer: ', idx, 'op_shape: ', op_shape)
                                    #     print('model: ', j, 'layer: ', idx, 'bad vec length: ', bad_vec_length)
                                    #     print('model: ', j, 'layer: ', idx, 'weighted: ', torch.mul(bad_vec_length, weights[:len(bad_vec_length)]))
                                    bad_vec_length = torch.sum(torch.mul(bad_vec_length, weights[:len(bad_vec_length)]))/torch.sum(weights[:len(bad_vec_length)])
                                    # bad_vec_length = torch.sum(bad_vec_length)
                                    # bad_vec_length = torch.sqrt(torch.sum(m(bad_vector) **2))
                                    # loss += 0.01*bad_vec_length
                                    # ortho_loss += 0.05*bad_vec_length
                                    ortho_loss += factor* bad_vec_length

                    idx += 1

                    VT_list.append(VT)
            lsv_list_dict[j] = VT_list

        reg_losses.update(loss_std.item(), batch_size)
        if i != 0:
            ortho_losses.update(ortho_loss.item(), batch_size)

        # if batch_counter % 200 != 24:
        if batch_counter % 200 != 199:
            loss = loss_std + ortho_loss
        else:
            # print('----------------', batch_counter)
            loss = loss_std

        # print(batch_counter)
        losses.update(loss.item(), batch_size)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        batch_counter += 1

    # if epoch % args.print_freq == 0:
    print('Epoch: ', epoch, 'Loss: ', losses.avg, 'Loss_std: ', reg_losses.avg, 'Ortho_loss: ', ortho_losses.avg)

    writer.add_scalar('train/batch_time', batch_time.avg, epoch)
    writer.add_scalar('train/loss', losses.avg, epoch)
    writer.add_scalar('train/reg_loss', reg_losses.avg, epoch)
    writer.add_scalar('train/ortho_loss', ortho_losses.avg, epoch)

    return losses.avg, batch_counter

def Naive_Trainer_ortho(args, loader: DataLoader, models, criterion, optimizer: Optimizer, epoch: int, device: torch.device, writer=None, batch_counter=0, layer_1_only=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    reg_losses = AverageMeter()
    ortho_losses = AverageMeter()

    end = time.time()

    factor = args.factor
    ortho_flag = True
    if factor < 0.000001:
        ortho_flag = False

    decrement = args.decrement
    weights = torch.from_numpy(np.array([1 - decrement*i for i in range(100)])).to(device)

    lsv_list_dict = {}
    for j in range(args.num_models):
        lsv_list_dict[j] = None

    for i in range(args.num_models):
        models[i].train()
        requires_grad_(models[i], True)


    for i, (inputs, targets) in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)
        inputs, targets = inputs.to(device), targets.to(device)
        batch_size = inputs.size(0)
        inputs.requires_grad = True
        loss_std = 0
        ortho_loss = 0
        for j in range(args.num_models):
            logits = models[j](inputs)
            loss = criterion(logits, targets)
            loss_std += loss

            VT_list = []
            idx = 0

            if not ortho_flag:
                continue

            for (m_name, m) in models[j].named_modules():
                if isinstance(m, (torch.nn.Conv2d)):
                    # with torch.no_grad():
                    attrs = vars(m)
                    for item in attrs.items():
                        if item[0] == '_buffers':
                            VT = item[1]['weight_VT']
                            if i != 0:# and j != 0:
                                first_flag = True
                                for k in range(args.num_models):
                                    if k == j:
                                        if i == 1:
                                            prev_VT = lsv_list_dict[k]
                                            sing_vector = prev_VT[idx]
                                            sing_vector = torch.nn.parameter.Parameter(data=sing_vector, requires_grad=False)
                                            op_shape = [i for i in range(1, len(sing_vector.shape))]
                                        continue

                                    prev_VT_list = lsv_list_dict[k]

                                    if layer_1_only and idx > 0:
                                        continue

                                    bad_vector = prev_VT_list[idx]
                                    if first_flag:
                                        first_flag = False
                                        item[1]['weight_BVT1'] = bad_vector
                                    else:
                                        item[1]['weight_BVT2'] = bad_vector

                                    bad_vector = torch.nn.parameter.Parameter(data=bad_vector, requires_grad=False)
                                    op_shape = [i for i in range(1, len(bad_vector.shape))]
                                    # bad_vec_length = torch.sqrt(torch.sum(m(bad_vector) **2, axis=op_shape))/torch.sqrt(torch.sum(bad_vector **2, axis=op_shape)) ##### fix this shit for multiple vectors!                       

                                    # bad_vector = torch.rand_like(bad_vector)
                                    print(m(torch.zeros_like(bad_vector)))
                                    bad_vec_length = torch.sqrt(torch.sum((m(bad_vector) - m(torch.zeros_like(bad_vector)) )**2, axis=op_shape))/torch.sqrt(torch.sum(bad_vector **2, axis=op_shape)) ##### fix this shit for multiple vectors!                       
                                    bad_vec_length = torch.nn.functional.relu(bad_vec_length-args.bottom_clip)

                                    bad_vec_length = torch.sum(torch.mul(bad_vec_length, weights[:len(bad_vec_length)]))/torch.sum(weights[:len(bad_vec_length)])

                                    ortho_loss += factor* bad_vec_length

                    idx += 1

                    VT_list.append(VT)

            if ortho_flag:
                lsv_list_dict[j] = VT_list

        reg_losses.update(loss_std.item(), batch_size)
        if i != 0 and ortho_flag:
            ortho_losses.update(ortho_loss.item(), batch_size)

        if batch_counter % 200 != 199 and ortho_flag:
            loss = loss_std + ortho_loss
        else:
            loss = loss_std

        losses.update(loss.item(), batch_size)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        batch_counter += 1

    # if epoch % args.print_freq == 0:
    print('Epoch: ', epoch, 'Loss: ', losses.avg, 'Loss_std: ', reg_losses.avg, 'Ortho_loss: ', ortho_losses.avg, 'time:', batch_time.avg)

    writer.add_scalar('train/batch_time', batch_time.avg, epoch)
    writer.add_scalar('train/loss', losses.avg, epoch)
    writer.add_scalar('train/reg_loss', reg_losses.avg, epoch)
    writer.add_scalar('train/ortho_loss', ortho_losses.avg, epoch)

    return losses.avg, batch_counter


def Naive_Trainer_ortho_catclip_2(args, loader: DataLoader, models, criterion, optimizer: Optimizer, epoch: int, device: torch.device, writer=None, catclip=False, no_effect_epochs=0, batch_counter=0, mal_freq=100, conv_freq=50, layer_1_only=False, conv_1st_only=False, lsv_list_dict={}, lsv_list_dict_conv={}, conv_only=False, cos_loss_flag=False, trs_flag=False, gal_flag=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    reg_losses = AverageMeter()
    ortho_losses = AverageMeter()

    if gal_flag:
        gal_losses = AverageMeter()

    if cos_loss_flag or trs_flag:
        cos_losses = AverageMeter()
        cos01_losses = AverageMeter()
        cos02_losses = AverageMeter()
        cos12_losses = AverageMeter()

    if trs_flag:
        smooth_losses = AverageMeter()

    end = time.time()
    ortho_total = 0.

    ortho_flag = True
    if args.conv_factor < 0.000001 and args.cat_factor < 0.000001:
        ortho_flag = False

    decrement = args.decrement
    weights = torch.from_numpy(np.array([1 - decrement*i for i in range(100)])).to(device)

    print_freq = max(1000, 10*conv_freq)
    # print_freq = mal_freq

    # lsv_list_dict = {}
    # lsv_list_dict_conv = {}
    if len(lsv_list_dict) == 0:
        print('initiating lsv list dict')
        for j in range(args.num_models):
            lsv_list_dict[j] = None
            lsv_list_dict_conv[j] = None

    for i in range(args.num_models):
        models[i].train()
        requires_grad_(models[i], True)

    cat_counter_info = 0
    conv_counter_info = 0
    for i, (inputs, targets) in enumerate(loader):
        # print(i)
        # if batch_counter % print_freq == 0:
        #     print('-------------------------------------- epoch: ', epoch, ' batch: ', i)
        # measure data loading time
        data_time.update(time.time() - end)
        inputs, targets = inputs.to(device), targets.to(device)
        batch_size = inputs.size(0)
        loss_std = 0
        ortho_loss = 0
        ortho_loss_conv = 0

        grads = []
        if cos_loss_flag or gal_flag or trs_flag:
            inputs.requires_grad = True

        for j in range(args.num_models):
            logits = models[j](inputs)
            loss = criterion(logits, targets)
            
            if cos_loss_flag or gal_flag or trs_flag:
                grad = autograd.grad(loss, inputs, create_graph=True)[0]
                grad = grad.flatten(start_dim=1)
                grads.append(grad)

            loss_std += loss

            if not ortho_flag:
                continue

            if i == len(loader)-1:
                continue

            # if epoch < no_effect_epochs:
            #     continue

            VT_list = []
            VT_list_conv = []
            idx = 0

            conv_count = 0
            cat_counter_info = 0
            conv_counter_info = 0

            # if i % 100 != 0:
            #     continue
            for (m_name, m) in models[j].named_modules():
                # print(m_name)
                # print(m)

                condition = isinstance(m, (torch.nn.Conv2d))
                condition_conv = isinstance(m, (torch.nn.Conv2d))
                if not condition_conv and conv_only:
                    continue
                if catclip:
                    # condition = isinstance(m, (torch.nn.Conv2d)) or (not isinstance(m, torch.nn.BatchNorm2d) and not isinstance(m, torch.nn.Linear))
                    condition = not conv_only and not isinstance(m, (torch.nn.Conv2d)) and (not isinstance(m, torch.nn.BatchNorm2d) and not isinstance(m, torch.nn.Linear))

                if not condition_conv and epoch < no_effect_epochs:
                    conv_factor = 0.0
                    cat_factor = 0.0
                else:
                    conv_factor = args.conv_factor
                    cat_factor = args.cat_factor

                if condition or condition_conv:
                    attrs = vars(m)
                    for item in attrs.items():
                        # print('item: ', item)
                        if item[0] == '_buffers' and 'weight_VT' in item[1]:
                            if epoch == 0 and i == 0:
                            # if i == 1:
                                if condition_conv:
                                    print('model: ', j, 'conv_count: ', idx)
                                else:
                                    print('model: ', j, 'idx: ', idx)
                            # print('item buffer: ', item[0])
                            # print('item buffer vals: ', item[1].keys())
                            VT = item[1]['weight_VT']
                            # VT = getattr(m, "weight_VT")
                            # if i != 0:# or epoch == no_effect_epochs:# and j != 0:
                            if batch_counter != 0:# or epoch == no_effect_epochs:# and j != 0:
                                first_flag = True
                                for k in range(args.num_models):
                                    if k == j:
                                        # if i == 1:
                                        if condition_conv:
                                            if batch_counter % print_freq != 0:
                                                continue
                                            prev_VT = lsv_list_dict_conv[k]
                                            sing_vector = prev_VT[conv_count]
                                        else:
                                            if batch_counter % mal_freq != 0:
                                                continue
                                            prev_VT = lsv_list_dict[k]
                                            sing_vector = prev_VT[idx]

                                        sing_vector = torch.nn.parameter.Parameter(data=sing_vector, requires_grad=False)
                                        op_shape = [i for i in range(1, len(sing_vector.shape))]
                                        lsv_check = torch.sqrt(torch.sum(m(sing_vector) **2, axis=op_shape))/torch.sqrt(torch.sum(sing_vector **2, axis=op_shape)) 
                                        lsv_check_noBias = torch.sqrt(torch.sum( (m(sing_vector) - m(torch.zeros_like(sing_vector) ) )**2, axis=op_shape))/torch.sqrt(torch.sum(sing_vector **2, axis=op_shape)) 
                                        # print('model: ', j, 'layer: ', idx, 'itr: ', batch_counter, 'lsv check: ', lsv_check[0].item(), 'no bias: ', lsv_check_noBias[0].item(), 'conv:', condition_conv)
                                        # print(sing_vector.shape)
                                        # print(lsv_check_noBias)
                                        continue

                                    if condition_conv:
                                        prev_VT_list = lsv_list_dict_conv[k]
                                    else:
                                        prev_VT_list = lsv_list_dict[k]

                                    if not condition_conv and layer_1_only and idx > 0:
                                        continue

                                    if condition_conv and conv_1st_only and conv_count > 0:
                                        continue

                                    if condition_conv:
                                        bad_vector = prev_VT_list[conv_count]
                                    else:
                                        bad_vector = prev_VT_list[idx]

                                    # if first_flag:
                                    #     first_flag = False
                                    #     item[1]['weight_BVT1'] = bad_vector
                                    # else:
                                    #     item[1]['weight_BVT2'] = bad_vector


                                    if batch_counter % mal_freq != 0 and not condition_conv:
                                        continue
                                    
                                    if batch_counter % conv_freq != 0 and condition_conv:
                                        continue

                                    bad_vector = torch.nn.parameter.Parameter(data=bad_vector, requires_grad=False)
                                    op_shape = [i for i in range(1, len(bad_vector.shape))]
                                    bad_vec_length = torch.sqrt(torch.sum((m(bad_vector) - m(torch.zeros_like(bad_vector)) )**2, axis=op_shape))/torch.sqrt(torch.sum(bad_vector **2, axis=op_shape)) ##### fix this shit for multiple vectors!                       

                                    if condition_conv:
                                        bad_vec_length_thresh = torch.nn.functional.relu(bad_vec_length-args.bottom_clip)
                                    else:
                                        bad_vec_length_thresh = torch.nn.functional.relu(bad_vec_length-args.cat_bottom_clip)

                                    bad_vec_length_weighted = torch.sum(torch.mul(bad_vec_length_thresh, weights[:len(bad_vec_length_thresh)]))/torch.sum(weights[:len(bad_vec_length_thresh)])

                                    if condition_conv:
                                        # if batch_counter % print_freq == 0:
                                        #     print('conv bad vec length: ', bad_vec_length.item(), bad_vec_length_thresh.item(), 'model ', j, 'for: ', k, 'layer ', conv_count)
                                        ortho_loss_conv += conv_factor * bad_vec_length_weighted
                                        conv_counter_info += 1
                                    if not condition_conv:
                                        # if batch_counter % mal_freq == 0:
                                        #     print('concat bad vec length: ', bad_vec_length.item(), bad_vec_length_thresh.item(), 'model ', j, 'for: ', k, 'layer ', idx)
                                        ortho_loss += cat_factor * bad_vec_length_weighted
                                        cat_counter_info += 1

                            if condition_conv:
                                VT_list_conv.append(VT.detach())
                                conv_count += 1
                            else:
                                VT_list.append(VT.detach())
                                idx += 1
                            

            lsv_list_dict_conv[j] = VT_list_conv
            if not conv_only:
                lsv_list_dict[j] = VT_list

        reg_losses.update(loss_std.item(), batch_size)
        # if i != 0 and epoch >= no_effect_epochs:
        #     ortho_losses.update(ortho_loss.item(), batch_size)
        loss = loss_std


        if gal_flag:
            cos_sim = []
            for ii in range(len(models)):
                for j in range(ii + 1, len(models)):
                        cos_sim.append(F.cosine_similarity(grads[ii], grads[j], dim=-1))

            cos_sim = torch.stack(cos_sim, dim=-1)
            gal_loss = torch.log(cos_sim.exp().sum(dim=-1) + 1e-20).mean()

            loss += 0.5 * gal_loss


        cos_loss, smooth_loss = 0, 0

        if cos_loss_flag or trs_flag:
            cos01 = Cosine(grads[0], grads[1])
            cos02 = Cosine(grads[0], grads[2])
            cos12 = Cosine(grads[1], grads[2])
            cos_loss = (cos01 + cos02 + cos12) / 3.

            if trs_flag:
                loss += args.scale * args.coeff * cos_loss
            else:
                loss += 0.5 * cos_loss
                # loss += cos_loss
        
        if trs_flag and (batch_counter % conv_freq) != 0:
            if (args.plus_adv):
                # if epoch == 0 and i == 0:
                if i == 0:
                    print('full TRS method!')

                N = inputs.shape[0] // 2
                cureps = (args.trs_adv_eps - args.init_eps) * epoch / args.epochs + args.init_eps
                clean_inputs = inputs[:N].detach()	# PGD(self.models, inputs[:N], targets[:N])
                adv_inputs = PGD(models, inputs[N:], targets[N:], cureps).detach()
                adv_x = torch.cat([clean_inputs, adv_inputs])

                adv_x.requires_grad = True
                for j in range(args.num_models):
                    outputs = models[j](adv_x)
                    loss = criterion(outputs, targets)
                    grad = autograd.grad(loss, adv_x, create_graph=True)[0]
                    grad = grad.flatten(start_dim=1)
                    smooth_loss += Magnitude(grad)

            else:
                # if epoch == 0 and i == 0:
                if i == 0:
                    print('TRS l2 method!')
                # grads = []
                for j in range(args.num_models):
                    outputs = models[j](inputs)
                    loss = criterion(outputs, targets)
                    grad = autograd.grad(loss, inputs, create_graph=True)[0]
                    grad = grad.flatten(start_dim=1)
                    smooth_loss += Magnitude(grad)

            smooth_loss /= 3

            loss += args.scale * args.lamda * smooth_loss


        if ortho_flag:
            pair_count = args.num_models * (args.num_models - 1) / 2
            conv_counter_info = conv_counter_info // (args.num_models-1)
            cat_counter_info = cat_counter_info // (args.num_models-1)
            
            conv_normalizer = conv_counter_info*pair_count
            cat_normalizer = cat_counter_info*pair_count

            conv_normalizer = 1.
            cat_normalizer = 1.

            if ortho_loss_conv > 0 and (batch_counter % 200 != 199) and batch_counter > 0 and conv_counter_info > 0 and (batch_counter % conv_freq) == 0:
                loss += ortho_loss_conv / conv_normalizer #/ (conv_counter_info*pair_count)
                ortho_total += ortho_loss_conv.item() / conv_normalizer #/ (conv_counter_info*pair_count)
                if batch_counter % print_freq == 0:
                    print('pairs', pair_count,  'conv', conv_counter_info,  'ortho loss conv: ', ortho_loss_conv.item()/ conv_normalizer )

            # if batch_counter % 200 != 199 and not conv_only and batch_counter > 0 and cat_counter_info > 0 and (batch_counter % mal_freq) == 50:
            if not conv_only and batch_counter > 0 and cat_counter_info > 0 and (batch_counter % mal_freq) == 50:
                loss += ortho_loss / cat_normalizer #/ (cat_counter_info*pair_count)
                ortho_total += ortho_loss.item() / cat_normalizer #/ (cat_counter_info*pair_count)
                if batch_counter % mal_freq == 0:
                    print('pairs', pair_count, 'cat', cat_counter_info, 'ortho loss cat: ', ortho_loss.item() / cat_normalizer)
        

        losses.update(loss.item(), batch_size)

        if cos_loss_flag or trs_flag:
            cos01_losses.update(cos01.item(), batch_size)
            cos02_losses.update(cos02.item(), batch_size)
            cos12_losses.update(cos12.item(), batch_size)
            cos_losses.update(cos_loss.item(), batch_size)

        if trs_flag and smooth_loss > 0:
            smooth_losses.update(smooth_loss.item(), batch_size)

        if gal_flag:
            gal_losses.update(gal_loss.item(), batch_size)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
    
        batch_counter += 1

    # if epoch % args.print_freq == 0:
    print('Epoch: ', epoch, 'Loss: ', losses.avg, 'Loss_std: ', reg_losses.avg, 'Ortho_loss: ', ortho_losses.avg, 'ortho total:', ortho_total)

    writer.add_scalar('train/batch_time', batch_time.avg, epoch)
    writer.add_scalar('train/loss', losses.avg, epoch)
    writer.add_scalar('train/reg_loss', reg_losses.avg, epoch)
    writer.add_scalar('train/ortho_loss', ortho_losses.avg, epoch)

    return losses.avg, batch_counter, lsv_list_dict, lsv_list_dict_conv


def Naive_Trainer_ortho_catclip(args, loader: DataLoader, models, criterion, optimizer: Optimizer, epoch: int, device: torch.device, writer=None, catclip=False, no_effect_epochs=0):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    reg_losses = AverageMeter()
    ortho_losses = AverageMeter()

    end = time.time()

    factor = args.factor
    decrement = args.decrement
    weights = torch.from_numpy(np.array([1 - decrement*i for i in range(100)])).to(device)

    lsv_list_dict = {}
    for j in range(args.num_models):
        lsv_list_dict[j] = None

    for i in range(args.num_models):
        models[i].train()
        requires_grad_(models[i], True)

    for i, (inputs, targets) in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)
        inputs, targets = inputs.to(device), targets.to(device)
        batch_size = inputs.size(0)
        loss_std = 0
        ortho_loss = 0

        if epoch < no_effect_epochs:
            factor = 0.0
        else:
            factor = args.factor

        for j in range(args.num_models):
            logits = models[j](inputs)
            loss = criterion(logits, targets)
            loss_std += loss


            if epoch < no_effect_epochs:
                continue

            VT_list = []
            idx = 0

            # if i % 100 != 0:
            #     continue
            for (m_name, m) in models[j].named_modules():
                # print(m_name)
                # print(m)

                condition = isinstance(m, (torch.nn.Conv2d))
                if catclip:
                    # condition = isinstance(m, (torch.nn.Conv2d)) or (not isinstance(m, torch.nn.BatchNorm2d) and not isinstance(m, torch.nn.Linear))
                    condition = not isinstance(m, (torch.nn.Conv2d)) and (not isinstance(m, torch.nn.BatchNorm2d) and not isinstance(m, torch.nn.Linear))

                if condition:
                    attrs = vars(m)
                    for item in attrs.items():
                        # print('item: ', item)
                        if item[0] == '_buffers' and 'weight_VT' in item[1]:
                            if epoch == 0 and i == 0:
                                print('model: ', j, 'idx: ', idx)
                            # print('item buffer: ', item[0])
                            # print('item buffer vals: ', item[1].keys())
                            VT = item[1]['weight_VT']
                            # VT = getattr(m, "weight_VT")
                            if i != 0:# or epoch == no_effect_epochs:# and j != 0:
                                first_flag = True
                                for k in range(args.num_models):
                                    if k == j:
                                        if i == 1:
                                            prev_VT = lsv_list_dict[k]
                                            sing_vector = prev_VT[idx]
                                            sing_vector = torch.nn.parameter.Parameter(data=sing_vector, requires_grad=False)
                                            op_shape = [i for i in range(1, len(sing_vector.shape))]
                                            # lsv_check = torch.sqrt(torch.sum(m(sing_vector) **2, axis=op_shape))/torch.sqrt(torch.sum(sing_vector **2, axis=op_shape)) 
                                            # print('model: ', j, 'layer: ', idx, 'lsv check: ', lsv_check)
                                        continue

                                    prev_VT_list = lsv_list_dict[k]
                                    bad_vector = prev_VT_list[idx]
                                    if first_flag:
                                        first_flag = False
                                        item[1]['weight_BVT1'] = bad_vector
                                    else:
                                        item[1]['weight_BVT2'] = bad_vector
                                    bad_vector = torch.nn.parameter.Parameter(data=bad_vector, requires_grad=False)
                                    op_shape = [i for i in range(1, len(bad_vector.shape))]
                                    # print('op shape: ', op_shape)
                                    # print('in: ', bad_vector.shape)
                                    # print('out: ', m(bad_vector).shape)
                                    # print('sum squared: ', torch.sum(m(bad_vector) **2, axis=op_shape).shape)
                                    # print('sqrt: ', torch.sqrt(torch.sum(m(bad_vector) **2, axis=op_shape)).shape)

                                    # bad_vec_length = torch.sqrt(torch.sum(m(bad_vector) **2))/torch.sqrt(torch.sum(bad_vector **2))
                                    bad_vec_length = torch.sqrt(torch.sum(m(bad_vector) **2, axis=op_shape))/torch.sqrt(torch.sum(bad_vector **2, axis=op_shape)) ##### fix this shit for multiple vectors!                       
                                    # bad_vec_length = torch.nn.ReLU(bad_vec_length-args.bottom_clip)
                                    # bad_vec_length = torch.clamp(bad_vec_length-args.bottom_clip, min=0.)
                                    bad_vec_length = torch.nn.functional.relu(bad_vec_length-args.bottom_clip)

                                    # if i == 1:
                                    #     if epoch == 0:
                                    #         print('model: ', j, 'layer: ', idx, 'bad vec shape: ', bad_vector.shape)
                                    #         print('model: ', j, 'layer: ', idx, 'op_shape: ', op_shape)
                                    #     print('model: ', j, 'layer: ', idx, 'bad vec length: ', bad_vec_length)
                                    #     print('model: ', j, 'layer: ', idx, 'weighted: ', torch.mul(bad_vec_length, weights[:len(bad_vec_length)]))
                                    bad_vec_length = torch.sum(torch.mul(bad_vec_length, weights[:len(bad_vec_length)]))/torch.sum(weights[:len(bad_vec_length)])
                                    # bad_vec_length = torch.sum(bad_vec_length)
                                    # bad_vec_length = torch.sqrt(torch.sum(m(bad_vector) **2))
                                    # loss += 0.01*bad_vec_length
                                    # ortho_loss += 0.05*bad_vec_length
                                    ortho_loss += factor* bad_vec_length

                            idx += 1

                            VT_list.append(VT)
            lsv_list_dict[j] = VT_list

        reg_losses.update(loss_std.item(), batch_size)
        if i != 0 and epoch >= no_effect_epochs:
            ortho_losses.update(ortho_loss.item(), batch_size)
        loss = loss_std + ortho_loss
        losses.update(loss.item(), batch_size)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    # if epoch % args.print_freq == 0:
    print('Epoch: ', epoch, 'Loss: ', losses.avg, 'Loss_std: ', reg_losses.avg, 'Ortho_loss: ', ortho_losses.avg)

    writer.add_scalar('train/batch_time', batch_time.avg, epoch)
    writer.add_scalar('train/loss', losses.avg, epoch)
    writer.add_scalar('train/reg_loss', reg_losses.avg, epoch)
    writer.add_scalar('train/ortho_loss', ortho_losses.avg, epoch)

    return losses.avg


def Naive_Trainer(args, loader: DataLoader, models, criterion, optimizer: Optimizer, epoch: int, device: torch.device, writer=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()

    for i in range(args.num_models):
        models[i].train()
        requires_grad_(models[i], True)

    # with tqdm(loader, unit="batch") as tepoch:
        # for i, (inputs, targets) in enumerate(tepoch):
    for i, (inputs, targets) in enumerate(loader):
        # tepoch.set_description(f"Epoch {epoch}")

        # measure data loading time
        data_time.update(time.time() - end)

        inputs, targets = inputs.to(device), targets.to(device)
        batch_size = inputs.size(0)
        # inputs.requires_grad = True
        # grads = []
        loss_std = 0
        for j in range(args.num_models):
            logits = models[j](inputs)
            loss = criterion(logits, targets)
            # grad = autograd.grad(loss, inputs, create_graph=True)[0]
            # grad = grad.flatten(start_dim=1)
            # grads.append(grad)
            loss_std += loss


        loss = loss_std


        """
        ensemble = Ensemble(models)
        logits = ensemble(inputs)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(logits, targets, topk=(1, 5))
        top1.update(acc1.item(), batch_size)
        top5.update(acc5.item(), batch_size)
        """

        losses.update(loss.item(), batch_size)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    # if epoch % args.print_freq == 0:
    print('Epoch: ', epoch, 'Loss: ', losses.avg)


    writer.add_scalar('train/batch_time', batch_time.avg, epoch)
    # writer.add_scalar('train/acc@1', top1.avg, epoch)
    # writer.add_scalar('train/acc@5', top5.avg, epoch)
    writer.add_scalar('train/loss', losses.avg, epoch)

    return losses.avg


def GAL_Trainer(args, loader: DataLoader, models, criterion, optimizer: Optimizer,
				epoch: int, device: torch.device, writer=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    gal_losses = AverageMeter()

    end = time.time()

    for i in range(args.num_models):
        models[i].train()
        requires_grad_(models[i], True)

    for i, (inputs, targets) in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)

        inputs, targets = inputs.to(device), targets.to(device)
        batch_size = inputs.size(0)
        inputs.requires_grad = True
        grads = []
        loss_std = 0
        for j in range(args.num_models):
            logits = models[j](inputs)
            loss = criterion(logits, targets)
            grad = autograd.grad(loss, inputs, create_graph=True)[0]
            grad = grad.flatten(start_dim=1)
            grads.append(grad)
            loss_std += loss

        cos_sim = []
        for ii in range(len(models)):
            for j in range(ii + 1, len(models)):
                    cos_sim.append(F.cosine_similarity(grads[ii], grads[j], dim=-1))

        cos_sim = torch.stack(cos_sim, dim=-1)
        gal_loss = torch.log(cos_sim.exp().sum(dim=-1) + 1e-20).mean()

        # loss = loss_std + args.coeff * gal_loss
        loss = loss_std + 0.5 * gal_loss


        ensemble = Ensemble(models)
        logits = ensemble(inputs)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(logits, targets, topk=(1, 5))
        losses.update(loss.item(), batch_size)
        top1.update(acc1.item(), batch_size)
        top5.update(acc5.item(), batch_size)
        gal_losses.update(gal_loss.item(), batch_size)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                            'Time {batch_time.avg:.3f}\t'
                            'Data {data_time.avg:.3f}\t'
                            'Loss {loss.avg:.4f}\t'
                            'Acc@1 {top1.avg:.3f}\t'
                            'Acc@5 {top5.avg:.3f}'.format(
                    epoch, i, len(loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5))

    writer.add_scalar('train/batch_time', batch_time.avg, epoch)
    writer.add_scalar('train/acc@1', top1.avg, epoch)
    writer.add_scalar('train/acc@5', top5.avg, epoch)
    writer.add_scalar('train/loss', losses.avg, epoch)
    writer.add_scalar('train/gal_loss', gal_losses.avg, epoch)


def ADP_Trainer(args, loader: DataLoader, models, criterion, optimizer: Optimizer, epoch: int, device: torch.device, writer=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    entropy_losses = AverageMeter()
    det_losses = AverageMeter()
    end = time.time()

    for i in range(args.num_models):
        models[i].train()
        requires_grad_(models[i], True)

    for i, (inputs, targets) in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)

        inputs, targets = inputs.to(device), targets.to(device)
        batch_size = inputs.size(0)
        inputs.requires_grad = True

        num_classes = 10
        y_true = torch.zeros(inputs.size(0), num_classes).cuda()
        y_true.scatter_(1, targets.view(-1, 1), 1)

        loss_std = 0
        mask_non_y_pred = []
        ensemble_probs = 0

        for j in range(args.num_models):
            outputs = models[j](inputs)
            loss_std += criterion(outputs, targets)

            # for log_det
            y_pred = F.softmax(outputs, dim=-1)
            bool_R_y_true = torch.eq(torch.ones_like(y_true) - y_true,
                                                             torch.ones_like(y_true))  # batch_size X (num_class X num_models), 2-D
            mask_non_y_pred.append(torch.masked_select(y_pred, bool_R_y_true).
                                                       reshape(-1, num_classes - 1))  # batch_size X (num_class-1) X num_models, 1-D

            # for ensemble entropy
            ensemble_probs += y_pred

        ensemble_probs = ensemble_probs / len(models)
        ensemble_entropy = torch.sum(-torch.mul(ensemble_probs, torch.log(ensemble_probs + 1e-20)),
                                                                 dim=-1).mean()

        mask_non_y_pred = torch.stack(mask_non_y_pred, dim=1)
        assert mask_non_y_pred.shape == (inputs.size(0), len(models), num_classes - 1)
        mask_non_y_pred = mask_non_y_pred / torch.norm(mask_non_y_pred, p=2, dim=-1,
                                                                                                   keepdim=True)  # batch_size X num_model X (num_class-1), 3-D
        matrix = torch.matmul(mask_non_y_pred,
                                                  mask_non_y_pred.permute(0, 2, 1))  # batch_size X num_model X num_model, 3-D
        log_det = torch.logdet(matrix + 1e-6 * torch.eye(len(models), device=matrix.device).unsqueeze(0)).mean()  # batch_size X 1, 1-D

        loss = loss_std - args.alpha * ensemble_entropy - args.beta * log_det


        ensemble = Ensemble(models)
        logits = ensemble(inputs)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(logits, targets, topk=(1, 5))
        losses.update(loss.item(), batch_size)
        top1.update(acc1.item(), batch_size)
        top5.update(acc5.item(), batch_size)
        entropy_losses.update(ensemble_entropy.item(), batch_size)
        det_losses.update(log_det.item(), batch_size)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                        'Time {batch_time.avg:.3f}\t'
                        'Data {data_time.avg:.3f}\t'
                        'Loss {loss.avg:.4f}\t'
                        'Acc@1 {top1.avg:.3f}\t'
                        'Acc@5 {top5.avg:.3f}'.format(
                epoch, i, len(loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))

    writer.add_scalar('train/batch_time', batch_time.avg, epoch)
    writer.add_scalar('train/acc@1', top1.avg, epoch)
    writer.add_scalar('train/acc@5', top5.avg, epoch)
    writer.add_scalar('train/loss', losses.avg, epoch)
    writer.add_scalar('train/entropy_loss', entropy_losses.avg, epoch)
    writer.add_scalar('train/det_loss', det_losses.avg, epoch)


def DVERGE_Trainer(args, loader, models, criterion, optimizers, epoch: int, distill_cfg, device: torch.device, writer=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    for i in range(args.num_models):
        models[i].train()
        requires_grad_(models[i], True)

    losses = [AverageMeter() for i in range(args.num_models)]

    for i, (si, sl, ti, tl) in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)

        si, sl, ti, tl = si.to(device), sl.to(device), ti.to(device), tl.to(device)
        batch_size = si.size(0)

        distilled_data_list = []
        for j in range(args.num_models):
            temp = Linf_distillation(models[j], si, ti, **distill_cfg)
            distilled_data_list.append(temp)

        for j in range(args.num_models):
            loss = 0
            for k, distilled_data in enumerate(distilled_data_list):
                if (j == k): continue
                outputs = models[j](distilled_data)
                loss += criterion(outputs, sl)

            losses[j].update(loss.item(), batch_size)
            optimizers[j].zero_grad()
            loss.backward()
            optimizers[j].step()


        """
        ensemble = Ensemble(models)
        logits = ensemble(si)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(logits, sl, topk=(1, 5))
        top1.update(acc1.item(), batch_size)
        top5.update(acc5.item(), batch_size)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        """

        """
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                            'Time {batch_time.avg:.3f}\t'
                            'Data {data_time.avg:.3f}\t'
                            'Loss {loss.avg:.4f}\t'
                            'Acc@1 {top1.avg:.3f}\t'
                            'Acc@5 {top5.avg:.3f}'.format(
                    epoch, i, len(loader), batch_time=batch_time,
                    data_time=data_time, loss=losses[0], top1=top1, top5=top5))
        """


    writer.add_scalar('train/batch_time', batch_time.avg, epoch)
    # writer.add_scalar('train/acc@1', top1.avg, epoch)
    # writer.add_scalar('train/acc@5', top5.avg, epoch)

    for i in range(args.num_models):
        writer.add_scalar('train/loss%d' % (i), losses[i].avg, epoch)

