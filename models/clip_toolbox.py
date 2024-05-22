import torch
import sys
import matplotlib.pyplot as plt
import torchvision
import numpy
import copy
import torch.optim as optim
import numpy as np
from torch.profiler import profile, record_function, ProfilerActivity
# from build_convs import *

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,2,3"



def power_qr(m, x0, device='cpu', n_iters = 20, record = False, x0_out = False, quiet = False, A=None, sort=True): 
    k = x0.shape[0]
    x0.requires_grad_(True)
    if record:
        out = torch.empty(n_iters, k, device = device)
    else:
        out = torch.empty(1, k, device = device)
        r = None
    for nit in range(n_iters):
        loss = (m(x0) ** 2).sum() / 2
        loss.backward()
        with torch.no_grad():
            x = 1*x0 + x0.grad
            shape = x.shape
            x = x.view(k, -1).T
            (q, r) = torch.linalg.qr(x) 
            x0 = q.T.view(shape) # - 7*torch.eye(q.T.shape[0], device=device)
            x0.requires_grad_(True)
            if record:
                sv = (r.diagonal() - 1).abs().sqrt()
                out[nit, :] = sv 
    if not record:
        sv = (r.diagonal() - 1).abs().sqrt()
        # sv,sort_idx = torch.sort(sv, descending=True)
        # x0 = x0[sort_idx]
        out[0, :] = sv 
    else:
        if sort:
            out,_ = torch.sort(out, descending=True)

    if x0_out:
        return (x0.detach(), out)
    else:
        return out


def power_qr_shifted(m, x0, n_iters = 20, record = False, x0_out = False, quiet = False, A=None, sort=True, device='cpu'): 
    k = x0.shape[0]
    x0.requires_grad_(True)
    if record:
        out = torch.empty(n_iters, k, device = device)
    else:
        out = torch.empty(1, k, device = device)
        r = None
    for nit in range(n_iters):
        loss = (m(x0) ** 2).sum() / 2
        loss.backward()
        with torch.no_grad():
            x = 1*x0 + x0.grad
            shape = x.shape
            x = x.view(k, -1).T
            (q, r) = torch.linalg.qr(x) 
            x0 = q.T.view(shape) # - 7*torch.eye(q.T.shape[0], device=device)
            x0.requires_grad_(True)
            if record:
                sv = (r.diagonal() - 1).abs().sqrt()
                out[nit, :] = sv 
    if not record:
        sv = (r.diagonal() - 1).abs().sqrt()
        # sv = (r.diagonal()).abs().sqrt()
        # sv,sort_idx = torch.sort(sv, descending=True)
        # x0 = x0[sort_idx]
        out[0, :] = sv 
    else:
        if sort:
            out,_ = torch.sort(out, descending=True)

    if x0_out:
        return (x0.detach(), out)
    else:
        return out


def implicit_linear_systems(model, target, init_param, nit=100, step_size=0.0001):
    model.eval()
    loss = 0.
    for itr in range(nit):
        # print(model(init_param).shape)
        loss = ((target-model(init_param)) ** 2).sum()/np.sum(target.shape)
        if itr % 50 == 0:
            print(loss)
        loss.backward()
        with torch.no_grad():
            init_param -= init_param.grad *step_size

    
        init_param.grad.zero_()
        model.zero_grad()
    print(loss)
         
    return init_param


def power_qr_ritz(m, x0, n_iters = 20, record = False, x0_out = False, quiet = False, A=None, sort=True, device='cpu'): 
    k = x0.shape[0]
    x0.requires_grad_(True)
    if record:
        out = torch.empty(n_iters, k, device = device)
    else:
        out = torch.empty(1, k, device = device)
        r = None
    for nit in range(n_iters):
        loss = (m(x0) ** 2).sum() / 2
        loss.backward()
        with torch.no_grad():
            x = 7*x0 + x0.grad
            shape = x.shape
            x = x.view(k, -1).T
            (q, r) = torch.linalg.qr(x) 
            x0 = q.T.view(shape) # - 7*torch.eye(q.T.shape[0], device=device)
            x0.requires_grad_(True)
            if record:
                sv = (r.diagonal() - 1).abs().sqrt()
                out[nit, :] = sv 
    if not record:
        sv = (r.diagonal() - 7).abs().sqrt()
        # sv,sort_idx = torch.sort(sv, descending=True)
        # x0 = x0[sort_idx]
        out[0, :] = sv 
    else:
        if sort:
            out,_ = torch.sort(out, descending=True)

    if x0_out:
        return (x0.detach(), out)
    else:
        return out



def clip_conv1d_model(conv_model, k, n, n_iters=500, device='cuda', same=False, clip=None, radius=1.):
    sh = conv_model.weight.shape
    x_batch = torch.randn(k, sh[1], n, device=device)
    VT, out = power_qr(lambda x: conv_model(x) - conv_model(torch.zeros_like(x)), torch.randn_like(x_batch), n_iters=n_iters, record=False, quiet=True, x0_out=True, device=device)

    # D_powerqr, _ = torch.sort(out[-1], descending=True)
    D_powerqr = out[-1]
    # print('D_powerqr: ', D_powerqr)

    if same:
        if clip is not None:
            new_D_powerqr = torch.clamp(D_powerqr, max=clip)
            # print('new_D_powerqr: ', new_D_powerqr)
        else:
            new_D_powerqr = D_powerqr

        return lambda x: conv_model((VT.view(k,sh[1]*n).T @ torch.diag(new_D_powerqr/D_powerqr) @ VT.view(k,sh[1]*n) @ x.view(-1,sh[1]*n).T).T.view(-1, sh[1], n)) - conv_model(torch.zeros_like(x))

    else:
        D_powerqr = D_powerqr / radius

        return lambda x: conv_model((VT.view(k,sh[1]*n).T @ torch.diag(1./D_powerqr) @ VT.view(k,sh[1]*n) @ x.view(-1,sh[1]*n).T).T.view(-1, sh[1], n)) - conv_model(torch.zeros_like(x))


def clip_conv2d_model_noPowerQR_func(conv_model, VT, D_powerqr, k, n, n_iters=500, device='cuda', clip=None):
    sh = conv_model.weight.shape

    if clip is not None:
        new_D_powerqr = torch.clamp(D_powerqr, max=clip)
    else:
        new_D_powerqr = D_powerqr

    return lambda x: conv_model(torch.reshape((torch.reshape(VT, (k, sh[1] * n**2)).T @ torch.diag(new_D_powerqr/D_powerqr) @ torch.reshape(VT, (k, sh[1] * n**2)) @ x.view(-1, sh[1] * n**2).T).T, (-1, sh[1], n, n))) - conv_model(torch.zeros_like(x))


def deflate_conv1d_model_noPowerQR(conv_model, VT, D_powerqr, n, n_iters=500, device='cuda', clip=None):
    sh = conv_model.weight.shape

    if clip is not None:
        new_D_powerqr = torch.clamp(D_powerqr, max=clip)
    else:
        new_D_powerqr = D_powerqr
    print(new_D_powerqr)

    # return lambda x: conv_model(torch.reshape((torch.reshape(VT, (k, sh[1] * n**2)).T @ torch.diag(new_D_powerqr/D_powerqr) @ torch.reshape(VT, (k, sh[1] * n**2)) @ x.view(-1, sh[1] * n**2).T).T, (-1, sh[1], n, n))) - conv_model(torch.zeros_like(x))
    # return lambda x: conv_model((VT.view(k, sh[1] * n**2).T @ torch.diag(new_D_powerqr/D_powerqr) @ (VT.view(k, sh[1] * n**2) @ x.view(-1, sh[1] * n**2).T)).T.view(-1, sh[1], n, n)) - conv_model(torch.zeros_like(x))
    return lambda x: conv_model(((new_D_powerqr/D_powerqr) * VT.view(1,sh[1]*n) @ x.view(-1,sh[1]*n).T).T.view(-1, sh[1], n)) - conv_model(torch.zeros_like(x))


def clip_conv1d_model_noPowerQR(conv_model, VT, D_powerqr, k, n, n_iters=500, device='cuda', clip=None):
    sh = conv_model.weight.shape

    if clip is not None:
        new_D_powerqr = torch.clamp(D_powerqr, max=clip)
    else:
        new_D_powerqr = D_powerqr

    # return lambda x: conv_model(torch.reshape((torch.reshape(VT, (k, sh[1] * n**2)).T @ torch.diag(new_D_powerqr/D_powerqr) @ torch.reshape(VT, (k, sh[1] * n**2)) @ x.view(-1, sh[1] * n**2).T).T, (-1, sh[1], n, n))) - conv_model(torch.zeros_like(x))
    # return lambda x: conv_model((VT.view(k, sh[1] * n**2).T @ torch.diag(new_D_powerqr/D_powerqr) @ (VT.view(k, sh[1] * n**2) @ x.view(-1, sh[1] * n**2).T)).T.view(-1, sh[1], n, n)) - conv_model(torch.zeros_like(x))
    return lambda x: conv_model(((VT.view(k,sh[1]*n).T * (new_D_powerqr/D_powerqr)) @ VT.view(k,sh[1]*n) @ x.view(-1,sh[1]*n).T).T.view(-1, sh[1], n)) - conv_model(torch.zeros_like(x))
    # return lambda x: conv_model(((VT.view(k, sh[1] * n**2).T * (new_D_powerqr/D_powerqr)) @ (VT.view(k, sh[1] * n**2) @ x.view(-1, sh[1] * n**2).T)).T.view(-1, sh[1], n, n)) - conv_model(torch.zeros_like(x))


def clip_conv2d_model_noPowerQR(conv_model, VT, D_powerqr, k, n, n_iters=500, device='cuda', clip=None):
    sh = conv_model.weight.shape

    if clip is not None:
        new_D_powerqr = torch.clamp(D_powerqr, max=clip)
    else:
        new_D_powerqr = D_powerqr

    # return lambda x: conv_model(torch.reshape((torch.reshape(VT, (k, sh[1] * n**2)).T @ torch.diag(new_D_powerqr/D_powerqr) @ torch.reshape(VT, (k, sh[1] * n**2)) @ x.view(-1, sh[1] * n**2).T).T, (-1, sh[1], n, n))) - conv_model(torch.zeros_like(x))
    # return lambda x: conv_model((VT.view(k, sh[1] * n**2).T @ torch.diag(new_D_powerqr/D_powerqr) @ (VT.view(k, sh[1] * n**2) @ x.view(-1, sh[1] * n**2).T)).T.view(-1, sh[1], n, n)) - conv_model(torch.zeros_like(x))
    return lambda x: conv_model(((VT.view(k, sh[1] * n**2).T * (new_D_powerqr/D_powerqr)) @ (VT.view(k, sh[1] * n**2) @ x.view(-1, sh[1] * n**2).T)).T.view(-1, sh[1], n, n)) - conv_model(torch.zeros_like(x))


def clip_conv2d_model(conv_model, k, n, n_iters=500, device='cuda', same=False, clip=None, radius=1., x0_out=False, last_x0=None, init_SV_out=False):
    sh = conv_model.weight.shape

    if last_x0 is not None:
        x_batch = last_x0
    else:
        n_iters = 100
        x_batch = torch.randn(k, sh[1], n, n, device=device)

    # VT, out = power_qr(lambda x: conv_model(x) - conv_model(torch.zeros_like(x)), torch.randn_like(x_batch), n_iters=n_iters, record=False, quiet=True, x0_out=True)
    VT, out = power_qr(lambda x: conv_model(x) - conv_model(torch.zeros_like(x)), x_batch, n_iters=n_iters, record=False, quiet=True, x0_out=True, device=device)

    D_powerqr = out[-1]

    if same:
        if clip is not None:
            new_D_powerqr = torch.clamp(D_powerqr, max=clip)
        else:
            new_D_powerqr = D_powerqr

        if x0_out:
            if init_SV_out:
                return lambda x: conv_model(torch.reshape((torch.reshape(VT, (k, sh[1] * n**2)).T @ torch.diag(1./D_powerqr) @ torch.diag(new_D_powerqr) @ torch.reshape(VT, (k, sh[1] * n**2)) @ x.view(-1, sh[1] * n**2).T).T, (-1, sh[1], n, n))) - conv_model(torch.zeros_like(x)), VT, D_powerqr
            return lambda x: conv_model(torch.reshape((torch.reshape(VT, (k, sh[1] * n**2)).T @ torch.diag(1./D_powerqr) @ torch.diag(new_D_powerqr) @ torch.reshape(VT, (k, sh[1] * n**2)) @ x.view(-1, sh[1] * n**2).T).T, (-1, sh[1], n, n))) - conv_model(torch.zeros_like(x)), VT

        if init_SV_out:
            return lambda x: conv_model(torch.reshape((torch.reshape(VT, (k, sh[1] * n**2)).T @ torch.diag(1./D_powerqr) @ torch.diag(new_D_powerqr) @ torch.reshape(VT, (k, sh[1] * n**2)) @ x.view(-1, sh[1] * n**2).T).T, (-1, sh[1], n, n))) - conv_model(torch.zeros_like(x)), D_powerqr
        return lambda x: conv_model(torch.reshape((torch.reshape(VT, (k, sh[1] * n**2)).T @ torch.diag(1./D_powerqr) @ torch.diag(new_D_powerqr) @ torch.reshape(VT, (k, sh[1] * n**2)) @ x.view(-1, sh[1] * n**2).T).T, (-1, sh[1], n, n))) - conv_model(torch.zeros_like(x))

    else:
        D_powerqr = D_powerqr / radius

        if x0_out:
            return lambda x: conv_model(torch.reshape((torch.reshape(VT, (k, sh[1] * n**2)).T @ torch.diag(1./D_powerqr) @ torch.reshape(VT, (k, sh[1] * n**2)) @ x.view(-1, sh[1] * n**2).T).T, (-1, sh[1], n, n))) - conv_model(torch.zeros_like(x)), VT
        return lambda x: conv_model(torch.reshape((torch.reshape(VT, (k, sh[1] * n**2)).T @ torch.diag(1./D_powerqr) @ torch.reshape(VT, (k, sh[1] * n**2)) @ x.view(-1, sh[1] * n**2).T).T, (-1, sh[1], n, n))) - conv_model(torch.zeros_like(x))



def linear_optimizer(m_orig, k=10, n=20, n_iters=100, epochs=20, step_size=0.01, same=False, clip=None, radius=1., rand_init=False, device='cuda'):
    m = copy.deepcopy(m_orig)

    if isinstance(m_orig, torch.nn.Linear):
        x0 = torch.randn(k, m_orig.weight.shape[1], device=device)
        (V_T, qr) = power_qr(m_orig, x0.clone(), n_iters, x0_out=True, quiet=True, device=device)
        svals = qr[-1]
        inv_sigma_mat = torch.diag(1./svals)
        new_weight = m_orig.weight.detach() @ V_T.T @ inv_sigma_mat @ V_T
        m.state_dict()['weight'].copy_(new_weight)


    if isinstance(m_orig, torch.nn.Conv1d):
        sh = m_orig.weight.shape

        if rand_init:
            new_filter = torch.randn_like(m_orig.weight)
            with torch.no_grad():
                m.weight = torch.nn.Parameter(new_filter)

        x_batch = torch.randn(k, sh[1], n, device=device)

        clipped_model = clip_conv1d_model(m_orig, k, n, n_iters=n_iters, same=same, clip=clip, radius=radius, device=device)
        
        build_conv_model(m, clipped_model, x_batch, epochs=30, step_size=0.01)


    if isinstance(m_orig, torch.nn.Conv2d):
        sh = m_orig.weight.shape

        if rand_init:
            new_filter = torch.randn_like(m_orig.weight)
            with torch.no_grad():
                m.weight = torch.nn.Parameter(new_filter)

        x_batch = torch.randn(k, sh[1], n, n, device=device)

        clipped_model = clip_conv2d_model(m_orig, k, n, n_iters=n_iters, same=same, clip=clip, radius=radius, device=device)

        build_conv_model(m, clipped_model, x_batch, epochs=30, step_size=0.01)

    return m


def build_conv_model_new(conv_model_trained, conv_model, data, VT, D_powerqr, k, n, clip=1., rand_data=False, epochs=1, early_stop=False, step_size=0.01, quiet=True, device='cuda'):
    loss_list = []
    min_loss = 100000000 # torch.Tensor(float("Inf")) # 100000000
    sh = conv_model.weight.shape

    data_new = data
    for epoch in range(epochs):
        if rand_data:
            data_new = torch.rand_like(data)

        conv_model_trained.zero_grad()

        new_D_powerqr = torch.clamp(D_powerqr, max=clip)

        if isinstance(conv_model_trained, torch.nn.Conv1d):
            loss = ((conv_model(torch.reshape((torch.reshape(VT, (k, sh[1] * n**2)).T @ torch.diag(new_D_powerqr/D_powerqr) @ torch.reshape(VT, (k, sh[1] * n**2)) @ data_new.view(-1, sh[1] * n**2).T).T, (-1, sh[1], n, n))) - conv_model(torch.zeros_like(data_new)) - conv_model_trained(data_new) + conv_model_trained(torch.zeros_like(data_new)))**2).sum((1,2)).mean()
        if isinstance(conv_model_trained, torch.nn.Conv2d):
            loss = ((conv_model(torch.reshape((torch.reshape(VT, (k, sh[1] * n**2)).T @ torch.diag(new_D_powerqr/D_powerqr) @ torch.reshape(VT, (k, sh[1] * n**2)) @ data_new.view(-1, sh[1] * n**2).T).T, (-1, sh[1], n, n))) - conv_model(torch.zeros_like(data_new)) - conv_model_trained(data_new) + conv_model_trained(torch.zeros_like(data_new)))**2).sum((1,2)).mean()

        loss.backward()


        loss_list.append(loss.item())
        if not quiet:
            if epoch % 50 == 0:
                print(loss.item())

        if loss.item() < min_loss: 
            min_loss = loss.item()
        else:
            if early_stop:
                print('not decreasing anymore!')
                break

        with torch.no_grad():
            for P, tup in zip(conv_model_trained.parameters(), conv_model_trained.named_parameters()):
                if tup[0] == 'weight':
                    P -= step_size * P.grad


    conv_model_trained.zero_grad()

    return loss_list



def build_linear_model(linear_model, svals, VT, clip=1., device='cuda'):
    new_weight = (linear_model.weight.detach() @ (VT.T * (clip/svals))) @ VT
    return new_weight
    

def build_conv_model(conv_model_trained, conv_model_clipped, data, rand_data=False, epochs=1, early_stop=False, step_size=0.01, quiet=True, device='cuda'):
    loss_list = []
    min_loss = 100000000 # torch.Tensor(float("Inf")) # 100000000

    data_new = data
    for epoch in range(epochs):
        if rand_data:
            data_new = torch.rand_like(data)

        conv_model_trained.zero_grad()

        if isinstance(conv_model_trained, torch.nn.Conv1d):
            loss = ((conv_model_clipped(data_new) - conv_model_trained(data_new) + conv_model_trained(torch.zeros_like(data_new)))**2).sum((1,2)).mean()
        if isinstance(conv_model_trained, torch.nn.Conv2d):
            loss = ((conv_model_clipped(data_new) - conv_model_trained(data_new) + conv_model_trained(torch.zeros_like(data_new)))**2).sum((1,2)).mean()

        loss.backward()

        loss_list.append(loss.item())
        if not quiet:
            if epoch % 50 == 0:
                print(loss.item())

        if loss.item() < min_loss: 
            min_loss = loss.item()
        else:
            if early_stop:
                print('not decreasing anymore!')
                break

        with torch.no_grad():
            for P, tup in zip(conv_model_trained.parameters(), conv_model_trained.named_parameters()):
                if tup[0] == 'weight':
                    P -= step_size * P.grad

        # del loss, dec_part, inc_part

    conv_model_trained.zero_grad()

    return loss_list


def deflate_model(conv_model_trained, conv_model_clipped, data, SV, clip=1., rand_data=False, epochs=1, early_stop=False, step_size=0.01, quiet=True, device='cuda'):
    loss_list = []
    min_loss = 100000000 # torch.Tensor(float("Inf")) # 100000000

    data_new = data
    for epoch in range(epochs):
        if rand_data:
            data_new = torch.rand_like(data)

        conv_model_trained.zero_grad()

        if isinstance(conv_model_trained, torch.nn.Linear):
            loss = ((conv_model_clipped((clip/SV) * data_new) - conv_model_clipped((clip/SV) * torch.zeros_like(data_new)) - conv_model_trained(data_new) + conv_model_trained(torch.zeros_like(data_new)))**2).sum((1)).mean()
        elif isinstance(conv_model_trained, torch.nn.Conv1d):
            loss = ((conv_model_clipped((clip/SV) * data_new) - conv_model_clipped((clip/SV) * torch.zeros_like(data_new)) - conv_model_trained(data_new) + conv_model_trained(torch.zeros_like(data_new)))**2).sum((1,2)).mean()
        elif isinstance(conv_model_trained, torch.nn.Conv2d):
            loss = ((conv_model_clipped((clip/SV) * data_new) - conv_model_clipped((clip/SV) * torch.zeros_like(data_new)) - conv_model_trained(data_new) + conv_model_trained(torch.zeros_like(data_new)))**2).sum((1,2)).mean()
        else:
            loss = ((conv_model_clipped((clip/SV) * data_new) - conv_model_trained(data_new))**2).sum((1,2)).mean()

        loss.backward()

        loss_list.append(loss.item())
        if not quiet:
            if epoch % 50 == 0:
                print(loss.item())

        if loss.item() < min_loss: 
            min_loss = loss.item()
        else:
            if early_stop:
                print('not decreasing anymore!')
                break

        with torch.no_grad():
            for P, tup in zip(conv_model_trained.parameters(), conv_model_trained.named_parameters()):
                # if tup[0] == 'weight':
                # if tup[0] in ['weight', 'conv.weight', 'bn.weight']:
                if tup[0].split('.')[-1] == 'weight':
                    # print(tup[0])
                    P -= step_size * P.grad

        # del loss, dec_part, inc_part

    conv_model_trained.zero_grad()

    return loss_list


def deflate_model_multi(trained_model, x0, clip=1., rand_data=True, epochs=1, deflate_iter=5, early_stop=False, pqr_iters=10, step_size=0.35, quiet=True, device='cuda'):
    # trained_model = copy.deepcopy(orig_model)
    loss_list = []
    VT = x0
    for i in range(deflate_iter):
        # print('--------> ', i)
        # x0 = torch.randn(1, num_inp_channels, n).to(device)
        # trained_model.eval()
        (VT, qr) = power_qr(lambda x: trained_model(x) - trained_model(torch.zeros_like(x)), VT.clone().detach(), n_iters=pqr_iters, x0_out=True, quiet=True, device=device)
        trained_model.zero_grad()
        # print('largest SV: ', qr[-1])
        D_powerqr = qr[-1]
        # trained_model.train()

        # if D_powerqr[0] < clip:
        if D_powerqr[0] <= clip and clip - D_powerqr[0] <= 0.05:
            break

        trained_conv_model_copy = copy.deepcopy(trained_model)
        loss_list_cur = deflate_model(trained_model, trained_conv_model_copy, VT, D_powerqr[0], clip=clip, rand_data=False, epochs=epochs, early_stop=early_stop, step_size=step_size, quiet=quiet)
        loss_list.append(loss_list_cur[-1])

        if rand_data:
            VT = torch.rand_like(VT).to(device)

    return loss_list 

    

def train_clip_conv2d(m, n, counter, last_VT=None, k=None, clip=1., init_pqr_iter=100, pqr_iter=1, clip_steps=50, clip_opt_iter=1, clip_opt_stepsize=0.01, clip_data_size=100):
    conv_model = copy.deepcopy(m)
    kernel_shape = m.weight.shape
    num_inp_channels = kernel_shape[1]
    num_out_channels = kernel_shape[0]

    if k is None:
        k = n*n*num_inp_channels

    if last_VT is not None:
        x_batch_sp = last_VT
        n_iters = pqr_iter
    else:
        n_iters = init_pqr_iter
        x_batch_sp = torch.randn(k, num_inp_channels, n, n, device=device)

    VT, out = power_qr(lambda x: conv_model(x) - conv_model(torch.zeros_like(x)), x_batch_sp, n_iters=n_iters, record=False, quiet=True, x0_out=True, device=device)
    D_powerqr = out[-1]
    # sv1_approx_list.append(D_powerqr[0].item())
    last_VT = VT

    counter += 1
    if counter % clip_steps == 0:
        counter = 0
        clipped_conv_model = clip_conv2d_model_noPowerQR(conv_model, VT, D_powerqr, k, n, n_iters=clip_opt_iter, clip=clip, device=device)

        data_size = clip_data_size
        z0 = torch.randn(data_size, num_inp_channels, n, n).to(device)
        z0.requires_grad_(False)

        # trained_conv_model = copy.deepcopy(conv_model)
        trained_conv_model = conv_model
        loss_list = build_conv_model(trained_conv_model, clipped_conv_model, z0, rand_data=False, epochs=clip_opt_iter, early_stop=False, step_size=clip_opt_stepsize, quiet=True)
        # build_loss_list += loss_list
        m.weight = copy.deepcopy(trained_conv_model.weight)

        return trained_conv_model.weight, counter, last_VT

    return None, counter, last_VT, D_powerqr[0].item(), loss_list


def build_conv1d_model():
    pass


def build_conv2d_model():
    pass



if __name__ == '__main__':
    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda:0")

    seed_val = 1
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    np.random.seed(seed_val)

    ################################################# Linear layer:
    print('\n\n\t\tProcessing linear layer...')

    M = torch.randn(10, 10)
    M = M.T@M
    svalues = M.svd()[1]

    l_layer = torch.nn.Linear(10, 10, bias=False)
    with torch.no_grad():
        l_layer.weight = torch.nn.Parameter(M)
    l_layer = l_layer.to(device)

    M = M.to(device)
    n_iters = 100
    k = 10

    x0 = torch.randn(k, M.shape[1], device=device)
    print('x0 shape: ', x0.shape)

    print("----> Weight matrix svalues:")
    print(svalues)

    print("----> Linear layer svalues:")
    (V, qr) = power_qr_shifted(l_layer, x0.clone().detach(), n_iters, x0_out=True, quiet=True)
    svals = qr[-1]
    print(svals)

    # target = torch.randn(10, M.shape[0], device=device)
    # print('target: ', target)
    # params = implicit_linear_systems(l_layer, target, torch.nn.Parameter(x0), nit=10000, step_size=0.01)
    # print('new target: ', l_layer(params))



    (VT, qr) = power_qr(lambda x: l_layer(x) - l_layer(torch.zeros_like(x)), x0.clone().detach(), n_iters=500, x0_out=True, quiet=True)
    print('orig conv layer: ') 
    svals_orig = qr[-1]
    print(svals_orig)
    print('VT shape: ', VT.shape)
    # print((VT**2).sum((1,2)))
    # print(VT.view(k+20,-1))
    VT_orig = copy.deepcopy(VT)
    print('init: ', [torch.sqrt(torch.sum(l_layer(VT_orig[i:i+1]) **2)) for i in range(5)])



    ############# new iterative deflation method:

    trained_conv_model = copy.deepcopy(l_layer)
    for i in range(10):
        print('--------> ', i)
        x0 = torch.randn(1, M.shape[1], device=device)
        (VT, qr) = power_qr(lambda x: trained_conv_model(x) - trained_conv_model(torch.zeros_like(x)), x0.clone().detach(), n_iters=500, x0_out=True, quiet=True)
        print('largest SV: ', qr[-1])
        D_powerqr = qr[-1]
        if svals_orig[i] < 10.:
            break

        # deflated_model = deflate_conv1d_model_noPowerQR(conv_model, VT, D_powerqr[0], n, n_iters=500, device='cuda', clip=10.)
        trained_conv_model_copy = copy.deepcopy(trained_conv_model)
        # loss_list = deflate_model(trained_conv_model, trained_conv_model_copy, VT, D_powerqr[0], clip=10., rand_data=False, epochs=400, early_stop=False, step_size=0.01, quiet=False)
        loss_list = deflate_model(trained_conv_model, trained_conv_model_copy, VT_orig[i:i+1], svals_orig[i], clip=10., rand_data=False, epochs=400, early_stop=False, step_size=0.01, quiet=False)
        print('length: ', [torch.sqrt(torch.sum(trained_conv_model(VT_orig[i:i+1]) **2)).item() for i in range(5)])


        x0 = torch.randn(k, M.shape[1], device=device)
        (VT, qr) = power_qr(lambda x: trained_conv_model(x) - trained_conv_model(torch.zeros_like(x)), x0.clone().detach(), n_iters=500, x0_out=True, quiet=True)
        print('complete: ', qr[-1])

    x0 = torch.randn(k, M.shape[1], device=device)
    (VT, qr) = power_qr(lambda x: trained_conv_model(x) - trained_conv_model(torch.zeros_like(x)), x0.clone().detach(), n_iters=500, x0_out=True, quiet=True)
    print('final: ', qr[-1])

    exit(0)

    data_size = 1000
    data = torch.randn(data_size, n).to(device)
    clipped_model = clip_conv1d_model_noPowerQR(conv_model, VT, D_powerqr, n*2, n, n_iters=500, device='cuda', clip=10.)
    loss_list = build_conv_model(trained_conv_model, clipped_model, data, rand_data=False, epochs=400, early_stop=False, step_size=0.01, quiet=False)

    x0 = torch.randn(2*n, num_inp_channels, n).to(device).to(device)
    (VT, qr) = power_qr(lambda x: trained_conv_model(x) - trained_conv_model(torch.zeros_like(x)), x0.clone().detach(), n_iters=500, x0_out=True, quiet=True)
    print('complete: ', qr[-1])
    exit(0)



    radius = 1.
    k = 4
    # new_l_layer = linear_optimizer(l_layer, n_iters=100, k=k, same=False, clip=None, radius=radius, device=device)

    new_l_layer = copy.deepcopy(l_layer)
    new_weight = build_linear_model(l_layer, svals, V, clip=1., device=device)
    new_l_layer.state_dict()['weight'].copy_(new_weight)

    (V, qr) = power_qr(new_l_layer, x0.clone().detach(), n_iters, x0_out=True, quiet=True)
    print('----> Linear layer clipped to be linear optimizer to ball r=' + str(radius) + ' in ' + str(k)  + ' dimensions:') 
    svals = qr[-1]
    print(svals)


    exit(0)


    

    ################################################# Conv 1d layer:

    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    np.random.seed(seed_val) 
    print('\n\n\t\tProcessing 1d Convolution...')
    n = 28
    k = 28

    num_inp_channels = 30
    num_out_channels = 20
    kernel_size = 3
    filter_shape = (num_out_channels, num_inp_channels, kernel_size)
    filter = np.random.randint(low=-8, high=8,size=filter_shape)
    transform_mat = create_conv1d_mat_mic_moc(n, filter, noPad=False)
    new_filter = torch.reshape(torch.tensor(filter), (num_out_channels, num_inp_channels, -1)).float()
    # conv_model = torch.nn.Conv1d(in_channels=num_inp_channels, out_channels=num_out_channels, kernel_size=kernel_size, padding_mode='circular', padding=1, bias=False)
    conv_model = torch.nn.Conv1d(in_channels=num_inp_channels, out_channels=num_out_channels, kernel_size=kernel_size, padding=1, bias=False)

    with torch.no_grad():
        conv_model.weight = torch.nn.Parameter(new_filter)
    conv_model = conv_model.to(device)

    x0 = torch.randn(2*n, num_inp_channels, n).to(device).to(device)


    (VT, qr) = power_qr(lambda x: conv_model(x) - conv_model(torch.zeros_like(x)), x0.clone().detach(), n_iters=500, x0_out=True, quiet=True, device=device)
    print('orig conv layer: ') 
    svals_orig = qr[-1]
    print(svals_orig)
    print('VT shape: ', VT.shape)
    print((VT**2).sum((1,2)))
    print(VT.view(k+20,-1))
    VT_orig = copy.deepcopy(VT)
    print('init: ', [torch.sqrt(torch.sum(conv_model(VT_orig[i:i+1]) **2)) for i in range(5)])



    ############# new iterative deflation method:


    trained_conv_model = copy.deepcopy(conv_model)
    x0 = torch.randn(1, num_inp_channels, n).to(device).to(device)

    loss_list = deflate_model_multi(trained_conv_model, x0, deflate_iter=200, epochs=100)
    print('length: ', [torch.sqrt(torch.sum(trained_conv_model(VT_orig[i:i+1]) **2)).item() for i in range(5)])
    print(loss_list)

    exit(0)

    for i in range(200):
        print('--------> ', i)
        x0 = torch.randn(1, num_inp_channels, n).to(device).to(device)
        (VT, qr) = power_qr(lambda x: trained_conv_model(x) - trained_conv_model(torch.zeros_like(x)), x0.clone().detach(), n_iters=500, x0_out=True, quiet=True, device=device)
        print('largest SV: ', qr[-1])
        D_powerqr = qr[-1]
        # if svals_orig[i] < 11.:
        if D_powerqr[0] < 10.:
            break

        # deflated_model = deflate_conv1d_model_noPowerQR(conv_model, VT, D_powerqr[0], n, n_iters=500, device='cuda', clip=10.)
        trained_conv_model_copy = copy.deepcopy(trained_conv_model)
        loss_list = deflate_model(trained_conv_model, trained_conv_model_copy, VT, D_powerqr[0], clip=10., rand_data=False, epochs=100, early_stop=False, step_size=0.01, quiet=False)
        # loss_list = deflate_model(trained_conv_model, trained_conv_model_copy, VT_orig[i:i+1], svals_orig[i], clip=10., rand_data=False, epochs=400, early_stop=False, step_size=0.01, quiet=False)
        print('length: ', [torch.sqrt(torch.sum(trained_conv_model(VT_orig[i:i+1]) **2)).item() for i in range(5)])


        # x0 = torch.randn(2*n, num_inp_channels, n).to(device).to(device)
        # (VT, qr) = power_qr(lambda x: trained_conv_model(x) - trained_conv_model(torch.zeros_like(x)), x0.clone().detach(), n_iters=500, x0_out=True, quiet=True)
        # print('complete: ', qr[-1])

    exit(0)

    data_size = 1000
    data = torch.randn(data_size, num_inp_channels, n).to(device)
    clipped_model = clip_conv1d_model_noPowerQR(conv_model, VT, D_powerqr, n*2, n, n_iters=500, device='cuda', clip=10.)
    loss_list = build_conv_model(trained_conv_model, clipped_model, data, rand_data=False, epochs=400, early_stop=False, step_size=0.01, quiet=False)

    x0 = torch.randn(2*n, num_inp_channels, n).to(device).to(device)
    (VT, qr) = power_qr(lambda x: trained_conv_model(x) - trained_conv_model(torch.zeros_like(x)), x0.clone().detach(), n_iters=500, x0_out=True, quiet=True)
    print('cont build: ', qr[-1])
    print('length: ', [torch.sqrt(torch.sum(trained_conv_model(VT_orig[i:i+1]) **2)).item() for i in range(5)])


    trained_conv_model = copy.deepcopy(conv_model)
    clipped_model = clip_conv1d_model_noPowerQR(conv_model, VT, D_powerqr, n*2, n, n_iters=500, device='cuda', clip=10.)
    loss_list = build_conv_model(trained_conv_model, clipped_model, data, rand_data=False, epochs=400, early_stop=False, step_size=0.01, quiet=False)
    (VT, qr) = power_qr(lambda x: trained_conv_model(x) - trained_conv_model(torch.zeros_like(x)), x0.clone().detach(), n_iters=500, x0_out=True, quiet=True)
    print('scratch build: ', qr[-1])
    print('length: ', [torch.sqrt(torch.sum(trained_conv_model(VT_orig[i:i+1]) **2)).item() for i in range(5)])
    exit(0)


    clipped_model = clip_conv1d_model_noPowerQR(conv_model, VT, D_powerqr, k, n, n_iters=500, device='cuda', clip=10.)
    (VT, qr) = power_qr(lambda x: clipped_model(x) - clipped_model(torch.zeros_like(x)), x0.clone().detach(), n_iters=500, x0_out=True, quiet=True)
    print(qr[-1])




    """
    ####################### manual conv layer:
    filter_shape = (num_out_channels, num_inp_channels, kernel_size)
    # filter = np.random.randint(low=-8, high=8,size=filter_shape)
    filter = np.zeros(filter_shape)
    filter[0,0,0] = 24.
    transform_mat_manual = create_conv1d_mat_mic_moc(n, filter, noPad=False)
    # filter = np.array([[[0,2,0],[0,0,0],[0,0,0]],
    #                    [[0,0,0],[0,0,0],[0,0,0]]])

    # filter = np.array([[[0,3,3],[1,5,0],[0,8,2]],
    #                    [[0,0,1],[0,0,1],[3,4,0]]])
    new_filter = torch.reshape(torch.tensor(filter), (num_out_channels, num_inp_channels, -1)).float()
    conv_model_manual = torch.nn.Conv1d(in_channels=num_inp_channels, out_channels=num_out_channels, kernel_size=kernel_size, padding_mode='circular', padding=1, bias=False)

    with torch.no_grad():
        conv_model_manual.weight = torch.nn.Parameter(new_filter)
    conv_model_manual = conv_model_manual.to(device)

    (V_T, qr) = power_qr(lambda x: conv_model_manual(x) - conv_model_manual(torch.zeros_like(x)), x0.clone().detach(), n_iters=500, x0_out=True, quiet=True)
    print('manual conv layer: ') 
    svals_orig = qr[-1]
    print(svals_orig)

    ################################################

    # print(transform_mat)
    # print(transform_mat_manual)
    # exit(0)
    """

    trained_conv_model = copy.deepcopy(conv_model)
    # trained_conv_model = copy.deepcopy(conv_model_manual)

    (V_T, qr) = power_qr(trained_conv_model, x0.clone().detach(), n_iters=50, x0_out=True, quiet=True)
    print('trained conv layer before: ')
    svals = qr[-1]
    print(svals)

    # clip = 2.
    clip = 10.

    clipped_conv_model = clip_conv1d_model(conv_model, k, n, same=True, clip=clip, device=device) 
    # clipped_conv_model = clip_conv1d_model(conv_model_manual, k, n, same=True, clip=clip, device=device) 

    x0 = torch.randn(n*min(num_inp_channels,num_out_channels), num_inp_channels, n).to(device).to(device)
    (V_T, qr) = power_qr(clipped_conv_model, x0.clone().detach(), n_iters=100, x0_out=True, quiet=True)
    print('clipped conv layer: ') 
    svals = qr[-1]
    print(svals)


    # print(V_T.T.shape)
    # print(V_T.T[0,...])



    # ####################### manual conv layer:
    # filter_shape = (num_out_channels, num_inp_channels, kernel_size)
    # # filter = np.random.randint(low=-8, high=8,size=filter_shape)
    # filter = np.zeros(filter_shape)
    # filter[0,0,1] = 24.
    # new_filter = torch.reshape(torch.tensor(filter), (num_out_channels, num_inp_channels, -1)).float()
    # conv_model = torch.nn.Conv1d(in_channels=num_inp_channels, out_channels=num_out_channels, kernel_size=kernel_size, padding_mode='circular', padding=1, bias=False)

    # with torch.no_grad():
    #     conv_model.weight = torch.nn.Parameter(new_filter)
    # conv_model = conv_model.to(device)

    # (V_T, qr) = power_qr(lambda x: conv_model(x) - conv_model(torch.zeros_like(x)), x0.clone().detach(), n_iters=500, x0_out=True, quiet=True)
    # print('manual conv layer: ') 
    # svals_orig = qr[-1]
    # print(svals_orig)

    # print('V_T: ', V_T.shape)
    # V = torch.permute(V_T, (2, 1, 0))
    # print('V: ', V.shape)
    # v1 = V_T[0,...]
    # print(v1)
    # print(v1.shape)

    # v1_cnn = v1.view(1, num_inp_channels, n)
    # print(torch.sqrt((conv_model(v1_cnn)**2).sum()))
    # print(torch.sqrt((v1**2).sum()))

    # v1_cnn = torch.zeros_like(v1_cnn)
    # v1_cnn[0,0,0] = 1.
    # print(torch.sqrt((conv_model(v1_cnn)**2).sum()))
    # print(torch.sqrt((v1**2).sum()))

    # exit(0)
    # ################################################


    v1 = V_T[0,...]
    print(v1.shape)
    v1_cnn = v1.view(1, num_inp_channels, n)


    # data_size = 10*n*min(num_inp_channels, num_out_channels)
    data_size = 1000
    data = torch.randn(data_size, num_inp_channels, n).to(device) 
    # data = torch.rand(data_size, num_inp_channels, n).to(device)
    # data = torch.cat((data, v1_cnn.repeat(100,1,1)))
    # data = torch.cat((data, V_T.view(-1, num_inp_channels, n)))
    # data = V_T.view(-1, num_inp_channels, n)
    # data = torch.nn.functional.normalize(data, p=2.0, dim=(1,2)).view(data_size, num_inp_channels, n)
    # data = v1_cnn.repeat(100,1,1)
    print('data shape: ', data.shape)
    data.requires_grad_(False)




    loss_list = build_conv_model(trained_conv_model, clipped_conv_model, data, rand_data=False, epochs=400, early_stop=False, step_size=0.01, quiet=False)
    # loss_list = build_conv_model(trained_conv_model, clipped_conv_model, data, epochs=1500, early_stop=False, step_size=0.01)

    # new_conv_model = linear_optimizer(conv_model, k=k, n=n, n_iters=50, epochs=300, same=True, clip=clip, rand_init=False, device=device, ret_sval_list=True)

    (V_T, qr) = power_qr(trained_conv_model, x0.clone().detach(), n_iters=50, x0_out=True, quiet=True)
    print('trained conv layer after: ')
    svals = qr[-1]
    print(svals)


    plt.plot(loss_list)
    print(loss_list[-10:])
    plt.savefig('plots/loss_list_opt.png', dpi=300)





    exit(0)




    #### deriving a new conv model with SVs of interest:

    # with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True) as prof:
    #     new_conv_model = linear_optimizer_adam(conv_model, k=k, n=n, n_iters=50, epochs=50, same=True, clip=clip, rand_init=False, device=device)

    # print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))

    new_conv_model, sval_list, diff_svals, loss_list, off_sval_list, target_svals, off_sval_diff_list, sval_whole_list = linear_optimizer_adam(conv_model, k=k, n=n, n_iters=50, epochs=3000, same=True, clip=clip, rand_init=False, device=device, ret_sval_list=True)
    plt.plot(sval_list, label='New model SV', alpha=0.5)
    plt.plot(np.abs(np.array(sval_list) - target_svals[0].item()), label='Diff of SV with target', alpha=0.5)
    plt.plot(diff_svals, label='SV of difference', alpha=0.5)
    plt.plot(off_sval_list, label="New model's off SV", alpha=0.5)
    plt.plot(off_sval_diff_list, label="Off SV diff", alpha=0.5)
    # plt.plot(loss_list)
    plt.legend()
    # plt.savefig('sval_1_list_plot_new_off_s2.png', dpi=300)
    plt.savefig('sval_list_plot_new_off_s1_complete_mirsky950_c24.png', dpi=300)
    plt.clf()

    # profile_json = prof.export_chrome_trace("trace.json")
    # import json
    # with open('profiler.json', 'w') as outfile:
    #     json.dump(profile_json, outfile)

    # new_conv_model = linear_optimizer_adam(conv_model, k=k, n=n, n_iters=100, epochs=50, same=True, clip=clip, rand_init=False, device='cuda')
    # new_conv_model = linear_optimizer(conv_model, k=k, n=n, n_iters=100, epochs=50, same=True, clip=clip, rand_init=True, device='cuda')

    (V, qr) = power_qr(lambda x: new_conv_model(x) - new_conv_model(torch.zeros_like(x)), x0.clone().detach(), n_iters=500, x0_out=True, quiet=True)
    print('new conv layer: ') 
    svals = qr[-1]
    print(svals)

    # print('diff new svals and target svals: \n', torch.abs(svals-target_svals))
    # print('diff new svals and target svals: \n', torch.abs(svals-clip*torch.ones_like(svals)))
    # print('diff orig svals and new svals: \n', torch.abs(svals-svals_orig))

    exit(0)


    ################################################# Conv 2d layer:
    print('\n\n\t\tProcessing 2d Convolution...')
    k = 10
    img = np.random.rand(k,n,n)
    x0 = torch.reshape(torch.tensor(img), (k, 1, n, n)).float().to(device)

    print(x0.shape)
    filter = np.array([[2, 7, -1], [1, 3, -8], [5, -3, 1]])
    new_filter = torch.reshape(torch.tensor(filter), (1, 1, 3, 3)).float()
    conv_model = torch.nn.Conv2d(1, 1, (3, 3), stride=1, padding_mode='circular', padding=(1, 1), bias=False)

    with torch.no_grad():
        conv_model.weight = torch.nn.Parameter(new_filter)

    conv_model = conv_model.to(device)

    new_conv_model = clip_conv2d_model(conv_model,k,n, radius=2.5)  # linear optimizer in a ball of radius 2.5


    (V_T, qr) = power_qr(new_conv_model, x0.clone().detach(), n_iters=500, x0_out=True, quiet=True)
    print('clipped conv layer: ') 
    svals = qr[-1]
    print(svals)


    print('clipping 2d convolution... ')
    num_inp_channels = 3
    num_out_channels = 5
    filter_x = 3
    filter_y = 5
    # filter_shape = (filter_x, filter_y, num_inp_channels, num_out_channels)
    filter_shape = (num_out_channels, num_inp_channels, filter_x, filter_y)
    filter = np.random.randint(low=-8, high=8,size=filter_shape)
    # new_filter = torch.permute(torch.tensor(filter), (3, 2, 0, 1)).float()
    new_filter = torch.tensor(filter).float()
    n = 32
    k = 20
    
    conv_model = torch.nn.Conv2d(num_inp_channels, num_out_channels, (filter_x, filter_y), stride=1, padding=(1, 2), padding_mode='circular', bias=False)

    with torch.no_grad():
        conv_model.weight = torch.nn.Parameter(new_filter)

    conv_model = conv_model.to(device)

    x_batch = torch.randn(k, num_inp_channels, n, n).float().to(device)
    VT, out = power_qr(conv_model, torch.randn_like(x_batch), n_iters=500, record = True, quiet=True, x0_out=True)
    D_powerqr = out[-1]
    print('the orig SVs: ', D_powerqr[:20])


    clipped_model = clip_conv2d_model(conv_model, k, n, radius=1.0) ### linear optimizer to ball of radius 1

    _, out_new = power_qr(clipped_model, x_batch.clone(), n_iters=500, record = True, quiet=True, x0_out=True)
    D_powerqr_tmp = out_new[-1]
    print('the clipped SVs: ', D_powerqr_tmp[:20])


    # new_conv_model = linear_optimizer_adam(conv_model, k=k, n=n, n_iters=100, epochs=200, step_size=0.2, same=False, clip=None, radius=1., device='cuda')
    new_conv_model = linear_optimizer(conv_model, k=k, n=n, n_iters=100, epochs=20, step_size=0.1, same=False, clip=None, radius=1., device='cuda')
    # new_conv_model = linear_optimizer(conv_model, k=k, n=n, n_iters=100, epochs=200, step_size=0.1, same=True, clip=20., radius=1., device='cuda')

    x_batch = torch.randn(2*k, num_inp_channels, n, n).float().to(device)
    VT, out = power_qr(new_conv_model, torch.randn_like(x_batch), n_iters=500, record = True, quiet=True, x0_out=True)
    D_powerqr_new = out[-1]
    print('the new conv: ', D_powerqr_new[:40])

    print('new weight shape: ', new_conv_model.weight.shape)

    # print('diff: \n', torch.abs(D_powerqr_new[:10]-D_powerqr[:10]))
    print('diff: \n', torch.abs(D_powerqr_new[:10]-1))


    new_conv_model_scaled = copy.deepcopy(new_conv_model)

    # with torch.no_grad:
    with torch.no_grad():
        new_conv_model_scaled.weight *= 1./D_powerqr_new[0]

    VT, out = power_qr(new_conv_model_scaled, torch.randn_like(x_batch), n_iters=500, record = True, quiet=True, x0_out=True)
    D_powerqr_new_scaled = out[-1]
    print('the new conv: ', D_powerqr_new_scaled[:40])


    loss, fw_model = sv_clip_fw(conv_model, k=k, n=n, n_iters=5, clip=1.)

    x_batch = torch.randn(2*k, num_inp_channels, n, n).float().to(device)
    VT, out = power_qr(fw_model, torch.randn_like(x_batch), n_iters=500, record = True, quiet=True, x0_out=True)
    D_powerqr_new = out[-1]
    print('the fw conv: ', D_powerqr_new[:40])

    exit(0)


    # D_sedghi = np.sort(np.absolute(np.fft.fft2(filter, [n, n]).flatten()))[::-1]
    # print('sedghi: ', D_sedghi.shape)
    # print('sedghi: ', D_sedghi[:10])
    # print('sedghi: ', D_sedghi[45:55])
    
    # U_T = conv_model(V_T)
    # U_T = torch.reshape(U_T, (k, n*n))
    # U = U_T.T
    # V_T = torch.reshape(V_T, (k, n*n))
    # print('V_T: ', V_T.shape)
    # print('U: ', U.shape)
    # eq_mat = U @ torch.diag(svals) @ V_T 
    # eq_mat = torch.reshape(eq_mat, (k, 1, n, n))
    # out_1 =  eq_mat @ x0 
    # out_2 = conv_model(x0)
    # print(out_1[0,...])
    # print(out_2[1,...])
    # print(out_1.shape)
    # print(out_2.shape)
    # exit(0)

    # exit(0)
    
    new_conv_model, factor, V_T = linear_optimizer_adam(conv_model, k=50, n_iters=1, device=device, return_factor=True)
    # new_conv_model = linear_optimizer(conv_model, x0=x0.clone().detach(), device=device)
    (V, qr) = power_qr(new_conv_model, x0.clone().detach(), n_iters=500, x0_out=True, quiet=True)
    print('new conv layer: ') 
    svals = qr[-1]
    print(svals[:10])
    print(svals[45:55])
    


    # x0 = torch.reshape(torch.tensor(img), (400, 1, 20, 20)).float().to(device)
    x0 = torch.randn(n*n, 1, n, n, device=device)
    (V, qr) = power_qr(lambda x: conv_model(torch.matmul(factor, x)), x0.clone().detach(), n_iters=500, x0_out=True, quiet=True)
    print('new conv layer: ') 
    svals = qr[-1]
    print(svals[:10])
    print(svals[45:55])

    
    exit(0)


    loss, m_clipped = extract_dimension(l_layer, x0, k=1, reverse=False)
    
    # loss, m_clipped = sv_clip_fw_correct(l_layer, x0.clone().detach(), clip=1.0, n_iters=20, quiet=True)

    print("qr torch layer again")
    (new_x0, qr) = power_qr(l_layer, x0.clone().detach(), n_iters, x0_out=True, quiet=True)
    print(qr[-1])


    # print("qr clipped")
    # (new_x0, qr_clipped) = power_qr(m_clipped, x0.clone().detach(), n_iters, x0_out=True, quiet=True)
    # print(qr_clipped[-1])
    

    

