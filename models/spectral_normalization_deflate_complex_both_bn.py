import torch
from torch.optim.optimizer import Optimizer, required

from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from torch.nn import Parameter
from models.clip_toolbox import *
# from clip_toolbox_new import *
import copy
import numpy as np
import pandas as pd


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,2,3"


class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', k=1, clip_flag=True, deflate_iter=1, init_delay=500, clip=1., init_pqr_iter=100, pqr_iter=1, clip_steps=50, clip_opt_iter=1, clip_opt_stepsize=0.35, clip_rsv_size=1, summary=False, device='cuda', bn_hard=False, writer=None, identifier=None, save_info=False, scale_only=False, exact_flag=True, warn_flag=False):
        super(SpectralNorm, self).__init__()
        self.name = name
        self.pqr_iter = pqr_iter
        self.init_pqr_iter = init_pqr_iter
        self.D_powerqr = None
        self.clip_steps = clip_steps
        self.clip = clip
        self.clip_opt_iter = clip_opt_iter
        self.clip_opt_stepsize = clip_opt_stepsize 
        self.device=device
        self.module = module.to(self.device)
        # print('-----------', self.device)
        self.counter = 0
        self.k = k
        self.clip_data_size = clip_rsv_size
        self.summary=summary
        self.clip_flag = clip_flag
        self.init_delay = init_delay
        self.deflate_iter = deflate_iter
        self.save_info = save_info
        self.scale_only = scale_only
        self.bn_hard = bn_hard
        self.warn_flag = warn_flag
        self.exact_flag = exact_flag

        if self.save_info:
            self.lsv_list = []
        if self.summary:
            self.lsv = None
            self.identifier = identifier
        self.writer = writer

        if self.clip_flag:
            print('!!!!!!! Clipping is active !!!!!!!! clip val: ', self.clip)
            print('def iter: ', self.deflate_iter)
            print('opt step size: ', self.clip_opt_stepsize)


    def _update_VT(self, *args, n_iters=None, rand_init=False):
        if self.save_info and self.counter > 0 and self.counter % 200 == 0:
            df = pd.DataFrame(self.lsv_list)
            # df.to_csv('lsv_dir/lsv_list_' + str(self.identifier) + '.csv')
            

        if isinstance(self.module, torch.nn.BatchNorm2d):
            if self.summary:
                running_var = torch.ones_like(self.module.weight)
                if vars(self.module)['_buffers']['running_var'] is not None:
                    input_ = args[0]
                    cur_var = torch.var(input_, unbiased=False, dim=[0,2,3])
                    running_var_prev = vars(self.module)['_buffers']['running_var'] 
                    # momentum = vars(self.module)['momentum'] 
                    # running_var = momentum*cur_var + (1.-momentum)*running_var_prev 
                    running_var = running_var_prev 
                    running_var = running_var + vars(self.module)['eps']  # ----------------> the one we use
                    # running_var = cur_var + vars(self.module)['eps']
                else:
                    input_ = args[0]
                    # print('train: ', input_.shape)
                    cur_var = torch.var(input_, unbiased=False, dim=[0,2,3])
                    running_var = cur_var + vars(self.module)['eps']

                self.lsv = torch.max(torch.abs(self.module.weight/torch.sqrt(running_var)))
                if self.counter % 50 == 0 and self.writer is not None:
                    self.writer.add_scalar('train/lsv_bn_' + str(self.identifier), self.lsv, self.counter)

            if self.save_info and self.counter % 100 == 0:
                self.lsv_list.append(self.lsv.item()) 
                    # self.writer.add_scalar('train/maxW_' + str(self.identifier), torch.max(self.module.weight), self.counter)
            return 


        if n_iters is None:
            n_iters = self.pqr_iter
        if not self._made_params():
            self._make_params(*args)
        VT = getattr(self.module, self.name + "_VT")
        if False:
            self.bad_vector = getattr(self.module, self.name + "_BVT1") 
            self.bad_vector_2 = getattr(self.module, self.name + "_BVT2") 
        if rand_init:
            VT = torch.rand_like(VT)


        # copy_module = copy.deepcopy(self.module)
        # VT, out = power_qr(lambda x: copy_module(x) - copy_module(torch.zeros_like(x)), VT, n_iters=n_iters, record=False, quiet=True, x0_out=True, device=self.device)

        self.module.eval()
        VT, out = power_qr(lambda x: self.module(x) - self.module(torch.zeros_like(x)), VT, n_iters=n_iters, record=False, quiet=True, x0_out=True, device=self.device)
        self.module.zero_grad()
        self.D_powerqr = out[-1].detach()
        self.module.train()

        if self.summary:
            self.lsv = self.D_powerqr[0]
            if self.counter % 50 == 0 and self.writer is not None:
                self.writer.add_scalar('train/lsv_' + str(self.identifier), self.lsv, self.counter)

                if False:
                #if self.bad_vector is not None:
                    self.bad_vec_length = torch.sqrt(torch.sum( (self.module(self.bad_vector[0:1])- self.module(torch.zeros_like(self.bad_vector[0:1])) )**2)).item()/torch.sqrt(torch.sum(self.bad_vector[0:1] **2)).item()
                    self.bad_vec_length_2 = torch.sqrt(torch.sum( (self.module(self.bad_vector_2[0:1])- self.module(torch.zeros_like(self.bad_vector_2[0:1])) )**2)).item()/torch.sqrt(torch.sum(self.bad_vector_2[0:1] **2)).item()
                    # self.bad_vec_length_2 = torch.sqrt(torch.sum((self.module(self.bad_vector[0:1]) - self.module(torch.zeros_like(self.bad_vector[0:1]))) **2)).item()
                    self.writer.add_scalar('train/badVecLen_' + str(self.identifier), self.bad_vec_length, self.counter)
                    self.writer.add_scalar('train/badVecLen2_' + str(self.identifier), self.bad_vec_length_2, self.counter)

        if self.save_info and self.counter % 100 == 0 and self.counter > 0:
            self.lsv_list.append(self.lsv.item()) 

        VT.requires_grad_(False)
        self.D_powerqr.requires_grad_(False)
        
        with torch.no_grad():
            setattr(self.module, self.name + "_VT", VT.detach())
        del VT


    def _clip_module(self, *args):

        # for (m_name, m) in self.module.named_modules():
        #     print('module: ', m_name)
        #     if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)):
        #         print('found: ', m_name)

                # for (m_name_, m_) in m.named_modules():
                #     print('module sub: ', m_name_)
                #     if isinstance(m_, (torch.nn.Linear, torch.nn.Conv2d)):
                #         print('found sub: ', m_name_)


        # VT = getattr(self.module, self.name + "_VT")

        # self.module.eval()
        if isinstance(self.module, torch.nn.Linear):
            VT = getattr(self.module, self.name + "_VT")
            # print('linear')
            w = getattr(self.module, self.name)
            data_size = self.clip_data_size
            trained_module = copy.deepcopy(self.module)

            if self.D_powerqr[0] > self.clip:
                deflate_model_multi(trained_module, VT[0:data_size].detach(), clip=self.clip, step_size=self.clip_opt_stepsize, epochs=self.clip_opt_iter, device=self.device, deflate_iter=self.deflate_iter)

            w.data = copy.deepcopy(trained_module.weight.detach())

        elif isinstance(self.module, (torch.nn.Conv1d, torch.nn.Conv2d)):
            if self.scale_only:
                w = getattr(self.module, self.name)
                if self.D_powerqr[0] > self.clip:
                    w.data = copy.deepcopy(self.module.weight.detach()/self.D_powerqr[0])

            else:
                VT = getattr(self.module, self.name + "_VT")
                # print('conv')
                w = getattr(self.module, self.name)
                with torch.no_grad():
                    data_size = self.clip_data_size
                    trained_module = copy.deepcopy(self.module)

                
                # if self.D_powerqr[0] > self.clip:
                # if torch.abs(self.D_powerqr[0] - self.clip) > 0.05:
                if self.exact_flag:
                    clip_condition = self.D_powerqr[0] > self.clip or self.clip - self.D_powerqr[0] >= 0.05
                else:
                    clip_condition = self.D_powerqr[0] > self.clip
                # if self.D_powerqr[0] > self.clip or self.clip - self.D_powerqr[0] >= 0.05: #### tlower
                if clip_condition:
                    deflate_model_multi(trained_module, VT[0:data_size].detach(), clip=self.clip, step_size=self.clip_opt_stepsize, epochs=self.clip_opt_iter, device=self.device, deflate_iter=self.deflate_iter)


                w.data = copy.deepcopy(trained_module.weight.detach())
                del trained_module

        elif isinstance(self.module, torch.nn.BatchNorm2d):

            # print('found BNnnnnn.............')
            w = getattr(self.module, self.name)
            with torch.no_grad():
                # self.module.weight = torch.nn.Parameter(self.clip*w/torch.max(w/vars(self.module)['_buffers']['running_var']))

                # print('var: ', vars(self.module)['_buffers']['running_var'])
                # print('w/var: ', w/torch.sqrt(vars(self.module)['_buffers']['running_var']))
                # print('max: ', torch.max(w/vars(self.module)['_buffers']['running_var']))
                # print('var: ', vars(self.module)['_buffers']['running_var'].shape)
                # print('w/var: ', (w/vars(self.module)['_buffers']['running_var']).shape) 
                # print('max: ', torch.max(w/vars(self.module)['_buffers']['running_var']).shape)
                # print(vars(self.module))

                running_var = torch.ones_like(w)
                if vars(self.module)['_buffers']['running_var'] is not None:
                    # print('yes!')
                    input_ = args[0]
                    # print('train: ', input_.shape)
                    cur_var = torch.var(input_, unbiased=False, dim=[0,2,3])
                    running_var_prev = vars(self.module)['_buffers']['running_var'] 
                    momentum = vars(self.module)['momentum'] 
                    new_running_var = momentum*cur_var + (1.-momentum)*running_var_prev 
                    # running_var = running_var + vars(self.module)['eps']

                    running_var = running_var_prev + vars(self.module)['eps'] # ----------> the one we use
                    # running_var = cur_var + vars(self.module)['eps']

                    if self.counter % 50 == 0 and self.writer is not None:
                        self.writer.add_scalar('train/prev_run_var_' + str(self.identifier), running_var_prev.sum().item()/running_var_prev.shape[0], self.counter)
                        self.writer.add_scalar('train/cur_run_var_' + str(self.identifier), cur_var.sum().item()/cur_var.shape[0], self.counter)
                        self.writer.add_scalar('train/new_run_var_' + str(self.identifier), new_running_var.sum().item()/new_running_var.shape[0], self.counter)

                else:
                    input_ = args[0]
                    # print('train: ', input_.shape)
                    cur_var = torch.var(input_, unbiased=False, dim=[0,2,3])
                    # running_var = cur_var + vars(self.module)['eps']

                    if self.counter % 50 == 0 and self.writer is not None:
                        self.writer.add_scalar('train/cur_run_var_' + str(self.identifier), cur_var.sum().item()/cur_var.shape[0], self.counter)

                    
                # w.data = copy.deepcopy(torch.nn.Parameter(torch.sqrt(running_var)*self.clip*w/(torch.max(torch.abs(w)) + vars(self.module)['eps'])))
                # self.module.reset_running_stats()

                # print('var: ', vars(self.module)['_buffers']['running_var'])
                # print('w/var: ', w/torch.sqrt(vars(self.module)['_buffers']['running_var']))
                # print('max: ', torch.max(w/torch.sqrt(vars(self.module)['_buffers']['running_var'] + vars(self.module)['eps'])))
                # exit(0)


            ######## the others:
            # norm = torch.max(torch.abs(self.module.weight / torch.sqrt(running_var)))
            # self.module.weight.copy_(self.module.weight / torch.clamp(norm / self.clip, min=1.0))
            #######################################
            
            
            w_clamped = torch.clamp(copy.deepcopy(w), min=-torch.sqrt(running_var)*self.clip, max=torch.sqrt(running_var)*self.clip)

            if self.bn_hard:
                w.data = w_clamped
            else:
                w.data = 0.9*copy.deepcopy(w) + 0.1*w_clamped

            # w.data = 0.5*copy.deepcopy(w) + 0.5*w_clamped

            # # if self.counter < 40000:
            # if self.lsv > 1:
            #     w.data = 0.95*copy.deepcopy(w) + 0.05*copy.deepcopy(torch.nn.Parameter(torch.sqrt(running_var)*self.clip*w/(torch.max(torch.abs(w)) + vars(self.module)['eps'])))
            # # else:
            # #     w.data = 0.999*copy.deepcopy(w) + 0.001*copy.deepcopy(torch.nn.Parameter(torch.sqrt(running_var)*self.clip*w/(torch.max(torch.abs(w)) + vars(self.module)['eps'])))


        else:
            ##### old solution:
            """
            self.clip_opt_stepsize = 0.01
            with torch.no_grad():
                data_size = self.clip_data_size
                trained_module = copy.deepcopy(self.module)

            if self.D_powerqr[0] > self.clip:
                deflate_model_multi(trained_module, VT[0:data_size].detach(), clip=self.clip, step_size=self.clip_opt_stepsize, epochs=self.clip_opt_iter, device=self.device, deflate_iter=self.deflate_iter)
                # deflate_model_multi(trained_module, VT[0:data_size].detach(), clip=self.clip, step_size=self.clip_opt_stepsize, epochs=1, device=self.device, deflate_iter=self.deflate_iter)


            trained_module.zero_grad()
            self.module.load_state_dict(trained_module.state_dict())
            del trained_module
            """


            ###### new code:
            # print('Another layer type!')

            VT = getattr(self.module, self.name + "_VT")

            layers = []
            for (m_name, m) in self.module.named_modules():
                # print('module: ', m_name)
                if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)):
                    # print('found: ', m_name)
                    for (m_name_, m_) in m.named_modules():
                        # print('module sub: ', m_name_)
                        if isinstance(m_, (torch.nn.Linear, torch.nn.Conv2d)):
                            # print('found sub: ', m_name_)
                            layers.append(copy.deepcopy(m))

                elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
                    # print('found: ', m_name)
                    layers.append(copy.deepcopy(m))

                # else:
                #     layers.append(copy.deepcopy(m))

            trained_module = nn.Sequential(*layers)


            self.clip_opt_stepsize = 0.01
            with torch.no_grad():
                data_size = self.clip_data_size
                # trained_module = copy.deepcopy(self.module)

            if self.D_powerqr[0] > self.clip:
                deflate_model_multi(trained_module, VT[0:data_size].detach(), clip=self.clip, step_size=self.clip_opt_stepsize, epochs=self.clip_opt_iter, device=self.device, deflate_iter=self.deflate_iter)
                # deflate_model_multi(trained_module, VT[0:data_size].detach(), clip=self.clip, step_size=self.clip_opt_stepsize, epochs=1, device=self.device, deflate_iter=self.deflate_iter)

            # w.data = copy.deepcopy(trained_module.weight.detach())
            trained_module.zero_grad()
            # self.module.load_state_dict(trained_module.state_dict())

            stateDict = {}
            for key, value in trained_module.state_dict().items():
                # print('key: ', key)
                if key.split('_')[-1] == 'VT':
                    continue
                ln = key.split('.')[0] 
                new_key = key[1:]
                if ln == '0':
                    new_key = 'sub_conv1.module' + new_key
                elif ln == '1':
                    new_key = 'bn1.module' + new_key

                # print('new key: ', new_key)
                stateDict[new_key] = value

            # print(stateDict.keys())
            # print(self.module.state_dict().keys())
            with torch.no_grad():
                self.module.load_state_dict(stateDict, strict=False)
            del trained_module



    def _made_params(self):
        try:
            VT = getattr(self.module, self.name + "_VT")
            return True
        except AttributeError:
            return False


    def _make_params(self, *args):
        # if isinstance(self.module, torch.nn.BatchNorm2d):
        #     self.module.register_buffer(self.name + "_VT", VT)
        #     return 

        input_shape = args[0].shape
        self.n = input_shape[-1]
        self.VT_shape = input_shape[1:]

        VT_shape = tuple([self.k] + list(self.VT_shape))
        print(VT_shape)
        x_batch = torch.randn(VT_shape, device=self.device)
        self.module.eval()
        VT, out = power_qr(lambda x: self.module(x) - self.module(torch.zeros_like(x)), x_batch, n_iters=self.init_pqr_iter, record=False, quiet=True, x0_out=True, device=self.device)
        self.module.zero_grad()
        self.D_powerqr = out[-1].detach()
        self.module.train()
        if self.summary:
            self.lsv = self.D_powerqr[0]
            if self.save_info:
                self.lsv_list.append(self.lsv.item()) 
            if self.counter % 50 == 0 and self.writer is not None:
                self.writer.add_scalar('train/lsv_' + str(self.identifier), self.lsv, self.counter)

        self.module.register_buffer(self.name + "_VT", VT)
        if False:
            self.module.register_buffer(self.name + "_BVT1", None)
            self.module.register_buffer(self.name + "_BVT2", None)
        del VT
        del x_batch


    def forward(self, *args):
        if self.training:
            self._update_VT(*args)
            self.counter += 1
            if self.counter % self.clip_steps == self.init_delay and self.counter > self.init_delay and self.clip_flag:
                # self._update_VT(*args, n_iters=10, rand_init=False)
                if self.warn_flag:
                    print('clipping started!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', self.identifier, self.counter)
                    self.warn_flag = False

                self._clip_module(*args)
                self._update_VT(*args, n_iters=10, rand_init=True)

        # else: 
        #     if self.counter > 100 and not self.module.training:
        #         if isinstance(self.module, torch.nn.BatchNorm2d):
        #             input_ = args[0]
        #             # print(len(args))
        #             print(input_.shape)
        #             cur_var = torch.var(input_, unbiased=False, dim=[0,2,3])
        #             self.writer.add_scalar('eval/cur_var_' + str(self.identifier), cur_var.sum().item(), self.counter)

        #             # print(vars(self.module))
        #             if vars(self.module)['_buffers']['running_var'] is not None:
        #                 running_var_prev = vars(self.module)['_buffers']['running_var']
        #                 self.writer.add_scalar('eval/run_var_' + str(self.identifier), running_var_prev.sum().item(), self.counter)


        return self.module.forward(*args)


