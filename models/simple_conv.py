'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
# from models.spectral_normalization_nondiff_final_deflate import SpectralNorm
# from models.spectral_normalization import SpectralNorm #as SpectralNorm_miyato
# from models.spectral_normalization_deflate_complex_both import SpectralNorm

from models.spectral_normalization_deflate_complex_both_condnum import SpectralNorm
# from models.spectral_normalization_new_bn import SpectralNorm


class CNNBN(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, padding_mode='zeros', bias=False, device='cpu', clip_flag=True, clip=1., bottom_clip=0.5, clip_steps=50, clip_opt_iter=1, init_delay=0, summary=False, identifier=0, writer=None, bn=True, save_info=False):
        super(CNNBN, self).__init__()
        if in_planes == 1:
            rank = 28*28
        else:
            rank = 32*32*3
        # self.sub_conv1_ = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False, padding_mode='circular')#, padding_mode='circular') ############## mnist
        self.sub_conv1 = SpectralNorm(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, padding_mode= padding_mode, bias=bias), k=rank, bottom_clip=bottom_clip, device=device, clip_flag=clip_flag, clip=clip, clip_steps=clip_steps, clip_opt_iter=clip_opt_iter, init_delay=init_delay, summary=summary, writer=writer, identifier=str(identifier) + '_1', save_info=save_info)
        # self.sub_conv1 = self.sub_conv1_
        self.bn1 = SpectralNorm(nn.BatchNorm2d(out_planes, momentum=0.1, track_running_stats=True), device=device, clip_flag=False, clip=1., clip_steps=clip_steps, clip_opt_iter=clip_opt_iter, init_delay=init_delay, summary=summary, writer=writer, identifier=str(identifier) + '_bn', save_info=save_info)
        # self.bn1 = nn.BatchNorm2d(out_planes, momentum=0.1, track_running_stats=True)
        self.bn_flag = bn

    def forward(self, x):
        x = self.sub_conv1(x)
        if self.bn_flag:
            x = self.bn1(x)
        return x


class SimpleConv(nn.Module):
    def __init__(self, concat_sv=False, in_chan=3, k=1, num_classes=10, leaky_relu=False, device='cpu', clip_flag=True, clip_outer=False, clip=1., clip_bottom=0.5, clip_concat=1., clip_steps=50, clip_opt_iter=5, init_delay=0, summary=False, identifier=0, writer=None, bias=True, bn=True, kernel_size=3, stride=1, padding=1, dilation=1, padding_mode='zeros', lin_layer=None, save_info=False):
        super(SimpleConv, self).__init__()
        # self.conv1_ = nn.Conv2d(1, 32, 3, stride=2, bias=bias, padding=(1,1), padding_mode='circular')#, padding_mode='circular') ############## mnist
        # self.conv1_ = nn.Conv2d(3, 32, 3, stride=2, bias=bias, padding=(1,1))#, padding_mode='circular')#, padding_mode='circular') ########### cifar
        # self.conv1_ = nn.Conv2d(1, 32, 3, stride=1, bias=bias, padding='same')#, padding_mode='circular')#, padding_mode='circular')
        if concat_sv:
            self.conv1_ = CNNBN(in_chan, 64, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, padding_mode=padding_mode, bias=False, device=device, clip_flag=clip_flag, clip=clip, clip_steps=clip_steps, clip_opt_iter=clip_opt_iter, init_delay=init_delay, summary=summary, writer=writer, identifier=identifier+1, bn=bn, save_info=save_info)
            self.conv1 = SpectralNorm(self.conv1_, device=device, clip_flag=clip_outer, clip=clip_concat, clip_steps=clip_steps, clip_opt_iter=clip_opt_iter, summary=summary, writer=writer, identifier=identifier+1, init_delay=init_delay, save_info=save_info)
        else:
            self.conv1_ = CNNBN(in_chan, 64, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, padding_mode=padding_mode, bias=False, device=device, clip_flag=clip_flag, clip=clip, bottom_clip=clip_bottom, clip_steps=clip_steps, clip_opt_iter=clip_opt_iter, init_delay=init_delay, summary=summary, writer=writer, identifier=identifier+1, bn=bn, save_info=save_info)
            self.conv1 = self.conv1_
        # self.conv1 = SpectralNorm(self.conv1_, device=device, clip_flag=False, clip=clip, clip_steps=clip_steps, clip_opt_iter=clip_opt_iter, summary=summary, writer=writer, identifier=identifier+1, init_delay=init_delay)

        W = 32
        if in_chan == 1:
            W = 28

        outdim = (W - kernel_size+2*padding)//stride + 1
        size = outdim*outdim*64

        if lin_layer is not None:
            size = lin_layer

        self.fc1_ = nn.Linear(size, 10)

        # self.fc1_ = nn.Linear(13072, 10)  ### cifar, 128 chan, stride 1
        # self.fc1_ = nn.Linear(65536, 10)  ### cifar, 64 chan, stride 1
        # self.fc1_ = nn.Linear(50176, 10)  ### mnist, 64 chan, stride 1
        # self.fc1_ = nn.Linear(12544, 10)  ### mnist, 64 chan, stride 2

        # self.fc1_ = nn.Linear(43264, 10)  ### mnist, 64 chan, stride 1 pad 0
        # self.fc1_ = nn.Linear(50176, 10)  ### mnist, 64 chan, stride 1
        # self.fc1_ = nn.Linear(12544, 10)  ### mnist, 64 chan, stride 2
        # self.fc1_ = nn.Linear(43264, 10)  ### mnist, 64 chan, dilation 2

        # self.fc1_ = nn.Linear(25088, 10)  ### mnist stride 1
        # self.fc1_ = nn.Linear(5408, 10)
        # self.fc1_ = nn.Linear(6272, 10) ### mnist stride 2
        # self.fc1_ = nn.Linear(8192, 10) ### cifar stride 2

        self.fc1 = SpectralNorm(self.fc1_, device=device, clip_flag=False, clip=clip, clip_steps=clip_steps, clip_opt_iter=1, summary=summary, writer=writer, identifier=identifier+3, clip_opt_stepsize=0.5, init_delay=init_delay, save_info=save_info)

        # self.fc1 = SpectralNorm(self.fc1_, writer=writer)
        # self.fc1 = self.fc1_


    def forward(self, x):
        x = self.conv1(x)
        # x = self.bn1(x)
        x = F.relu(x)
        # print(x.shape)

        # x = self.conv2(x)
        # x = F.relu(x)

        # x = torch.flatten(x, 1)
        x = x.view(x.size(0), -1)
        # print(x.shape)

        x = self.fc1(x)
        return x
    


def simpleConv(**kwargs):
    return SimpleConv(**kwargs)



