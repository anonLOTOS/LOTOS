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

from models.spectral_normalization_deflate_complex_both_bn import SpectralNorm
# from models.spectral_normalization_gouk_farnia import SpectralNorm_gouk_farnia as SpectralNorm



class CNNBN(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=0, dilation=1, padding_mode='zeros', bias=False, device='cpu', clip_flag=True, clip=1., clip_steps=50, clip_opt_iter=1, init_delay=0, summary=False, identifier=0, writer=None, bn=True, save_info=False):
        super(CNNBN, self).__init__()
        self.input_size = [28]

        self.sub_conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, padding_mode=padding_mode, bias=bias)

        self.kernels = [(self.sub_conv1.weight, self.input_size)]

        self.bn1 = nn.BatchNorm2d(out_planes, momentum=0.1, track_running_stats=True)
        # self.bn1 = nn.BatchNorm2d(out_planes, momentum=0.1, track_running_stats=True)
        self.bn_flag = bn

    def get_all_kernels(self):
        return self.kernels

    def forward(self, x):
        self.input_size[0] = x.shape[-1]
        x = self.sub_conv1(x)
        if self.bn_flag:
            x = self.bn1(x)
        return x



class SimpleConv_orig(nn.Module):
    def __init__(self, in_chan=3, k=1, num_classes=10, leaky_relu=False, device='cpu', clip_flag=True, clip=1., clip_steps=50, clip_opt_iter=5, init_delay=0, summary=False, identifier=0, writer=None, bias=True, bn=True, kernel_size=3, stride=1, padding=1, dilation=1, padding_mode='zeros', lin_layer=None):
        super(SimpleConv_orig, self).__init__()
        self.kernels = []
        # self.conv1_ = nn.Conv2d(1, 32, 3, stride=2, bias=bias, padding=(1,1), padding_mode='circular')#, padding_mode='circular') ############## mnist
        # self.conv1_ = nn.Conv2d(3, 32, 3, stride=2, bias=bias, padding=(1,1))#, padding_mode='circular')#, padding_mode='circular') ########### cifar
        # self.conv1_ = nn.Conv2d(1, 32, 3, stride=1, bias=bias, padding='same')#, padding_mode='circular')#, padding_mode='circular')
        # self.conv1 = SpectralNorm(self.conv1_, device=device, clip_flag=clip_flag, clip=clip, clip_steps=clip_steps, clip_opt_iter=clip_opt_iter, summary=summary, writer=writer, identifier=identifier+1, init_delay=init_delay)
        self.conv1 = CNNBN(in_chan, 64, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False, padding_mode=padding_mode, device=device, clip_flag=clip_flag, clip=clip, clip_steps=clip_steps, clip_opt_iter=clip_opt_iter, init_delay=init_delay, summary=summary, writer=writer, identifier=identifier+1, bn=bn)
        # self.conv1 = self.conv1_
        # self.bn1 = nn.BatchNorm2d(32)
        self.kernels.extend(self.conv1.get_all_kernels())


        # self.conv2 = CNNBN(64, 128, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, padding_mode='zeros', device=device, clip_flag=clip_flag, clip=clip, clip_steps=clip_steps, clip_opt_iter=clip_opt_iter, init_delay=init_delay, summary=summary, writer=writer, identifier=identifier+1, bn=bn)
        # self.kernels.extend(self.conv2.get_all_kernels())

        # self.fc1_ = nn.Linear(13072, 10)  ### cifar, 128 chan, stride 1
        # self.fc1_ = nn.Linear(65536, 10)  ### cifar, 64 chan, stride 1
        W = 32
        if in_chan == 1:
            W = 28

        outdim = (W - kernel_size+2*padding)//stride + 1
        size = outdim*outdim*64

        if lin_layer is not None:
            size = lin_layer

        self.fc1_ = nn.Linear(size, 10)  

        # self.fc1_ = nn.Linear(9216, 10)  ### mnist, 64 chan, stride 2 pad 0 k5
        # self.fc1_ = nn.Linear(43264, 10)  ### mnist, 64 chan, stride 1 pad 0
        # self.fc1_ = nn.Linear(50176, 10)  ### mnist, 64 chan, stride 1
        # self.fc1_ = nn.Linear(12544, 10)  ### mnist, 64 chan, stride 2
        # self.fc1_ = nn.Linear(43264, 10)  ### mnist, 64 chan, dilation 2

        # self.fc1_ = nn.Linear(25088, 10)  ### mnist, 2 conv layers stride 2

        # self.fc1_ = nn.Linear(25088, 10)  ### mnist stride 1
        # self.fc1_ = nn.Linear(5408, 10)
        # self.fc1_ = nn.Linear(6272, 10) ### mnist stride 2
        # self.fc1_ = nn.Linear(8192, 10) ### cifar stride 2


        # self.fc1 = SpectralNorm(self.fc1_, writer=writer)
        self.fc1 = self.fc1_


    def get_all_kernels(self):
        return self.kernels


    def forward(self, x):
        # print(x.shape)
        x = self.conv1(x)
        # print(x.shape)
        # print(x.shape)
        # x = self.bn1(x)
        x = F.relu(x)
        # print(x.shape)

        # x = self.conv2(x)
        # x = F.relu(x)
        # print(x.shape)

        # x = torch.flatten(x, 1)
        x = x.view(x.size(0), -1)
        # print(x.shape)

        x = self.fc1(x)
        return x
    


def simpleConv_orig(**kwargs):
    return SimpleConv_orig(**kwargs)



