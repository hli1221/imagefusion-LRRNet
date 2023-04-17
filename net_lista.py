# -*- coding:utf-8 -*-
# @Author: Li Hui, Jiangnan University
# @Email: hui_li_jnu@163.com
# @Project : LRRNet
# @File : net_lista.py
# @Time : 2021/4/7 15:26

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EPSILON = 1e-4
MAX = 1e2


# Convolution operation
class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


# Dense convolution unit
class DenseConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(DenseConv2d, self).__init__()
        self.dense_conv = ConvLayer(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.dense_conv(x)
        out = torch.cat([x, out], 1)
        return out


# Dense Block unit
class DenseBlock(torch.nn.Module):
    def __init__(self, in_channels, kernel_size, stride):
        super(DenseBlock, self).__init__()
        out_channels_def = 64
        denseblock = []
        denseblock += [DenseConv2d(in_channels, out_channels_def, kernel_size, stride),
                       DenseConv2d(in_channels+out_channels_def, out_channels_def, kernel_size, stride)]
        self.denseblock = nn.Sequential(*denseblock)
        self.conv = ConvLayer(in_channels+out_channels_def * 2, in_channels, 1, stride)

    def forward(self, x, isLast=False):
        out = self.denseblock(x)
        if not isLast:
            out = self.conv(out)
        return out


def eta_l1(r_, lam_):
    # l_1 norm based
    # implement a soft threshold function y=sign(r)*max(0,abs(r)-lam)
    B, C, H, W = r_.shape
    lam_ = torch.reshape(lam_, [1, C, 1, 1])
    lam_ = lam_.repeat(B, 1, H, W)
    R = torch.sign(r_) * torch.clamp(torch.abs(r_) - lam_, 0)
    return R


class LRR_Block_lista(nn.Module):
    def __init__(self, s, n, c, stride):
        super(LRR_Block_lista, self).__init__()
        self.conv_Wdz = ConvLayer(n, c, s, stride)
        self.conv_Wdtz = ConvLayer(c, n, s, stride)

    def forward(self, x, tensor_z, lam_theta, lam_z):
        # Updating
        convZ1 = self.conv_Wdz(tensor_z)
        midZ = x - convZ1
        tensor_c = lam_z*tensor_z + self.conv_Wdtz(midZ)
        # tensor_c = tensor_b + hZ
        Z = eta_l1(tensor_c, lam_theta)
        return Z


class GetLS_Net(nn.Module):
    def __init__(self, s, n, channel, stride, num_block):
        super(GetLS_Net, self).__init__()
        # n = 128  # number of filters
        # s = 3  # filter size
        # num_block = 4  # number of layers
        # Channel = 3
        self.n = n
        self.num_block = num_block
        self.conv_W00 = ConvLayer(channel, 2*n, s, stride)
        self.lamj = nn.Parameter(torch.rand(1, self.n*2))  # l1-norm
        self.lamz = nn.Parameter(torch.rand(1, 1))
        self.up = nn.Upsample(scale_factor=2)
        for i in range(num_block):
            self.add_module('lrrblock' + str(i), LRR_Block_lista(s, 2 * n, channel, stride))

    def forward(self, x):
        b, c, h, w = x.shape
        tensor_l = self.conv_W00(x)  # Z_0
        tensor_z = eta_l1(tensor_l, self.lamj)

        for i in range(self.num_block):
            # print('num_block - ' + str(i))
            lrrblock = getattr(self, 'lrrblock' + str(i))
            tensor_z = lrrblock(x, tensor_z, self.lamj, self.lamz)
        L = tensor_z[:, :self.n, :, :]
        S = tensor_z[:, self.n: 2 * self.n, :, :]
        return L, S


class GetLS_dense(nn.Module):
    def __init__(self, s, n, channel, stride, num_block):
        super(GetLS_dense, self).__init__()
        # n = 128  # number of filters
        # s = 3  # filter size
        # num_block = 4  # number of layers
        # Channel = 3
        self.num_block = num_block
        self.n = n
        self.conv_0 = ConvLayer(channel, n, s, stride)
        for i in range(num_block):
            self.add_module('denseblock' + str(i), DenseBlock(n, s, stride))

    def forward(self, x):
        tensor_z = self.conv_0(x)
        last_block = False
        for i in range(self.num_block):
            denseblock = getattr(self, 'denseblock' + str(i))
            if i == (self.num_block-1):
                last_block = True
            tensor_z = denseblock(tensor_z, last_block)
        L = tensor_z[:, :self.n, :, :]
        S = tensor_z[:, self.n: 2 * self.n, :, :]
        return L, S


class GetLS_Conv8(nn.Module):
    def __init__(self, s, n, channel, stride, num_block):
        super(GetLS_Conv8, self).__init__()
        # n = 128  # number of filters
        # s = 3  # filter size
        # num_block = 4  # number of layers
        # Channel = 3
        self.n = n
        self.conv_8 = nn.Sequential(
            ConvLayer(channel, 2 * n, s, stride),
            ConvLayer(2 * n, 2 * n, s, stride),
            ConvLayer(2 * n, 2 * n, s, stride),
            ConvLayer(2 * n, 2 * n, s, stride),
            ConvLayer(2 * n, 2 * n, s, stride),
            ConvLayer(2 * n, 2 * n, s, stride),
            ConvLayer(2 * n, 2 * n, s, stride),
            ConvLayer(2 * n, 2 * n, s, stride)
        )

    def forward(self, x):
        tensor_z = self.conv_8(x)
        L = tensor_z[:, :self.n, :, :]
        S = tensor_z[:, self.n: 2 * self.n, :, :]
        return L, S


class Decoder_simple(nn.Module):
    def __init__(self, s, n, channel, stride, fusion_type):
        super(Decoder_simple, self).__init__()
        self.type = fusion_type
        self.conv_ReZx = ConvLayer(n, channel, s, stride)
        self.conv_ReUx = ConvLayer(n, channel, s, stride)
        self.conv_ReZy = ConvLayer(n, channel, s, stride)
        self.conv_ReUy = ConvLayer(n, channel, s, stride)

        if self.type.__contains__('cat'):
            # cat
            self.conv_ReL = ConvLayer(2 * channel, channel, s, stride)
            self.conv_ReH = ConvLayer(2 * channel, channel, s, stride)
        else:
            # add
            self.conv_ReL = ConvLayer(channel, channel, s, stride)
            self.conv_ReH = ConvLayer(channel, channel, s, stride)

    def forward(self, Z_x, U_x, Z_y, U_y):
        # get loww parts and sparse parts
        x_l = self.conv_ReZx(Z_x)
        x_h = self.conv_ReUx(U_x)
        y_l = self.conv_ReZy(Z_y)
        y_h = self.conv_ReUy(U_y)
        # reconstructure
        if self.type.__contains__('cat'):
            # cat
            low = self.conv_ReL(torch.cat([x_l, y_l], 1))
            high = self.conv_ReH(torch.cat([x_h, y_h], 1))
        else:
            # add
            low = self.conv_ReL(x_l + y_l)
            high = self.conv_ReH(x_h + y_h)
        out = low + high
        return out, x_l, x_h, y_l, y_h, low, high


class LRR_NET(nn.Module):
    def __init__(self, s, n, channel, stride, num_block, fusion_type):
        super(LRR_NET, self).__init__()
        self.get_ls1 = GetLS_Net(s, n, channel, stride, num_block)
        self.get_ls2 = GetLS_Net(s, n, channel, stride, num_block)

        # self.get_ls1 = GetLS_Conv8(s, n, channel, stride, num_block)
        # self.get_ls2 = GetLS_Conv8(s, n, channel, stride, num_block)

        #self.get_ls1 = GetLS_dense(s, n, channel, stride, num_block)
        #self.get_ls2 = GetLS_dense(s, n, channel, stride, num_block)

        self.decoder = Decoder_simple(s, n, channel, stride, fusion_type)

    def forward(self, x, y):
        fea_x_l, fea_x_s = self.get_ls1(x)
        fea_y_l, fea_y_s = self.get_ls2(y)
        f, x_l, x_h, y_l, y_h, fl, fh = self.decoder(fea_x_l, fea_x_s, fea_y_l, fea_y_s)
        out = {'fea_x_l': fea_x_l, 'fea_x_s': fea_x_s,
               'fea_y_l': fea_y_l, 'fea_y_s': fea_y_s,
               'x_l': x_l, 'x_h': x_h,
               'y_l': y_l, 'y_h': y_h,
               'fl': fl, 'fh': fh,
               'fuse': f}
        return out


class Vgg16(torch.nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

    def forward(self, X):
        b, c, h, w = X.shape
        if c == 1:
            X = X.repeat(1, 3, 1, 1)

        h = F.relu(self.conv1_1(X))
        h = F.relu(self.conv1_2(h))
        relu1_2 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        relu2_2 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))
        relu3_3 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h))
        relu4_3 = h
        # [relu1_2, relu2_2, relu3_3, relu4_3]
        return [relu1_2, relu2_2, relu3_3, relu4_3]
