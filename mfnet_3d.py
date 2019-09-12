"""
Author: Yunpeng Chen
"""
import logging
import os
from collections import OrderedDict
import torch
import torch.nn as nn


class BN_AC_CONV3D(nn.Module):

    def __init__(self, num_in, num_filter,
                 kernel=(1, 1, 1), pad=(0, 0, 0), stride=(1, 1, 1), g=1, bias=False):
        super(BN_AC_CONV3D, self).__init__()
        self.bn = nn.BatchNorm3d(num_in)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv3d(num_in, num_filter, kernel_size=kernel, padding=pad,
                              stride=stride, groups=g, bias=bias)

    def forward(self, x):
        h = self.relu(self.bn(x))
        h = self.conv(h)
        return h


class MF_UNIT(nn.Module):

    def __init__(self, num_in, num_mid, num_out, g=1, stride=(1, 1, 1), first_block=False, use_3d=True):
        super(MF_UNIT, self).__init__()
        num_ix = int(num_mid / 4)
        kt, pt = (3, 1) if use_3d else (1, 0)
        # prepare input
        self.conv_i1 = BN_AC_CONV3D(num_in=num_in, num_filter=num_ix, kernel=(1, 1, 1), pad=(0, 0, 0))
        self.conv_i2 = BN_AC_CONV3D(num_in=num_ix, num_filter=num_in, kernel=(1, 1, 1), pad=(0, 0, 0))
        # main part
        self.conv_m1 = BN_AC_CONV3D(num_in=num_in, num_filter=num_mid, kernel=(kt, 3, 3), pad=(pt, 1, 1), stride=stride,
                                    g=g)
        if first_block:
            self.conv_m2 = BN_AC_CONV3D(num_in=num_mid, num_filter=num_out, kernel=(1, 1, 1), pad=(0, 0, 0))
        else:
            self.conv_m2 = BN_AC_CONV3D(num_in=num_mid, num_filter=num_out, kernel=(1, 3, 3), pad=(0, 1, 1), g=g)
        # adapter
        if first_block:
            self.conv_w1 = BN_AC_CONV3D(num_in=num_in, num_filter=num_out, kernel=(1, 1, 1), pad=(0, 0, 0),
                                        stride=stride)

    def forward(self, x):

        h = self.conv_i1(x)
        x_in = x + self.conv_i2(h)

        h = self.conv_m1(x_in)
        h = self.conv_m2(h)

        if hasattr(self, 'conv_w1'):
            x = self.conv_w1(x)

        return h + x


class MFNET_3D(nn.Module):

    def __init__(self, num_classes, pretrained=False, **kwargs):
        super(MFNET_3D, self).__init__()
        groups = 16
        k_sec = {2: 3, \
                 3: 4, \
                 4: 6, \
                 5: 3}

        # conv1 - x224 (x16)
        conv1_num_out = 16
        self.conv1 = nn.Sequential(OrderedDict([
            ('conv',
             nn.Conv3d(3, conv1_num_out, kernel_size=(3, 5, 5), padding=(1, 2, 2), stride=(1, 2, 2), bias=False)),
            ('bn', nn.BatchNorm3d(conv1_num_out)),
            ('relu', nn.ReLU(inplace=True))
        ]))
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        # conv2 - x56 (x8)
        num_mid = 96
        conv2_num_out = 96
        self.conv2 = nn.Sequential(OrderedDict([
            ("B%02d" % i, MF_UNIT(num_in=conv1_num_out if i == 1 else conv2_num_out,
                                  num_mid=num_mid,
                                  num_out=conv2_num_out,
                                  stride=(2, 1, 1) if i == 1 else (1, 1, 1),
                                  g=groups,
                                  first_block=(i == 1))) for i in range(1, k_sec[2] + 1)
        ]))

        # conv3 - x28 (x8)
        num_mid *= 2
        conv3_num_out = 2 * conv2_num_out
        self.conv3 = nn.Sequential(OrderedDict([
            ("B%02d" % i, MF_UNIT(num_in=conv2_num_out if i == 1 else conv3_num_out,
                                  num_mid=num_mid,
                                  num_out=conv3_num_out,
                                  stride=(1, 2, 2) if i == 1 else (1, 1, 1),
                                  g=groups,
                                  first_block=(i == 1))) for i in range(1, k_sec[3] + 1)
        ]))

        # conv4 - x14 (x8)
        num_mid *= 2
        conv4_num_out = 2 * conv3_num_out
        self.conv4 = nn.Sequential(OrderedDict([
            ("B%02d" % i, MF_UNIT(num_in=conv3_num_out if i == 1 else conv4_num_out,
                                  num_mid=num_mid,
                                  num_out=conv4_num_out,
                                  stride=(1, 2, 2) if i == 1 else (1, 1, 1),
                                  g=groups,
                                  first_block=(i == 1))) for i in range(1, k_sec[4] + 1)
        ]))

        # conv5 - x7 (x8)
        num_mid *= 2
        conv5_num_out = 2 * conv4_num_out
        self.conv5 = nn.Sequential(OrderedDict([
            ("B%02d" % i, MF_UNIT(num_in=conv4_num_out if i == 1 else conv5_num_out,
                                  num_mid=num_mid,
                                  num_out=conv5_num_out,
                                  stride=(1, 2, 2) if i == 1 else (1, 1, 1),
                                  g=groups,
                                  first_block=(i == 1))) for i in range(1, k_sec[5] + 1)
        ]))

        # final
        self.tail = nn.Sequential(OrderedDict([
            ('bn', nn.BatchNorm3d(conv5_num_out)),
            ('relu', nn.ReLU(inplace=True))
        ]))

        self.globalpool = nn.Sequential(OrderedDict([
            ('avg', nn.AvgPool3d(kernel_size=(8, 7, 7), stride=(1, 1, 1))),
            ('dropout', nn.Dropout(p=0.5))  # 0.5))
        ]))

        self.classifier = nn.Linear(conv5_num_out, num_classes)

        # self.nlb0  = _NonLocalBlock(16, inter_channels=4, sub_sample=True, bn_layer=True)
        """
        self.nlb1  = _NonLocalBlock(96, inter_channels=16, sub_sample=True, bn_layer=True)
        self.nlb2  = _NonLocalBlock(192, inter_channels=32, sub_sample=True, bn_layer=True)
        self.nlb3  = _NonLocalBlock(384, inter_channels=64, sub_sample=True, bn_layer=True)
        
        
        
        self.conv_align   = BN_AC_CONV3D(6, 3,kernel=(1,1,1), pad=(0,0,0), stride=(1,1,1), g=1, bias=False)
        self.conv_align1  = BN_AC_CONV3D(6, 3,kernel=(1,1,1), pad=(0,0,0), stride=(1,1,1), g=1, bias=False)
        self.conv_align2  = BN_AC_CONV3D(6, 3,kernel=(1,1,1), pad=(0,0,0), stride=(1,1,1), g=1, bias=False)
        self.conv_align3  = BN_AC_CONV3D(6, 3,kernel=(1,1,1), pad=(0,0,0), stride=(1,1,1), g=1, bias=False)
        self.conv_align4  = BN_AC_CONV3D(6, 3,kernel=(1,1,1), pad=(0,0,0), stride=(1,1,1), g=1, bias=False)
        self.conv_align5  = BN_AC_CONV3D(6, 3,kernel=(1,1,1), pad=(0,0,0), stride=(1,1,1), g=1, bias=False)
        self.conv_align6  = BN_AC_CONV3D(6, 3,kernel=(1,1,1), pad=(0,0,0), stride=(1,1,1), g=1, bias=False)
        self.conv_align7  = BN_AC_CONV3D(6, 3,kernel=(1,1,1), pad=(0,0,0), stride=(1,1,1), g=1, bias=False)
        self.conv_align8  = BN_AC_CONV3D(6, 3,kernel=(1,1,1), pad=(0,0,0), stride=(1,1,1), g=1, bias=False)
        self.conv_align9  = BN_AC_CONV3D(6, 3,kernel=(1,1,1), pad=(0,0,0), stride=(1,1,1), g=1, bias=False)
        self.conv_align10 = BN_AC_CONV3D(6, 3,kernel=(1,1,1), pad=(0,0,0), stride=(1,1,1), g=1, bias=False)
        """

    def forward(self, x):
        assert x.shape[2] == 16

        """
        out   = self.conv_align(torch.unsqueeze(torch.cat((x[:,:,0,:,:],x[:,:,1,:,:]),dim = 1),dim=2))
        out1  = self.conv_align1(torch.unsqueeze(torch.cat((x[:,:,0,:,:],x[:,:,2,:,:]),dim = 1),dim=2))
        out2  = self.conv_align2(torch.unsqueeze(torch.cat((x[:,:,0,:,:],x[:,:,3,:,:]),dim = 1),dim=2))
        out3  = self.conv_align3(torch.unsqueeze(torch.cat((x[:,:,0,:,:],x[:,:,4,:,:]),dim = 1),dim=2))
        out4  = self.conv_align4(torch.unsqueeze(torch.cat((x[:,:,0,:,:],x[:,:,5,:,:]),dim = 1),dim=2))
        out5  = self.conv_align5(torch.unsqueeze(torch.cat((x[:,:,0,:,:],x[:,:,6,:,:]),dim = 1),dim=2))
        out6  = self.conv_align6(torch.unsqueeze(torch.cat((x[:,:,0,:,:],x[:,:,7,:,:]),dim = 1),dim=2))
        out7  = self.conv_align7(torch.unsqueeze(torch.cat((x[:,:,0,:,:],x[:,:,8,:,:]),dim = 1),dim=2))
        out8  = self.conv_align8(torch.unsqueeze(torch.cat((x[:,:,0,:,:],x[:,:,9,:,:]),dim = 1),dim=2))
        out9  = self.conv_align9(torch.unsqueeze(torch.cat((x[:,:,0,:,:],x[:,:,10,:,:]),dim = 1),dim=2))
        out10 = self.conv_align10(torch.unsqueeze(torch.cat((x[:,:,0,:,:],x[:,:,11,:,:]),dim = 1),dim=2))
        x = torch.cat((torch.unsqueeze(x[:,:,0,:,:],2),out,out1,out2,out3,out4,out5,out6,out7,out8,out9,out10),dim=2)
        """

        h = self.conv1(x)  # x224 -> x112
        h = self.maxpool(h)  # x112 ->  x56
        # h = self.nlb0(h)
        h = self.conv2(h)  # x56 ->  x56
        # h = self.nlb1(h)
        h = self.conv3(h)  # x56 ->  x28
        # h = self.nlb2(h)
        h = self.conv4(h)  # x28 ->  x14
        # h = self.nlb3(h)
        h = self.conv5(h)  # x14 ->   x7

        h = self.tail(h)
        h = self.globalpool(h)

        h = h.view(h.shape[0], -1)
        h = self.classifier(h)

        return h


if __name__ == "__main__":
    import torch

    logging.getLogger().setLevel(logging.DEBUG)
    # ---------
    net = MFNET_3D(num_classes=100, pretrained=False)
    data = torch.autograd.Variable(torch.randn(1, 3, 16, 224, 224))
    output = net(data)
    torch.save({'state_dict': net.state_dict()}, './tmp.pth')
    print(output.shape)
