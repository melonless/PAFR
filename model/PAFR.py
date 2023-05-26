import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from .resnet_model import *


class OPM(nn.Module):
    def __init__(self, in_channel):
        super(OPM, self).__init__()
        self.convx0 = nn.Conv2d(in_channel, 512, kernel_size=3, stride=1, padding=1)
        self.bnx0 = nn.BatchNorm2d(512)
        self.convx1 = nn.Conv2d(in_channel, 1024, kernel_size=3, stride=1, padding=1)

        self.convx2 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0)
        self.bnx2   = nn.BatchNorm2d(512)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
                nn.Conv2d(512, 32, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 512, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        x0 = x
        x_0 = F.relu(self.bnx0(self.convx0(x0)), inplace=True)  # 256 channels
        x_1 = self.convx1(x)  # wb
        w, b = x_1[:, :512, :, :], x_1[:, 512:, :, :]
        f_0 = F.relu(w * x_0 + b, inplace=True)

        x_2 = F.relu(self.bnx2(self.convx2(x0)), inplace=True)
        f_1 = self.avg_pool(f_0)
        f_2 = self.conv_du(f_1)
        f_2 = x_2 * f_2
        # fout = f_2 + x0
        return f_2


class CRM(nn.Module):
    def __init__(self, in_channel_l, in_channel_d, out_channel):
        super(CRM, self).__init__()
        self.conv_l1 = nn.Conv2d(in_channel_l, out_channel, kernel_size=3, stride=1, padding=1)
        self.bn_l1 = nn.BatchNorm2d(out_channel)
        self.conv_d1 = nn.Conv2d(in_channel_d, out_channel, kernel_size=3, stride=1, padding=1)
        self.bn_d1 = nn.BatchNorm2d(out_channel)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
                nn.Conv2d(out_channel, out_channel // 16, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channel // 16, out_channel, 1, padding=0, bias=True),
        )
        self.conv_d2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(out_channel * 2, out_channel, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(out_channel)

    def forward(self, left, down):
        left_ = F.relu(self.bn_l1(self.conv_l1(left)), inplace=True)
        down_ = F.relu(self.bn_d1(self.conv_d1(down)), inplace=True)
        down_1 = self.conv_d2(down_)

        att1 = self.avg_pool(left_)  # n=1
        att1 = self.conv_du(att1)
        att1_1 = torch.sigmoid(att1)  # 1x1

        att1_2 = att1_1*left_

        if down_1.size()[2:] != left_.size()[2:]:
            down_2 = F.interpolate(down_1, size=left_.size()[2:], mode='bilinear', align_corners=False)
            att2_ = F.relu(left_ * down_2, inplace=True)
        else:
            att2_ = F.relu(left_ * down_1, inplace=True)

        att1_3 = att1_2 + att2_

        f0 = torch.cat((att1_3, att2_), dim=1)
        return F.relu(self.bn3(self.conv3(f0)), inplace=True), down_1


class RefUnet(nn.Module):

    def __init__(self, in_ch, inc_ch):
        super(RefUnet, self).__init__()

        self.conv0 = nn.Conv2d(in_ch, inc_ch, 3, padding=1)

        self.conv1 = nn.Conv2d(inc_ch, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)

        self.pool1 = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)

        self.pool2 = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)

        self.pool3 = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU(inplace=True)

        self.pool4 = nn.MaxPool2d(2, 2, ceil_mode=True)

        #####

        self.conv5 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.relu5 = nn.ReLU(inplace=True)

        #####

        self.conv_d4 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn_d4 = nn.BatchNorm2d(64)
        self.relu_d4 = nn.ReLU(inplace=True)

        self.conv_d3 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn_d3 = nn.BatchNorm2d(64)
        self.relu_d3 = nn.ReLU(inplace=True)

        self.conv_d2 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn_d2 = nn.BatchNorm2d(64)
        self.relu_d2 = nn.ReLU(inplace=True)

        self.conv_d1 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn_d1 = nn.BatchNorm2d(64)
        self.relu_d1 = nn.ReLU(inplace=True)

        self.conv_d0 = nn.Conv2d(64, 1, 3, padding=1)

        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x):

        hx = x
        hx = self.conv0(hx)

        hx1 = self.relu1(self.bn1(self.conv1(hx)))
        hx = self.pool1(hx1)

        hx2 = self.relu2(self.bn2(self.conv2(hx)))
        hx = self.pool2(hx2)

        hx3 = self.relu3(self.bn3(self.conv3(hx)))
        hx = self.pool3(hx3)

        hx4 = self.relu4(self.bn4(self.conv4(hx)))
        hx = self.pool4(hx4)

        hx5 = self.relu5(self.bn5(self.conv5(hx)))

        hx = self.upscore2(hx5)

        d4 = self.relu_d4(self.bn_d4(self.conv_d4(torch.cat((hx, hx4), 1))))
        hx = self.upscore2(d4)

        d3 = self.relu_d3(self.bn_d3(self.conv_d3(torch.cat((hx, hx3), 1))))
        hx = self.upscore2(d3)

        d2 = self.relu_d2(self.bn_d2(self.conv_d2(torch.cat((hx, hx2), 1))))
        hx = self.upscore2(d2)

        d1 = self.relu_d1(self.bn_d1(self.conv_d1(torch.cat((hx, hx1), 1))))

        residual = self.conv_d0(d1)

        return x + residual


class Net(nn.Module):
    def __init__(self, n_channels):
        super(Net, self).__init__()

        resnet = models.resnet34(pretrained=True)

        # -------------Encoder--------------

        self.inconv = nn.Conv2d(n_channels, 64, 3, padding=1)
        self.inbn = nn.BatchNorm2d(64)
        self.inrelu = nn.ReLU(inplace=True)

        # stage 1
        self.encoder1 = resnet.layer1  # 224
        # stage 2
        self.encoder2 = resnet.layer2  # 112
        # stage 3
        self.encoder3 = resnet.layer3  # 56
        # stage 4
        self.encoder4 = resnet.layer4  # 28

        self.pool4 = nn.MaxPool2d(2, 2, ceil_mode=True)

        # stage 5
        self.resb5_1 = BasicBlock(512, 512)
        self.resb5_2 = BasicBlock(512, 512)
        self.resb5_3 = BasicBlock(512, 512)  # 14

        self.pool5 = nn.MaxPool2d(2, 2, ceil_mode=True)

        # stage 6
        self.resb6_1 = BasicBlock(512, 512)
        self.resb6_2 = BasicBlock(512, 512)
        self.resb6_3 = BasicBlock(512, 512)  # 7

        # -------------Bridge--------------
        # stage Bridge
        self.convbg_1 = nn.Conv2d(512, 512, 3, dilation=2, padding=2)  # 7
        self.bnbg_1 = nn.BatchNorm2d(512)
        self.relubg_1 = nn.ReLU(inplace=True)
        self.convbg_m = nn.Conv2d(512, 512, 3, dilation=2, padding=2)
        self.bnbg_m = nn.BatchNorm2d(512)
        self.relubg_m = nn.ReLU(inplace=True)
        self.convbg_2 = nn.Conv2d(512, 512, 3, dilation=2, padding=2)
        self.bnbg_2 = nn.BatchNorm2d(512)
        self.relubg_2 = nn.ReLU(inplace=True)

        # -------------Decoder--------------
        self.aia0 = OPM(512)
        self.afm5 = CRM(512, 512, 512)
        self.afm4 = CRM(512, 512, 256)
        self.afm3 = CRM(256, 256, 128)
        self.afm2 = CRM(128, 128, 64)
        self.afm1 = CRM(64, 64, 64)

        # stage 6d
        self.conv6d_1 = nn.Conv2d(1024, 512, 3, padding=1)  # 16
        self.bn6d_1 = nn.BatchNorm2d(512)
        self.relu6d_1 = nn.ReLU(inplace=True)

        self.conv6d_m = nn.Conv2d(512, 512, 3, dilation=2, padding=2)  #
        self.bn6d_m = nn.BatchNorm2d(512)
        self.relu6d_m = nn.ReLU(inplace=True)

        self.conv6d_2 = nn.Conv2d(512, 512, 3, dilation=2, padding=2)
        self.bn6d_2 = nn.BatchNorm2d(512)
        self.relu6d_2 = nn.ReLU(inplace=True)

        # -------------Bilinear Upsampling--------------
        self.upscorea = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=False)  #
        self.upscore5 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=False)
        self.upscore4 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)
        self.upscore3 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        # -------------Side Output--------------
        self.outconva = nn.Conv2d(512, 1, 3, padding=1)

        self.outconv5 = nn.Conv2d(512, 1, 3, padding=1)
        self.outconv4 = nn.Conv2d(256, 1, 3, padding=1)
        self.outconv3 = nn.Conv2d(128, 1, 3, padding=1)
        self.outconv2 = nn.Conv2d(64, 1, 3, padding=1)
        self.outconv1 = nn.Conv2d(64, 1, 3, padding=1)

        # -------------Refine Module-------------
        self.refunet = RefUnet(1, 64)

        # -------------Global Attention-------------
        self.conv_attout = nn.Conv2d(1024, 64, 3, padding=1)
        self.bn_attout = nn.BatchNorm2d(64)
        self.outconvatt = nn.Conv2d(64, 1, 3, padding=1)

    def forward(self, x):
        hx = x

        # -------------Encoder-------------
        hx = self.inconv(hx)
        hx = self.inbn(hx)
        hx = self.inrelu(hx)

        h1 = self.encoder1(hx)  # 256
        h2 = self.encoder2(h1)  # 128
        h3 = self.encoder3(h2)  # 64
        h4 = self.encoder4(h3)  # 32

        hx = self.pool4(h4)  # 16

        hx = self.resb5_1(hx)
        hx = self.resb5_2(hx)
        h5 = self.resb5_3(hx)  # [1, 512, 14, 14]

        hda = self.aia0(h5)  # [1, 512, 14, 14]

        # -------------Decoder-------------

        hd5, l5 = self.afm5(hda, h5)    # [1, 512, 14, 14], [1, 512, 14, 14]
        hx = self.upscore2(hd5)            # [1, 512, 28, 28]
        hd4, l4 = self.afm4(hx, h4)    # [1, 256, 28, 28], [1, 256, 28, 28]
        hx = self.upscore2(hd4)            # [1, 256, 56, 56]
        hd3, l3 = self.afm3(hx, h3)    # [1, 128, 56, 56], [1, 128, 56, 56]
        hx = self.upscore2(hd3)            # [1, 128, 112, 112]
        hd2, l2 = self.afm2(hx, h2)    # [1, 64, 112, 112], [1, 64, 112, 112]
        hx = self.upscore2(hd2)            # [1, 64, 224, 224]
        hd1, l1 = self.afm1(hx, h1)    # [1, 64, 224, 224], [1, 64, 224, 224]

        # -------------Side Output-------------

        da = self.outconva(hda)
        da = torch.sigmoid(self.upscorea(da))  # 512_in

        d50 = self.outconv5(hd5)
        d5 = torch.sigmoid(self.upscore5(d50))  # 512_in

        d40 = self.outconv4(hd4)
        d4 = torch.sigmoid(self.upscore4(d40))  # 256_in

        d30 = self.outconv3(hd3)
        d3 = torch.sigmoid(self.upscore3(d30))  # 128_in

        d20 = self.outconv2(hd2)
        d2 = torch.sigmoid(self.upscore2(d20))  # 64_in

        d10 = self.outconv1(hd1)  # 64_in [1, 1, 224, 224]
        d1 = torch.sigmoid(d10)

        # -------------Refine Module-------------
        dout = torch.sigmoid(self.refunet(d10))

        return dout, d1, d2, d3, d4, d5, da
