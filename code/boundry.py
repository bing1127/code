import torch.nn as nn
import torch



class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x
class EDGModule(nn.Module):
    def __init__(self, dims):
        super(EDGModule, self).__init__()
        self.relu = nn.ReLU(True)
        self.conv_upsample1 = BasicConv2d(dims[0], 64, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(dims[1], 64, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(dims[2], 64, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(dims[3], 64, 3, padding=1)

        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.upsample4 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.upsample6 = nn.UpsamplingBilinear2d(scale_factor=8)

        self.upsample5 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv_concat2 = BasicConv2d(2*64, 64, 3, padding=1)
        self.conv_concat3 = BasicConv2d(2*64, 64, 3, padding=1)
        self.conv_concat4 = BasicConv2d(2 * 64, 64, 3, padding=1)

    def forward(self, x1, x2, x3,x4):
        x4 = self.upsample(x4)
        conv_x4 = self.conv_upsample4(x4)
        conv_x3 = self.conv_upsample3(x3)
        conv_x2 = self.conv_upsample2(x2)
        conv_x1 = self.conv_upsample1(x1)
        catx3x4 = self.conv_concat2(torch.cat((conv_x3, conv_x4), 1))
        catx3x4 = self.upsample(catx3x4)
        catx2x3x4 = self.conv_concat2(torch.cat((conv_x2, catx3x4), 1))
        catx2x3x4 = self.upsample(catx2x3x4)
        catx1x2x3x4 = self.conv_concat2(torch.cat((conv_x1, catx2x3x4), 1))
        x = self.upsample(catx1x2x3x4)

        return x