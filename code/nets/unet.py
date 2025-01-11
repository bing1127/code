import torch
import torch.nn as nn
from Segmentataion_inter import EDGModule
from nets.resnet import resnet50
from nets.vgg import VGG16
#from attention import eca_block

class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs



class Unet(nn.Module):
    def __init__(self, num_classes=21, pretrained=True, backbone='vgg',dims=[128,256,512]):
        super(Unet, self).__init__()
        if backbone == 'vgg':
            self.vgg = VGG16(pretrained=pretrained)
            in_filters = [192, 384, 768, 1024]
        elif backbone == "resnet50":
            self.resnet = resnet50(pretrained=pretrained)
            in_filters = [192, 512, 1024, 3072]
        else:
            raise ValueError('Unsupported backbone - `{}`, Use vgg, resnet50.'.format(backbone))
        out_filters = [64, 128, 256, 512]

        # upsampling
        # 64,64,512
        self.up_concat4 = unetUp(in_filters[3], out_filters[3])
        # 128,128,256
        self.up_concat3 = unetUp(in_filters[2], out_filters[2])
        # 256,256,128
        self.up_concat2 = unetUp(in_filters[1], out_filters[1])
        # 512,512,64
        self.up_concat1 = unetUp(in_filters[0], out_filters[0])
       # self.conv5 = nn.Conv2d(64, 2, 1, 1)
        if backbone == 'resnet50':
            self.up_conv = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
                nn.ReLU(),
            )
        else:
            self.up_conv = None
        self.side_output = nn.Conv2d(in_channels=128,out_channels=2,kernel_size=3)

        self.final = nn.Conv2d(out_filters[0], num_classes, 1)
        self.edge = EDGModule(dims)
        self.backbone = backbone
      #  self.attention_edge=eca_block(128)
      #  self.attention_up1=eca_block(64)

    def forward(self, inputs):
        if self.backbone == "vgg":
            [feat1, feat2, feat3, feat4, feat5] = self.vgg.forward(inputs)
        elif self.backbone == "resnet50":
            [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(inputs)

        up4 = self.up_concat4(feat4, feat5)#512

        up3 = self.up_concat3(feat3, up4)#256

        up2 = self.up_concat2(feat2, up3)#128

        up1 = self.up_concat1(feat1, up2)#64
       # up1=self.attention_up1(up1)
        edge_feat = self.edge(up2, up3, up4)#128
       # edge_feat=self.attention_edge(edge_feat)

        if self.up_conv != None:
            up1 = self.up_conv(up1)
       # a= torch.cat([up1, edge_feat],1)
        x = torch.cat((up1, edge_feat),1)
        y=self.side_output(x)
       # final = self.final(x)
        return y

    def freeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = False
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = True
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = True
