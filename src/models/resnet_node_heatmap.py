"""Pretrained ResNet-18 encoder with a lightweight single-channel heatmap decoder."""
import math
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
class ResNet18NodeHeatmap(nn.Module):
    def __init__(self, heatmap_prior_probability=0.10):
        super().__init__(); b=resnet18(weights=ResNet18_Weights.DEFAULT); self.stem=nn.Sequential(b.conv1,b.bn1,b.relu,b.maxpool); self.l1=b.layer1; self.l2=b.layer2; self.l3=b.layer3; self.l4=b.layer4; self.head=nn.Sequential(nn.Conv2d(512,128,3,padding=1),nn.ReLU(),nn.Conv2d(128,64,3,padding=1),nn.ReLU(),nn.Conv2d(64,1,1))
        if heatmap_prior_probability is not None:
            if not 0 < heatmap_prior_probability < 1: raise ValueError('heatmap_prior_probability must be in (0,1)')
            nn.init.constant_(self.head[-1].bias, math.log(heatmap_prior_probability/(1-heatmap_prior_probability)))
    def forward(self,x): return F.interpolate(self.head(self.l4(self.l3(self.l2(self.l1(self.stem(x)))))),size=(144,192),mode='bilinear',align_corners=False)
