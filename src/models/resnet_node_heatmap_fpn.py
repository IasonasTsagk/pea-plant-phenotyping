"""ResNet-18 FPN-style multi-scale heatmap decoder for Phase 7D."""
import math
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18,ResNet18_Weights
class ResNet18NodeHeatmapFPN(nn.Module):
 def __init__(self,heatmap_prior_probability=.10,channels=64):
  super().__init__(); b=resnet18(weights=ResNet18_Weights.DEFAULT); self.stem=nn.Sequential(b.conv1,b.bn1,b.relu,b.maxpool); self.l1,self.l2,self.l3,self.l4=b.layer1,b.layer2,b.layer3,b.layer4
  self.lat1,self.lat2,self.lat3,self.lat4=nn.Conv2d(64,channels,1),nn.Conv2d(128,channels,1),nn.Conv2d(256,channels,1),nn.Conv2d(512,channels,1)
  self.fuse3,self.fuse2,self.fuse1=[nn.Sequential(nn.Conv2d(channels,channels,3,padding=1),nn.ReLU()) for _ in range(3)]; self.head=nn.Sequential(nn.Conv2d(channels,channels,3,padding=1),nn.ReLU(),nn.Conv2d(channels,1,1)); nn.init.constant_(self.head[-1].bias,math.log(heatmap_prior_probability/(1-heatmap_prior_probability)))
 def features(self,x):
  c1=self.l1(self.stem(x)); c2=self.l2(c1); c3=self.l3(c2); c4=self.l4(c3); return c1,c2,c3,c4
 def forward(self,x):
  c1,c2,c3,c4=self.features(x); p4=self.lat4(c4); p3=self.fuse3(F.interpolate(p4,size=c3.shape[-2:],mode='bilinear',align_corners=False)+self.lat3(c3)); p2=self.fuse2(F.interpolate(p3,size=c2.shape[-2:],mode='bilinear',align_corners=False)+self.lat2(c2)); p1=self.fuse1(F.interpolate(p2,size=c1.shape[-2:],mode='bilinear',align_corners=False)+self.lat1(c1)); return self.head(p1)
