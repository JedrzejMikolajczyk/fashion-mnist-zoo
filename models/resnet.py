"""
based on Dive into Deep Learning
https://d2l.ai/d2l-en.pdf#14f
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class Residual(nn.Module):
    """The Residual block of ResNet."""
    def __init__(self, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.LazyConv2d(num_channels, kernel_size=3, padding=1,
                                   stride=strides)
        self.conv2 = nn.LazyConv2d(num_channels, kernel_size=3, padding=1)
        
        if use_1x1conv:
            self.conv3 = nn.LazyConv2d(num_channels, kernel_size=1,
                                   stride=strides)
        else:
            self.conv3 = None
            
        self.bn1 = nn.LazyBatchNorm2d()
        self.bn2 = nn.LazyBatchNorm2d()
        
    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        
        if self.conv3:
            X = self.conv3(X)
            
        Y += X
        return F.relu(Y)
    
class ResNet(nn.Module):
    def __init__(self, arch = ((2, 64), (2, 128), (2, 256), (2, 512)), num_classes=10):
        super(ResNet, self).__init__()
        self.net = nn.Sequential(self.block1())
        
        for i, b in enumerate(arch):
            self.net.add_module(f'block{i+2}', self.build_block(*b, first_block=(i==0)))
            
        self.net.add_module('last', nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.LazyLinear(num_classes)))
        
    def block1(self):
        return nn.Sequential(
            nn.LazyConv2d(64, kernel_size=7, stride=2, padding=3),
            nn.LazyBatchNorm2d(), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        
    def build_block(self, num_residuals, num_channels, first_block=False):
        block = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                block.append(Residual(num_channels, use_1x1conv=True, strides=2))
            else:
                block.append(Residual(num_channels))
        return nn.Sequential(*block)
    
    def forward(self, X):
       for layer in self.net:
           X = layer(X)
       return X