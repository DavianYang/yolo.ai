import torch
import torch.nn as nn

class DarkNet19(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            ConvBlock(3, 32, kernel=3, padding=1),
            nn.MaxPool2d((2, 2), 2)
        )
        self.conv2 = nn.Sequential(
            ConvBlock(32, 64, kernel=3, padding=1),
            nn.MaxPool2d((2, 2), 2)
        )
        self.conv3 = nn.Sequential(
            ConvBlock(64, 128, kernel=3, padding=1),
            ConvBlock(128, 64, kernel=1),
            ConvBlock(64, 128, kernel=3, padding=1),
            nn.MaxPool2d((2, 2), 2)
        )
        self.conv4 = nn.Sequential(
            ConvBlock(128, 256, kernel=3, padding=1),
            ConvBlock(256, 128, kernel=1),
            ConvBlock(128, 256, kernel=3, padding=1),
        )
        self.maxpool_4 = nn.MaxPool2d((2, 2), 2)
        self.conv5 = nn.Sequential(
            ConvBlock(256, 512, kernel=3, padding=1),
            ConvBlock(512, 256, kernel=1),
            ConvBlock(256, 512, kernel=3, padding=1),
            ConvBlock(512, 256, kernel=1),
            ConvBlock(256, 512, kernel=3, padding=1),
        )
        self.maxpool_5 = nn.MaxPool2d((2, 2), 2)
        self.conv6 = nn.Sequential(
            ConvBlock(512, 1024, kernel=3, padding=1),
            ConvBlock(1024, 512, kernel=1),
            ConvBlock(512, 1024, kernel=3, padding=1),
            ConvBlock(1024, 512, kernel=1),
            ConvBlock(512, 1024, kernel=3, padding=1),
        )
        
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        c_4 = self.conv4(x)
        c_5 = self.conv5(self.maxpool_4(c_4))
        c_6 = self.conv6(self.maxpool_5(c_5))
        
        return c_4, c_5, c_6