
# model.py

import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNetPlusPlus(nn.Module):
    def __init__(self, num_classes):
        super(UNetPlusPlus, self).__init__()
        # Encoder
        self.conv0_0 = ConvBlock(3, 64)
        self.pool0 = nn.MaxPool2d(2)
        self.conv1_0 = ConvBlock(64, 128)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2_0 = ConvBlock(128, 256)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3_0 = ConvBlock(256, 512)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4_0 = ConvBlock(512, 1024)

        # Decoder with nested skip connections
        self.up1_0 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv0_1 = ConvBlock(128, 64)

        self.up2_0 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv1_1 = ConvBlock(256, 128)
        self.up1_1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv0_2 = ConvBlock(192, 64)

        self.up3_0 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv2_1 = ConvBlock(512, 256)
        self.up2_1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv1_2 = ConvBlock(384, 128)
        self.up1_2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv0_3 = ConvBlock(256, 64)

        self.up4_0 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv3_1 = ConvBlock(1024, 512)
        self.up3_1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv2_2 = ConvBlock(768, 256)
        self.up2_2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv1_3 = ConvBlock(512, 128)
        self.up1_3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv0_4 = ConvBlock(320, 64)

        # Final classification layer
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool0(x0_0))
        x2_0 = self.conv2_0(self.pool1(x1_0))
        x3_0 = self.conv3_0(self.pool2(x2_0))
        x4_0 = self.conv4_0(self.pool3(x3_0))

        x0_1 = self.conv0_1(torch.cat([x0_0, self.up1_0(x1_0)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up2_0(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up1_1(x1_1)], 1))

        x2_1 = self.conv2_1(torch.cat([x2_0, self.up3_0(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, x2_1], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up1_2(x1_2)], 1))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up4_0(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, x3_1], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, x2_2], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up1_3(x1_3)], 1))

        output = self.final(x0_4)
        return output
