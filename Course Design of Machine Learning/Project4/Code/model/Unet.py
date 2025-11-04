import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=1, base_c=64):
        super(UNet, self).__init__()

        self.encoder1 = DoubleConv(in_channels, base_c)
        self.pool1 = nn.MaxPool2d(2)

        self.encoder2 = DoubleConv(base_c, base_c * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.encoder3 = DoubleConv(base_c * 2, base_c * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.encoder4 = DoubleConv(base_c * 4, base_c * 8)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(base_c * 8, base_c * 16)

        self.upconv4 = nn.ConvTranspose2d(base_c * 16, base_c * 8, kernel_size=2, stride=2)
        self.decoder4 = DoubleConv(base_c * 16, base_c * 8)

        self.upconv3 = nn.ConvTranspose2d(base_c * 8, base_c * 4, kernel_size=2, stride=2)
        self.decoder3 = DoubleConv(base_c * 8, base_c * 4)

        self.upconv2 = nn.ConvTranspose2d(base_c * 4, base_c * 2, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(base_c * 4, base_c * 2)

        self.upconv1 = nn.ConvTranspose2d(base_c * 2, base_c, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(base_c * 2, base_c)

        self.final_conv = nn.Conv2d(base_c, num_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool1(e1))
        e3 = self.encoder3(self.pool2(e2))
        e4 = self.encoder4(self.pool3(e3))

        b = self.bottleneck(self.pool4(e4))

        d4 = self.upconv4(b)
        d4 = self.decoder4(torch.cat([d4, e4], dim=1))

        d3 = self.upconv3(d4)
        d3 = self.decoder3(torch.cat([d3, e3], dim=1))

        d2 = self.upconv2(d3)
        d2 = self.decoder2(torch.cat([d2, e2], dim=1))

        d1 = self.upconv1(d2)
        d1 = self.decoder1(torch.cat([d1, e1], dim=1))

        return self.final_conv(d1)