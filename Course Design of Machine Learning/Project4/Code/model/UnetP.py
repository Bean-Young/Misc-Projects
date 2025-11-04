import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetPlus(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super(UNetPlus, self).__init__()

        # 编码器
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.AvgPool2d(kernel_size=2)

        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.02)
        )
        self.pool2 = nn.AvgPool2d(kernel_size=2)

        self.conv3 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.02)
        )
        self.pool3 = nn.AvgPool2d(kernel_size=2)

        self.conv4 = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.02)
        )
        self.pool4 = nn.AvgPool2d(kernel_size=2)

        self.conv5 = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.02)
        )

        self.pool5 = nn.AvgPool2d(kernel_size=2)

        # 解码器
        self.up7 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv7 = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.up8 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv8 = nn.Sequential(
            nn.Conv2d(256 + 128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.up9 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv9 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.up10 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv10 = nn.Sequential(
            nn.Conv2d(64 + 8, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # 输出层：直接放在 conv10 后面
        self.out_conv = nn.Conv2d(32, num_classes, kernel_size=1)
        # self.up11 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.conv11 = nn.Sequential(
        #     nn.Conv2d(32, 16, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(16, 8, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True)
        # )

        # self.out_conv = nn.Conv2d(8, num_classes, kernel_size=1)

    def forward(self, x):
        # 编码路径
        c1 = self.conv1(x)
        p1 = self.pool1(c1)

        c2 = self.conv2(p1)
        p2 = self.pool2(c2)

        c3 = self.conv3(p2)
        p3 = self.pool3(c3)

        c4 = self.conv4(p3)
        p4 = self.pool4(c4)

        c5 = self.conv5(p4)
        p5 = self.pool5(p4)

        # 解码路径
        up_7 = self.up7(c5)
        merge7 = self.conv7(up_7)

        up_8 = self.up8(merge7)
        merge8 = torch.cat([up_8, c3], dim=1)
        c8 = self.conv8(merge8)

        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, c2], dim=1)
        c9 = self.conv9(merge9)

        up_10 = self.up10(c9)
        merge10 = torch.cat([up_10, c1], dim=1)
        c10 = self.conv10(merge10)

        # up_11 = self.up11(c10)
        # c11 = self.conv11(up_11)

        out = self.out_conv(c10)
        return out