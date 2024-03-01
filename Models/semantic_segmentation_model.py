import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.conv_down1 = self.conv_block(3, 64)
        self.conv_down2 = self.conv_block(64, 128)
        self.conv_down3 = self.conv_block(128, 256)
        self.conv_down4 = self.conv_block(256, 512)

        self.bottleneck = self.conv_block(512, 1024)

        self.conv_up1 = self.conv_block(1024, 512)
        self.conv_up2 = self.conv_block(512, 256)
        self.conv_up3 = self.conv_block(256, 128)
        self.conv_up4 = self.conv_block(128, 64)

        self.output = nn.Conv2d(64, 1, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        down1 = self.conv_down1(x)
        down_pool1 = nn.MaxPool2d(kernel_size=2)(down1)
        down2 = self.conv_down2(down_pool1)
        down_pool2 = nn.MaxPool2d(kernel_size=2)(down2)
        down3 = self.conv_down3(down_pool2)
        down_pool3 = nn.MaxPool2d(kernel_size=2)(down3)
        down4 = self.conv_down4(down_pool3)
        down_pool4 = nn.MaxPool2d(kernel_size=2)(down4)

        bottleneck = self.bottleneck(down_pool4)

        up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)(bottleneck)
        up1 = torch.cat((up1, down4), dim=1)
        up1 = self.conv_up1(up1)
        up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)(up1)
        up2 = torch.cat((up2, down3), dim=1)
        up2 = self.conv_up2(up2)
        up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)(up2)
        up3 = torch.cat((up3, down2), dim=1)
        up3 = self.conv_up3(up3)
        up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)(up3)
        up4 = torch.cat((up4, down1), dim=1)
        up4 = self.conv_up4(up4)

        output = nn.Sigmoid()(self.output(up4))
        return output
