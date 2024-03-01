import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # Encoder
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1)
        self.conv7 = nn.Conv2d(64, 256, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        # Decoder
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv9 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv10 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv11 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv13 = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, x):
        # Encoder
        out = torch.relu(self.conv1(x))
        out = torch.relu(self.conv2(out))
        out = torch.relu(self.conv3(out))
        out = torch.relu(self.conv4(out))
        out = torch.relu(self.conv5(out))
        out = torch.relu(self.conv6(out))
        out = torch.relu(self.conv7(out))
        out = torch.relu(self.conv8(out))

        # Decoder
        out = self.up1(out)
        out = torch.relu(self.conv9(out))
        out = torch.relu(self.conv10(out))
        out = self.up2(out)
        out = torch.relu(self.conv11(out))
        out = torch.relu(self.conv12(out))
        out = torch.sigmoid(self.conv13(out))
        return out
