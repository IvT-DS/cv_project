import torch.nn as nn


class DenoiseEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.Dropout(),
            nn.SELU(),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.Dropout(),
            nn.SELU(),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1), 
            nn.BatchNorm2d(128), 
            nn.Tanh()
        )

        self.layer1_t = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.Dropout(),
            nn.SELU(),
        )
        self.layer2_t = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.Dropout(),
            nn.LeakyReLU(),
        )
        self.layer3_t = nn.Sequential(
            nn.ConvTranspose2d(32, 1, kernel_size=3, padding=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

    def decode(self, x):
        x = self.layer1_t(x)
        x = self.layer2_t(x)
        x = self.layer3_t(x)
        return x

    def forward(self, x):
        latent = self.encode(x)
        out = self.decode(latent)
        return out
