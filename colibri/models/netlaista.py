import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1):
        x1 = self.up(x1)
        return self.conv(x1)


class Decoder(nn.Module):
    def __init__(self, n_channels, bilinear=False, scaling:int=1):
        super(Decoder, self).__init__()
        self.bilinear = bilinear
        factor = 2 if bilinear else 1
        self.up1 = (Up(256 * scaling, 128 * scaling // factor, bilinear))
        self.up2 = (Up(128 * scaling, 64 * scaling // factor, bilinear))
        self.up3 = (Up(64 * scaling, 32 * scaling // factor, bilinear))
        self.up4 = (Up(32 * scaling, 16 * scaling, bilinear))
        self.outc = (OutConv(16 * scaling, n_channels))

    def forward(self, x):
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        logits = self.outc(x)
        return logits
    

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    

class Encoder(nn.Module):
    def __init__(self, n_channels, bilinear=False, scaling:int=1):
        super(Encoder, self).__init__()
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 16 * scaling))
        self.down1 = (Down(16 * scaling, 32 * scaling))
        self.down2 = (Down(32 * scaling, 64 * scaling))
        self.down3 = (Down(64 * scaling, 128 * scaling))
        factor = 2 if bilinear else 1
        self.down4 = (Down(128 * scaling, 256 * scaling // factor))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        return x5

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
    

class Accelerator(nn.Module):
    def __init__(self, num_iterations, n_channels, bilinear=False, scaling:int=1):
        super(Accelerator, self).__init__()

        self.decoder = Decoder(n_channels, bilinear, scaling)
        self.encoders = nn.ModuleList([Encoder(n_channels, bilinear, scaling) for _ in range(num_iterations)])
        self.history = []

    def forward(self, x):
        # Siempre añade la reconstrucción más reciente
        self.history.append(x.detach())

        # Si el historial es demasiado largo, quita el elemento más antiguo
        if len(self.history) > len(self.encoders): # len(self.encoders) es tu T
            self.history.pop(0)

        
        h = 0
        for i in range(len(self.history)):
            h_i = self.encoders[i](self.history[i])
            h = h + h_i  
        h = h / len(self.history)

        v = self.decoder(h)
        v = v + self.history[-1] 
        return v