import torch
import torch.nn as nn

class FReLU(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv_frelu = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels)
        self.bn_frelu = nn.BatchNorm2d(in_channels)
        self.ln_frelu = nn.LayerNorm(in_channels)
        
    def forward(self, x, layer_norm=False):
        y = self.conv_frelu(x)
        if layer_norm:
            y = y.permute(0,2,3,1).contiguous()
            y = self.ln_frelu(y)
            y = y.permute(0,3,1,2).contiguous()
        else:
            y = self.bn_frelu(y)
        x = torch.max(x, y)
        return x