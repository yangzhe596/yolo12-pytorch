import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # 防止 in_planes < ratio 时出现 0 通道
        reduced_planes = max(in_planes // ratio, 1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, reduced_planes, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_planes, in_planes, kernel_size=1, bias=False)
        )

    def forward(self, x):
        # x: [B, C, H, W]
        max_out = self.fc(self.max_pool(x))
        avg_out = self.fc(self.avg_pool(x))
        out = max_out + avg_out
        return torch.sigmoid(out)   # [B, C, 1, 1]


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), "kernel_size must be 3 or 7"
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size,
                              padding=padding, bias=False)

    def forward(self, x):
        # x: [B, C, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)     # [B, 1, H, W]
        avg_out = torch.mean(x, dim=1, keepdim=True)       # [B, 1, H, W]
        x_cat = torch.cat([max_out, avg_out], dim=1)       # [B, 2, H, W]
        x_out = self.conv(x_cat)                           # [B, 1, H, W]
        return torch.sigmoid(x_out)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super().__init__()
        self.channel_att = ChannelAttention(in_planes, ratio)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x):
        # 通道注意力
        ca_weight = self.channel_att(x)
        x = x * ca_weight

        # 空间注意力
        sa_weight = self.spatial_att(x)
        x = x * sa_weight

        return x