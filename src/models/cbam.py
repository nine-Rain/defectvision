"""CBAM：卷积块注意力模块（Convolutional Block Attention Module, ECCV 2018）。

结构：先通道注意力（"看什么"），再空间注意力（"看哪里"），两者串行、都做残差式相乘。

    x -> ChannelAttention(x) * x -> SpatialAttention(...) * ... -> out

放在 FPN 每一层输出之后，作用是多尺度特征里把缺陷区域"提亮"、把背景纹理压下去。
"""

from __future__ import annotations

import torch
from torch import nn


class ChannelAttention(nn.Module):
    """通道注意力：同时用平均池化和最大池化，各自过一个小 MLP，相加后 sigmoid。"""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # 两个池化共享同一组 MLP 权重，参数量只有 2*C*C/r
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.sigmoid(self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x)))
        return x * weights


class SpatialAttention(nn.Module):
    """空间注意力：沿通道维取平均和最大，拼成 2 通道，用 7x7 卷积压成 1 通道。"""

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map = torch.max(x, dim=1, keepdim=True).values
        weights = self.sigmoid(self.conv(torch.cat([avg_map, max_map], dim=1)))
        return x * weights


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7) -> None:
        super().__init__()
        self.channel = ChannelAttention(channels, reduction)
        self.spatial = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spatial(self.channel(x))
