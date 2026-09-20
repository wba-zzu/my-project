"""Small building blocks used by the UAV-DETR encoder port.

The convolution block mirrors the Ultralytics ``Conv`` used by the supplied
UAV-DETR implementation: Conv2d(bias=False) + BatchNorm2d + SiLU by default.
"""

import torch
import torch.nn as nn


def autopad(kernel_size, padding=None, dilation=1):
    if dilation > 1:
        if isinstance(kernel_size, int):
            kernel_size = dilation * (kernel_size - 1) + 1
        else:
            kernel_size = [dilation * (k - 1) + 1 for k in kernel_size]
    if padding is None:
        padding = kernel_size // 2 if isinstance(kernel_size, int) else [k // 2 for k in kernel_size]
    return padding


class ConvBNAct(nn.Module):
    """Conv2d + BatchNorm2d + activation, matching UAV-DETR's Conv block."""

    def __init__(
        self,
        c1,
        c2,
        k=1,
        s=1,
        p=None,
        g=1,
        d=1,
        act=True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            c1,
            c2,
            k,
            s,
            autopad(k, p, d),
            groups=g,
            dilation=d,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(c2)
        if act is True:
            self.act = nn.SiLU()
        elif isinstance(act, nn.Module):
            self.act = act
        else:
            self.act = nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Focus(nn.Module):
    """Space-to-depth Focus used by UAV-DETR on the P2 feature."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = ConvBNAct(c1 * 4, c2, k, s, p, g, act=act)

    def forward(self, x):
        return self.conv(
            torch.cat(
                (
                    x[..., ::2, ::2],
                    x[..., 1::2, ::2],
                    x[..., ::2, 1::2],
                    x[..., 1::2, 1::2],
                ),
                dim=1,
            )
        )
