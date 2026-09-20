"""Frequency-Focused DownSampling (FD) from UAV-DETR."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import ConvBNAct
from .msff_fe import FFM


class FrequencyFocusedDownSampling(nn.Module):
    """UAV-DETR frequency-focused 2x downsampling block."""

    def __init__(self, c1, c2):
        super().__init__()
        if c1 % 2 != 0 or c2 % 2 != 0:
            raise ValueError("FD expects even input and output channel counts")

        self.c = c2 // 2
        self.cv1 = ConvBNAct(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = ConvBNAct(c1 // 2, self.c, 1, 1, 0)
        self.ffm = FFM(self.c)
        self.conv_reduce = ConvBNAct(self.c * 2, self.c, 1, 1)
        self.conv_resize = ConvBNAct(self.c, self.c, 3, 2, 1)

    def forward(self, x):
        x = F.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, dim=1)

        x1 = self.cv1(x1)

        fgm_out = self.ffm(x2)
        fgm_out = self.conv_resize(fgm_out)

        pooled_out = F.max_pool2d(x2, 3, 2, 1)
        pooled_out = self.cv2(pooled_out)

        x2 = torch.cat((fgm_out, pooled_out), dim=1)
        x2 = self.conv_reduce(x2)

        return torch.cat((x1, x2), dim=1)
