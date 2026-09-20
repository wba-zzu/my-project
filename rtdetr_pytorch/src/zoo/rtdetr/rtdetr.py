"""by lyuwenyu

Small logging addition:
- records the real multi-scale training size in self._last_input_size
- this value is only used for terminal logging and does not affect the model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from src.core import register


__all__ = ['RTDETR', ]


@register
class RTDETR(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', ]

    def __init__(self, backbone: nn.Module, encoder, decoder, multi_scale=None):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        self.multi_scale = multi_scale

        # 仅用于日志显示，不参与网络计算。
        self._last_input_size = None

    def forward(self, x, targets=None):
        if self.multi_scale and self.training:
            sz = int(np.random.choice(self.multi_scale))
            self._last_input_size = sz
            x = F.interpolate(x, size=[sz, sz])
        else:
            # eval 或关闭 multi-scale 时记录真实输入尺寸。
            self._last_input_size = int(x.shape[-1])

        x = self.backbone(x)
        x = self.encoder(x)
        x = self.decoder(x, targets)

        return x

    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self
