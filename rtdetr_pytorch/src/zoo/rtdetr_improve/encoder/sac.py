"""Semantic Alignment Calibration (SAC) from UAV-DETR."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import ConvBNAct
from .msff_fe import FFM


class SemanticAlignmentCalibration(nn.Module):
    """Align high-resolution spatial features with low-resolution semantics."""

    def __init__(self, spatial_channels=256, semantic_channels=256, groups=2):
        super().__init__()
        if spatial_channels % groups != 0:
            raise ValueError("SAC spatial_channels must be divisible by groups")

        hidden_channels = spatial_channels
        self.groups = groups

        self.spatial_conv = ConvBNAct(spatial_channels, hidden_channels, 3)
        self.semantic_conv = ConvBNAct(semantic_channels, hidden_channels, 3)

        self.frequency_enhancer = FFM(hidden_channels)
        self.gating_conv = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            kernel_size=1,
            padding=0,
            bias=True,
        )

        self.offset_conv = nn.Sequential(
            ConvBNAct(hidden_channels * 2, 64),
            nn.Conv2d(
                64,
                self.groups * 4 + 2,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
        )

        self._init_weights_like_uavdetr()
        self.offset_conv[1].weight.data.zero_()

    def _init_weights_like_uavdetr(self):
        # The supplied UAV-DETR implementation only applies this loop to
        # immediate Conv2d children. We preserve that behavior here.
        for layer in self.children():
            if isinstance(layer, (nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0.0)

    def forward(self, coarse_features, semantic_features):
        batch_size, _, out_h, out_w = coarse_features.shape

        semantic_features = self.semantic_conv(semantic_features)
        semantic_features = F.interpolate(
            semantic_features,
            size=(out_h, out_w),
            mode="bilinear",
            align_corners=True,
        )

        enhanced_frequency = self.frequency_enhancer(semantic_features)
        gate = torch.sigmoid(self.gating_conv(semantic_features))
        fused_features = semantic_features * (1 - gate) + enhanced_frequency * gate

        coarse_features = self.spatial_conv(coarse_features)
        conv_results = self.offset_conv(torch.cat((coarse_features, fused_features), dim=1))

        fused_grouped = fused_features.reshape(
            batch_size * self.groups,
            -1,
            out_h,
            out_w,
        )
        coarse_grouped = coarse_features.reshape(
            batch_size * self.groups,
            -1,
            out_h,
            out_w,
        )

        offset_low = conv_results[:, : self.groups * 2].reshape(
            batch_size * self.groups,
            2,
            out_h,
            out_w,
        )
        offset_high = conv_results[:, self.groups * 2 : self.groups * 4].reshape(
            batch_size * self.groups,
            2,
            out_h,
            out_w,
        )

        grid_y = torch.linspace(-1.0, 1.0, out_h, device=coarse_features.device, dtype=coarse_features.dtype)
        grid_x = torch.linspace(-1.0, 1.0, out_w, device=coarse_features.device, dtype=coarse_features.dtype)
        yy, xx = torch.meshgrid(grid_y, grid_x, indexing="ij")
        base_grid = torch.stack((xx, yy), dim=-1)
        base_grid = base_grid.unsqueeze(0).repeat(batch_size * self.groups, 1, 1, 1)

        normalizer = torch.tensor(
            [out_w, out_h],
            dtype=coarse_features.dtype,
            device=coarse_features.device,
        ).view(1, 1, 1, 2)

        adjusted_grid_l = base_grid + offset_low.permute(0, 2, 3, 1) / normalizer
        adjusted_grid_h = base_grid + offset_high.permute(0, 2, 3, 1) / normalizer

        coarse_grouped = F.grid_sample(
            coarse_grouped,
            adjusted_grid_l,
            align_corners=True,
        )
        fused_grouped = F.grid_sample(
            fused_grouped,
            adjusted_grid_h,
            align_corners=True,
        )

        coarse_features = coarse_grouped.reshape(batch_size, -1, out_h, out_w)
        fused_features = fused_grouped.reshape(batch_size, -1, out_h, out_w)

        attention_weights = 1 + torch.tanh(conv_results[:, self.groups * 4 :, :, :])
        final_features = (
            fused_features * attention_weights[:, 0:1, :, :]
            + coarse_features * attention_weights[:, 1:2, :, :]
        )
        return final_features
