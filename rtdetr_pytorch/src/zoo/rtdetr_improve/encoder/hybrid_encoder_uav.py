"""UAV-DETR Efficient Hybrid Encoder port for the original RT-DETR PyTorch codebase.

This class keeps the original RT-DETR model/decoder untouched and replaces only
its HybridEncoder with the feature path from the supplied UAV-DETR-R18 config:

P5 -> projection -> AIFI -> Y5
                    | DySample
P4 -> projection -> concat -> CSPRep -> reduce -> Y4
                                      | DySample
P2 -> Focus --------------------------|
P3 -----------------------------------|-> MFFF/MSFF-FE -> P3_out
P3_out -> FD -> concat(Y4) -> CSPRep -> P4_out
P4_out -> FD -> concat(Y5) -> CSPRep -> P5_out
SAC(P3_out, P5_out) -> calibrated P3
Return [P3_calibrated, P4_out, P5_out] for the original RTDETRTransformer.
"""

import copy

import torch
import torch.nn as nn

from src.core import register
from src.zoo.rtdetr.hybrid_encoder import (
    ConvNormLayer,
    CSPRepLayer,
    HybridEncoder,
    TransformerEncoder,
    TransformerEncoderLayer,
)

from .fd import FrequencyFocusedDownSampling
from .msff_fe import DySample, MSFFFE
from .sac import SemanticAlignmentCalibration


@register
class UAVHybridEncoder(nn.Module):
    """Hybrid encoder reproducing the UAV-DETR-R18 feature-neck improvements."""

    def __init__(
        self,
        in_channels=[64, 128, 256, 512],
        feat_strides=[4, 8, 16, 32],
        hidden_dim=256,
        nhead=8,
        dim_feedforward=1024,
        dropout=0.0,
        enc_act="gelu",
        use_encoder_idx=[3],
        num_encoder_layers=1,
        pe_temperature=10000,
        expansion=0.5,
        depth_mult=1.0,
        act="silu",
        eval_spatial_size=None,
        p2_focus_channels=128,
        y4_channels=128,
        mfff_expansion=0.25,
        dysample_groups=4,
        sac_groups=2,
    ):
        super().__init__()

        if len(in_channels) != 4 or len(feat_strides) != 4:
            raise ValueError(
                "UAVHybridEncoder expects four backbone levels [P2, P3, P4, P5]."
            )
        if use_encoder_idx != [3]:
            raise ValueError(
                "The supplied UAV-DETR-R18 applies AIFI only to P5; "
                "set use_encoder_idx: [3]."
            )

        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_spatial_size = eval_spatial_size

        # Output contract stays identical to the original RT-DETR decoder.
        self.out_channels = [hidden_dim, hidden_dim, hidden_dim]
        self.out_strides = [8, 16, 32]

        p2_c, p3_c, p4_c, p5_c = in_channels

        # UAV-DETR layer 8: P5 input projection, no activation.
        self.p5_proj = ConvNormLayer(p5_c, hidden_dim, 1, 1, act=None)

        encoder_layer = TransformerEncoderLayer(
            hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=enc_act,
        )
        self.encoder = nn.ModuleList(
            [
                TransformerEncoder(copy.deepcopy(encoder_layer), num_encoder_layers)
                for _ in range(len(use_encoder_idx))
            ]
        )

        # UAV-DETR layer 10: Y5 lateral conv.
        self.y5_lateral = ConvNormLayer(hidden_dim, hidden_dim, 1, 1, act=act)
        self.y5_upsample = DySample(
            hidden_dim,
            scale=2,
            style="lp",
            groups=dysample_groups,
        )

        # UAV-DETR layer 12: P4 projection, no activation.
        self.p4_proj = ConvNormLayer(p4_c, hidden_dim, 1, 1, act=None)
        self.p4_fpn = CSPRepLayer(
            hidden_dim * 2,
            hidden_dim,
            round(3 * depth_mult),
            expansion=expansion,
            act=act,
        )

        # UAV-DETR layer 15: reduce Y4 to 128 channels before MSFF-FE.
        self.y4_reduce = ConvNormLayer(hidden_dim, y4_channels, 1, 1, act=act)

        # Layers 16-20: DySample + Focus + P3 + MFFF + RepC3.
        self.msff_fe = MSFFFE(
            p2_channels=p2_c,
            p3_channels=p3_c,
            y4_channels=y4_channels,
            hidden_dim=hidden_dim,
            focus_channels=p2_focus_channels,
            mfff_expansion=mfff_expansion,
            csp_expansion=expansion,
            depth_mult=depth_mult,
            act=act,
            dysample_groups=dysample_groups,
        )

        # Layers 21-26: two FD stages with the UAV-DETR skip widths.
        self.fd1 = FrequencyFocusedDownSampling(hidden_dim, hidden_dim)
        self.pan1 = CSPRepLayer(
            hidden_dim + y4_channels,
            hidden_dim,
            round(3 * depth_mult),
            expansion=expansion,
            act=act,
        )

        self.fd2 = FrequencyFocusedDownSampling(hidden_dim, hidden_dim)
        self.pan2 = CSPRepLayer(
            hidden_dim * 2,
            hidden_dim,
            round(3 * depth_mult),
            expansion=expansion,
            act=act,
        )

        # Layer 27: SAC(P3_out, P5_out).
        self.sac = SemanticAlignmentCalibration(
            spatial_channels=hidden_dim,
            semantic_channels=hidden_dim,
            groups=sac_groups,
        )

        self._reset_parameters()

    def _reset_parameters(self):
        if self.eval_spatial_size:
            stride = self.feat_strides[3]
            pos_embed = HybridEncoder.build_2d_sincos_position_embedding(
                self.eval_spatial_size[1] // stride,
                self.eval_spatial_size[0] // stride,
                self.hidden_dim,
                self.pe_temperature,
            )
            self.register_buffer("pos_embed3", pos_embed, persistent=False)

    def _encode_p5(self, p5):
        p5 = self.p5_proj(p5)
        if self.num_encoder_layers <= 0:
            return p5

        h, w = p5.shape[2:]
        src_flatten = p5.flatten(2).permute(0, 2, 1)

        if self.training or self.eval_spatial_size is None:
            pos_embed = HybridEncoder.build_2d_sincos_position_embedding(
                w,
                h,
                self.hidden_dim,
                self.pe_temperature,
            ).to(src_flatten.device)
        else:
            pos_embed = getattr(self, "pos_embed3", None)

        memory = self.encoder[0](src_flatten, pos_embed=pos_embed)
        return memory.permute(0, 2, 1).reshape(
            -1,
            self.hidden_dim,
            h,
            w,
        ).contiguous()

    def forward(self, feats):
        if len(feats) != 4:
            raise ValueError(
                f"UAVHybridEncoder needs [P2, P3, P4, P5], got {len(feats)} tensors."
            )

        p2, p3, p4, p5 = feats

        # P5 -> AIFI -> Y5.
        p5 = self._encode_p5(p5)
        y5 = self.y5_lateral(p5)

        # Y5 -> DySample, fuse with projected P4, then reduce Y4 to 128.
        y5_up = self.y5_upsample(y5)
        p4_proj = self.p4_proj(p4)
        if y5_up.shape[-2:] != p4_proj.shape[-2:]:
            raise RuntimeError(
                f"P4 top-down mismatch: Y5_up={tuple(y5_up.shape)}, P4={tuple(p4_proj.shape)}"
            )
        p4_td = self.p4_fpn(torch.cat((y5_up, p4_proj), dim=1))
        y4 = self.y4_reduce(p4_td)

        # MSFF-FE: Focus(P2) + P3 + DySample(Y4) -> MFFF -> P3_out.
        p3_out = self.msff_fe(p2, p3, y4)

        # FD + Y4 -> P4_out.
        p4_down = self.fd1(p3_out)
        if p4_down.shape[-2:] != y4.shape[-2:]:
            raise RuntimeError(
                f"First FD mismatch: FD(P3)={tuple(p4_down.shape)}, Y4={tuple(y4.shape)}"
            )
        p4_out = self.pan1(torch.cat((p4_down, y4), dim=1))

        # FD + Y5 -> P5_out.
        p5_down = self.fd2(p4_out)
        if p5_down.shape[-2:] != y5.shape[-2:]:
            raise RuntimeError(
                f"Second FD mismatch: FD(P4)={tuple(p5_down.shape)}, Y5={tuple(y5.shape)}"
            )
        p5_out = self.pan2(torch.cat((p5_down, y5), dim=1))

        # SAC calibrates the high-resolution P3 feature with P5 semantics.
        p3_calibrated = self.sac(p3_out, p5_out)

        return [p3_calibrated, p4_out, p5_out]
