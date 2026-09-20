"""MSFF-FE components ported from the supplied UAV-DETR implementation.

The paper-level MSFF-FE path is implemented here as:
    P2 -> Focus -----------\
    P3 -------------------- concat -> MFFF -> CSPRepLayer
    Y4 -> DySample --------/

MFFF itself contains the frequency-domain FFM/ImprovedFFTKernel path used by
UAV-DETR.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.zoo.rtdetr.hybrid_encoder import CSPRepLayer

from .common import ConvBNAct, Focus


class DySample(nn.Module):
    """Dynamic upsampling module used by UAV-DETR."""

    def __init__(self, in_channels, scale=2, style="lp", groups=4, dyscope=False):
        super().__init__()
        self.scale = scale
        self.style = style
        self.groups = groups

        if style not in ("lp", "pl"):
            raise ValueError(f"Unsupported DySample style: {style}")
        if style == "pl":
            if in_channels < scale ** 2 or in_channels % (scale ** 2) != 0:
                raise ValueError("For style='pl', channels must be divisible by scale^2")
        if in_channels < groups or in_channels % groups != 0:
            raise ValueError("DySample in_channels must be divisible by groups")

        offset_in_channels = in_channels
        if style == "pl":
            offset_in_channels = in_channels // (scale ** 2)
            offset_out_channels = 2 * groups
        else:
            offset_out_channels = 2 * groups * (scale ** 2)

        self.offset = nn.Conv2d(offset_in_channels, offset_out_channels, 1)
        nn.init.normal_(self.offset.weight, mean=0.0, std=0.001)
        if self.offset.bias is not None:
            nn.init.constant_(self.offset.bias, 0.0)

        if dyscope:
            self.scope = nn.Conv2d(offset_in_channels, offset_out_channels, 1)
            nn.init.constant_(self.scope.weight, 0.0)
            if self.scope.bias is not None:
                nn.init.constant_(self.scope.bias, 0.0)

        self.register_buffer("init_pos", self._init_pos())

    def _init_pos(self):
        h = torch.arange(
            (-self.scale + 1) / 2,
            (self.scale - 1) / 2 + 1,
            dtype=torch.float32,
        ) / self.scale
        grid0, grid1 = torch.meshgrid(h, h, indexing="ij")
        # Keep the same ordering as the supplied UAV-DETR implementation:
        # torch.stack(torch.meshgrid([h, h])).transpose(1, 2)
        pos = torch.stack((grid0, grid1), dim=0)
        pos = pos.transpose(1, 2).repeat(1, self.groups, 1)
        return pos.reshape(1, -1, 1, 1)

    def sample(self, x, offset):
        b, _, h, w = offset.shape
        offset = offset.view(b, 2, -1, h, w)

        coords_y = torch.arange(h, dtype=x.dtype, device=x.device) + 0.5
        coords_x = torch.arange(w, dtype=x.dtype, device=x.device) + 0.5
        grid_y, grid_x = torch.meshgrid(coords_y, coords_x, indexing="ij")
        coords = torch.stack((grid_x, grid_y), dim=0).unsqueeze(0).unsqueeze(2)

        normalizer = torch.tensor([w, h], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1

        coords = F.pixel_shuffle(coords.view(b, -1, h, w), self.scale)
        coords = coords.view(b, 2, -1, self.scale * h, self.scale * w)
        coords = coords.permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)

        sampled = F.grid_sample(
            x.reshape(b * self.groups, -1, h, w),
            coords,
            mode="bilinear",
            align_corners=False,
            padding_mode="border",
        )
        return sampled.view(b, -1, self.scale * h, self.scale * w)

    def forward_lp(self, x):
        if hasattr(self, "scope"):
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward_pl(self, x):
        x_up = F.pixel_shuffle(x, self.scale)
        if hasattr(self, "scope"):
            offset = F.pixel_unshuffle(
                self.offset(x_up) * self.scope(x_up).sigmoid(),
                self.scale,
            ) * 0.5 + self.init_pos
        else:
            offset = F.pixel_unshuffle(self.offset(x_up), self.scale) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward(self, x):
        return self.forward_pl(x) if self.style == "pl" else self.forward_lp(x)


class FFM(nn.Module):
    """Frequency feature modulation block from UAV-DETR."""

    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim * 2, 3, 1, 1, groups=dim)
        self.dwconv1 = nn.Conv2d(dim, dim, 1, 1, groups=1)
        self.dwconv2 = nn.Conv2d(dim, dim, 1, 1, groups=1)
        self.alpha = nn.Parameter(torch.zeros(dim, 1, 1))
        self.beta = nn.Parameter(torch.ones(dim, 1, 1))

    def forward(self, x):
        x1 = self.dwconv1(x)
        x2 = self.dwconv2(x)
        x2_fft = torch.fft.fft2(x2, norm="backward")
        out = x1 * x2_fft
        out = torch.fft.ifft2(out, dim=(-2, -1), norm="backward")
        out = torch.abs(out)
        return out * self.alpha + x * self.beta


class ImprovedFFTKernel(nn.Module):
    """Frequency-enhanced kernel used inside MFFF."""

    def __init__(self, dim):
        super().__init__()
        ker = 31
        pad = ker // 2

        self.in_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1),
            nn.GELU(),
        )
        self.out_conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1)
        self.dw_33 = nn.Conv2d(dim, dim, kernel_size=ker, padding=pad, stride=1, groups=dim)
        self.dw_11 = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=dim)
        self.act = nn.SiLU()

        self.conv1x1 = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, bias=True)
        self.conv3x3 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=dim, bias=True)
        self.conv5x5 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, stride=1, groups=dim, bias=True)

        self.fac_conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, bias=True)
        self.fac_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.ffm = FFM(dim)

        hidden = max(dim // 4, 1)
        self.channel_attention = nn.Sequential(
            nn.Conv2d(dim, hidden, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden, dim, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out = self.in_conv(x)

        x_att = self.fac_conv(self.fac_pool(out))
        x_fft = torch.fft.fft2(out, norm="backward")
        x_fft = x_att * x_fft
        x_fca = torch.fft.ifft2(x_fft, dim=(-2, -1), norm="backward")
        x_fca = torch.abs(x_fca)

        x_sca = self.conv1x1(x_fca) + self.conv3x3(x_fca) + self.conv5x5(x_fca)
        channel_weights = self.channel_attention(x_att)
        x_sca = x_sca * channel_weights
        x_sca = self.ffm(x_sca)

        out = x + self.dw_33(out) + self.dw_11(out) + x_sca
        out = self.act(out)
        return self.out_conv(out)


class MFFF(nn.Module):
    """Multi-scale feature fusion with frequency enhancement from UAV-DETR."""

    def __init__(self, dim, e=0.25):
        super().__init__()
        self.e = e
        self.cv1 = ConvBNAct(dim, dim, 1)
        self.cv2 = ConvBNAct(dim, dim, 1)
        freq_channels = int(round(dim * e))
        if freq_channels <= 0 or freq_channels >= dim:
            raise ValueError("MFFF expansion e must produce 0 < frequency channels < dim")
        self.freq_channels = freq_channels
        self.m = ImprovedFFTKernel(freq_channels)

    def forward(self, x):
        x = self.cv1(x)
        freq, identity = torch.split(
            x,
            [self.freq_channels, x.shape[1] - self.freq_channels],
            dim=1,
        )
        return self.cv2(torch.cat((self.m(freq), identity), dim=1))


class MSFFFE(nn.Module):
    """Complete MSFF-FE branch used by UAV-DETR R18."""

    def __init__(
        self,
        p2_channels=64,
        p3_channels=128,
        y4_channels=128,
        hidden_dim=256,
        focus_channels=128,
        mfff_expansion=0.25,
        csp_expansion=0.5,
        depth_mult=1.0,
        act="silu",
        dysample_groups=4,
    ):
        super().__init__()
        if p3_channels != focus_channels or y4_channels != focus_channels:
            raise ValueError(
                "UAV-DETR MSFF-FE expects Focus(P2), P3 and DySample(Y4) "
                "to have the same channel width"
            )

        self.focus = Focus(p2_channels, focus_channels, k=1, s=1)
        self.upsample = DySample(
            y4_channels,
            scale=2,
            style="lp",
            groups=dysample_groups,
        )

        concat_channels = focus_channels + p3_channels + y4_channels
        self.mfff = MFFF(concat_channels, e=mfff_expansion)
        self.fpn_block = CSPRepLayer(
            concat_channels,
            hidden_dim,
            round(3 * depth_mult),
            expansion=csp_expansion,
            act=act,
        )

    def forward(self, p2, p3, y4):
        p2_focus = self.focus(p2)
        y4_up = self.upsample(y4)

        if p2_focus.shape[-2:] != p3.shape[-2:] or y4_up.shape[-2:] != p3.shape[-2:]:
            raise RuntimeError(
                "MSFF-FE spatial mismatch: "
                f"Focus(P2)={tuple(p2_focus.shape)}, P3={tuple(p3.shape)}, "
                f"DySample(Y4)={tuple(y4_up.shape)}"
            )

        fused = torch.cat((p2_focus, y4_up, p3), dim=1)
        fused = self.mfff(fused)
        return self.fpn_block(fused)
