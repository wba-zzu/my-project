"""Inner-SIoU used by the supplied UAV-DETR implementation.

The supplied code computes:
    1 - bbox_inner_iou(pred, gt, xywh=True, SIoU=True, ratio=1.25)

This file provides the aligned-box form needed by the original RT-DETR
SetCriterion, where prediction and target tensors both have shape [N, 4] in
normalized cx, cy, w, h format.
"""

import math

import torch


def _inner_iou_xywh(box1, box2, ratio=1.25, eps=1e-7):
    x1, y1, w1, h1 = box1.unbind(-1)
    x2, y2, w2, h2 = box2.unbind(-1)

    b1_x1 = x1 - (w1 * ratio) / 2
    b1_x2 = x1 + (w1 * ratio) / 2
    b1_y1 = y1 - (h1 * ratio) / 2
    b1_y2 = y1 + (h1 * ratio) / 2

    b2_x1 = x2 - (w2 * ratio) / 2
    b2_x2 = x2 + (w2 * ratio) / 2
    b2_y1 = y2 - (h2 * ratio) / 2
    b2_y2 = y2 + (h2 * ratio) / 2

    inter_w = (torch.minimum(b1_x2, b2_x2) - torch.maximum(b1_x1, b2_x1)).clamp(min=0)
    inter_h = (torch.minimum(b1_y2, b2_y2) - torch.maximum(b1_y1, b2_y1)).clamp(min=0)
    inter = inter_w * inter_h
    union = w1 * h1 * ratio * ratio + w2 * h2 * ratio * ratio - inter + eps
    return inter / union


def inner_siou(box1, box2, ratio=1.25, eps=1e-7):
    """Return aligned Inner-SIoU scores for normalized cxcywh boxes."""

    if box1.shape != box2.shape or box1.shape[-1] != 4:
        raise ValueError(
            f"inner_siou expects equal [N,4] tensors, got {tuple(box1.shape)} and {tuple(box2.shape)}"
        )

    if box1.numel() == 0:
        return box1.new_zeros((0,))

    x1, y1, w1, h1 = box1.unbind(-1)
    x2, y2, w2, h2 = box2.unbind(-1)

    half_w1 = w1 / 2
    half_h1 = h1 / 2
    half_w2 = w2 / 2
    half_h2 = h2 / 2

    b1_x1, b1_x2 = x1 - half_w1, x1 + half_w1
    b1_y1, b1_y2 = y1 - half_h1, y1 + half_h1
    b2_x1, b2_x2 = x2 - half_w2, x2 + half_w2
    b2_y1, b2_y2 = y2 - half_h2, y2 + half_h2

    inner_iou_value = _inner_iou_xywh(box1, box2, ratio=ratio, eps=eps)

    cw = torch.maximum(b1_x2, b2_x2) - torch.minimum(b1_x1, b2_x1)
    ch = torch.maximum(b1_y2, b2_y2) - torch.minimum(b1_y1, b2_y1)
    cw = cw.clamp_min(eps)
    ch = ch.clamp_min(eps)

    s_cw = (b2_x1 + b2_x2 - b1_x1 - b1_x2) * 0.5 + eps
    s_ch = (b2_y1 + b2_y2 - b1_y1 - b1_y2) * 0.5 + eps

    sigma = torch.sqrt(s_cw ** 2 + s_ch ** 2).clamp_min(eps)
    sin_alpha_1 = torch.abs(s_cw) / sigma
    sin_alpha_2 = torch.abs(s_ch) / sigma
    threshold = math.sqrt(2.0) / 2.0
    sin_alpha = torch.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)
    sin_alpha = sin_alpha.clamp(min=0.0, max=1.0)

    angle_cost = torch.cos(torch.arcsin(sin_alpha) * 2 - math.pi / 2)
    rho_x = (s_cw / cw) ** 2
    rho_y = (s_ch / ch) ** 2
    gamma = angle_cost - 2
    distance_cost = 2 - torch.exp(gamma * rho_x) - torch.exp(gamma * rho_y)

    omega_w = torch.abs(w1 - w2) / torch.maximum(w1, w2).clamp_min(eps)
    omega_h = torch.abs(h1 - h2) / torch.maximum(h1, h2).clamp_min(eps)
    shape_cost = (
        torch.pow(1 - torch.exp(-omega_w), 4)
        + torch.pow(1 - torch.exp(-omega_h), 4)
    )

    return inner_iou_value - 0.5 * (distance_cost + shape_cost) + eps
