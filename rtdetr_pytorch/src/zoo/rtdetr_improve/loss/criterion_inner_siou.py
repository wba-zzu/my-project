"""RT-DETR SetCriterion with UAV-DETR Inner-SIoU box regression."""

import torch
import torch.nn.functional as F

from src.core import register
from src.zoo.rtdetr.rtdetr_criterion import SetCriterion

from .inner_siou import inner_siou


@register
class SetCriterionInnerSIoU(SetCriterion):
    """Keep RT-DETR matching/classification logic and replace GIoU loss with Inner-SIoU."""

    __share__ = ["num_classes"]
    __inject__ = ["matcher"]

    def __init__(
        self,
        matcher,
        weight_dict,
        losses,
        alpha=0.2,
        gamma=2.0,
        eos_coef=1e-4,
        num_classes=80,
        inner_iou_ratio=1.25,
    ):
        super().__init__(
            matcher=matcher,
            weight_dict=weight_dict,
            losses=losses,
            alpha=alpha,
            gamma=gamma,
            eos_coef=eos_coef,
            num_classes=num_classes,
        )
        self.inner_iou_ratio = inner_iou_ratio

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """L1 + Inner-SIoU, matching the supplied UAV-DETR regression change."""

        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat(
            [t["boxes"][i] for t, (_, i) in zip(targets, indices)],
            dim=0,
        )

        losses = {}

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        siou = inner_siou(
            src_boxes,
            target_boxes,
            ratio=self.inner_iou_ratio,
        )
        # losses["loss_inner_siou"] = (1.0 - siou).sum() / num_boxes
        losses["loss_giou"] = (1.0 - siou).sum() / num_boxes

        return losses
