"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
https://github.com/facebookresearch/detr/blob/main/engine.py

by lyuwenyu

Terminal-display modification:
- YOLO-like one-line tqdm progress during training
- compact validation progress
- preserves the original COCO evaluator and its complete AP/AR summary
"""

import math
import sys
from typing import Iterable

import torch
import torch.amp

try:
    from tqdm.auto import tqdm
except ImportError as exc:
    raise ImportError(
        "YOLO-style terminal progress requires tqdm. "
        "Install it with: pip install tqdm"
    ) from exc

from src.data import CocoEvaluator
from src.misc import MetricLogger, SmoothedValue, reduce_dict
import src.misc.dist as dist


def _model_core(model):
    """Return the underlying module for normal / DDP models."""
    return model.module if hasattr(model, 'module') else model


def _current_input_size(model, samples):
    """Get the real size used after RT-DETR multi-scale interpolation."""
    core = _model_core(model)
    size = getattr(core, '_last_input_size', None)
    if size is None:
        size = samples.shape[-1]
    return int(size)


def _gpu_mem_gb(device):
    if not torch.cuda.is_available():
        return 0.0
    # Similar to common detector logs: reserved CUDA memory, shown in GiB.
    return torch.cuda.memory_reserved(device=device) / (1024 ** 3)


def _meter_value(metric_logger, name):
    """Epoch-running average for one named loss; 0 if the loss does not exist."""
    if name not in metric_logger.meters:
        return 0.0
    return float(metric_logger.meters[name].global_avg)


def _split_learning_rates(optimizer):
    """
    Return (backbone_lr, encoder_lr) for the official RT-DETR optimizer setup.

    In the official config, backbone parameter groups use the smaller LR (1e-5)
    while encoder/decoder and the remaining parameters use the main LR (1e-4).
    MultiStepLR scales all groups by the same factor, so min/max remains valid
    after each milestone as well.
    """
    lrs = [float(group['lr']) for group in optimizer.param_groups]
    if not lrs:
        return 0.0, 0.0
    return min(lrs), max(lrs)


def _global_count(value, device):
    """Sum an integer counter across DDP workers."""
    tensor = torch.tensor([value], dtype=torch.long, device=device)
    if dist.is_dist_available_and_initialized():
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return int(tensor.item())


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, **kwargs):
    model.train()
    criterion.train()

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('backbone_lr', SmoothedValue(window_size=1, fmt='{value:.8f}'))
    metric_logger.add_meter('encoder_lr', SmoothedValue(window_size=1, fmt='{value:.8f}'))

    ema = kwargs.get('ema', None)
    scaler = kwargs.get('scaler', None)
    total_epochs = kwargs.get('total_epochs', None)

    # YOLO-like header. RT-DETR uses VFL, so classification is named vfl_loss.
    if dist.is_main_process():
        print(
            f"\n"
            f"{'Epoch':>11}"
            f"{'GPU_mem':>11}"
            f"{'giou_loss':>12}"
            f"{'vfl_loss':>12}"
            f"{'l1_loss':>12}"
            f"{'Instances':>12}"
            f"{'Size':>8}"
            f"{'Backbone_lr':>14}"
            f"{'Encoder_lr':>13}"
        )

    pbar = tqdm(
        data_loader,
        total=len(data_loader),
        dynamic_ncols=True,
        disable=not dist.is_main_process(),
        leave=True,
        bar_format='{desc}: {percentage:3.0f}%|{bar:20}| {n_fmt}/{total_fmt} '
                   '[{elapsed}<{remaining}, {rate_fmt}]',
    )

    for samples, targets in pbar:
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if scaler is not None:
            with torch.autocast(device_type=str(device), cache_enabled=True):
                outputs = model(samples, targets)

            with torch.autocast(device_type=str(device), enabled=False):
                loss_dict = criterion(outputs, targets)

            loss = sum(loss_dict.values())
            scaler.scale(loss).backward()

            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        else:
            outputs = model(samples, targets)
            loss_dict = criterion(outputs, targets)

            loss = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()

            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()

        # EMA
        if ema is not None:
            ema.update(model)

        # Reduce losses across DDP workers for logging.
        loss_dict_reduced = reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())

        if not math.isfinite(float(loss_value)):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced)

        # The official RT-DETR optimizer uses a smaller LR for backbone and
        # the main LR for encoder/decoder. Log them separately.
        backbone_lr, encoder_lr = _split_learning_rates(optimizer)
        metric_logger.update(backbone_lr=backbone_lr, encoder_lr=encoder_lr)

        # Number of GT instances in the current batch.
        instances = sum(len(t['labels']) for t in targets)

        # The real network input size after multi-scale interpolation.
        img_size = _current_input_size(model, samples)

        giou_loss = _meter_value(metric_logger, 'loss_giou')
        vfl_loss = _meter_value(metric_logger, 'loss_vfl')
        l1_loss = _meter_value(metric_logger, 'loss_bbox')
        gpu_mem = _gpu_mem_gb(device)

        if total_epochs is not None and int(total_epochs) > 0:
            epoch_text = f'{epoch + 1}/{int(total_epochs)}'
        else:
            epoch_text = str(epoch + 1)

        desc = (
            f"{epoch_text:>11}"
            f"{gpu_mem:>10.2f}G"
            f"{giou_loss:>12.4f}"
            f"{vfl_loss:>12.4f}"
            f"{l1_loss:>12.4f}"
            f"{instances:>12d}"
            f"{img_size:>8d}"
            f"{backbone_lr:>14.2e}"
            f"{encoder_lr:>13.2e}"
        )
        pbar.set_description(desc, refresh=False)

    pbar.close()

    # Gather epoch averages across DDP workers.
    metric_logger.synchronize_between_processes()

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model: torch.nn.Module, criterion: torch.nn.Module, postprocessors,
             data_loader, base_ds, device, output_dir):
    model.eval()
    criterion.eval()

    # Keep MetricLogger because it is part of the original engine structure,
    # although validation losses are intentionally not computed here.
    metric_logger = MetricLogger(delimiter="  ")

    iou_types = postprocessors.iou_types
    coco_evaluator = CocoEvaluator(base_ds, iou_types)

    panoptic_evaluator = None

    local_images = 0
    local_instances = 0

    if dist.is_main_process():
        print(
            f"\n"
            f"{'Class':>16}"
            f"{'Images':>10}"
            f"{'Instances':>12}"
        )

    pbar = tqdm(
        data_loader,
        total=len(data_loader),
        dynamic_ncols=True,
        disable=not dist.is_main_process(),
        leave=True,
        bar_format='{desc}: {percentage:3.0f}%|{bar:20}| {n_fmt}/{total_fmt} '
                   '[{elapsed}<{remaining}, {rate_fmt}]',
    )

    for samples, targets in pbar:
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        local_images += len(targets)
        local_instances += sum(len(t['labels']) for t in targets)

        outputs = model(samples)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors(outputs, orig_target_sizes)

        res = {
            target['image_id'].item(): output
            for target, output in zip(targets, results)
        }
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if dist.is_main_process():
            pbar.set_description(
                f"{'all':>16}{local_images:>10d}{local_instances:>12d}",
                refresh=False,
            )

    pbar.close()

    # Preserve original distributed synchronization.
    metric_logger.synchronize_between_processes()
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # Preserve the COMPLETE original COCO AP/AR output.
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    stats = {}
    if coco_evaluator is not None:
        if 'bbox' in iou_types:
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in iou_types:
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()

    # A concise YOLO-like summary row in addition to, not instead of, COCO summary.
    total_images = _global_count(local_images, device)
    total_instances = _global_count(local_instances, device)

    if dist.is_main_process() and 'coco_eval_bbox' in stats:
        coco_stats = stats['coco_eval_bbox']
        map_50_95 = coco_stats[0] if len(coco_stats) > 0 else 0.0
        map_50 = coco_stats[1] if len(coco_stats) > 1 else 0.0

        print(
            f"\n"
            f"{'Class':>16}"
            f"{'Images':>10}"
            f"{'Instances':>12}"
            f"{'mAP50':>12}"
            f"{'mAP50-95':>12}"
        )
        print(
            f"{'all':>16}"
            f"{total_images:>10d}"
            f"{total_instances:>12d}"
            f"{map_50:>12.4f}"
            f"{map_50_95:>12.4f}"
        )

    return stats, coco_evaluator
