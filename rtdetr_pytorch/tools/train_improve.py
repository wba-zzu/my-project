"""RT-DETR training entry for custom improvement modules.

The upstream ``tools/train.py`` is intentionally left unchanged.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import argparse

import src.misc.dist as dist
# Importing this package registers UAVHybridEncoder and SetCriterionInnerSIoU.
import src.zoo.rtdetr_improve  # noqa: F401,E402
from src.core import YAMLConfig
from src.solver import TASKS


def main(args) -> None:
    dist.init_distributed()

    if args.seed is not None:
        dist.set_seed(args.seed)

    assert not all([args.tuning, args.resume]), \
        "Only support from_scratch or resume or tuning at one time"

    cfg = YAMLConfig(
        args.config,
        resume=args.resume,
        tuning=args.tuning,
        use_amp=args.amp,
        epoches=args.epochs,
        output_dir=args.output_dir,
        train_dataloader={
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
        },
        val_dataloader={
            "batch_size": args.val_batch_size,
            "num_workers": args.num_workers,
        },
    )

    solver = TASKS[cfg.yaml_cfg["task"]](cfg)

    if args.test_only:
        solver.val()
    else:
        solver.fit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=(
            "/home/ubuntu/WBA/RT-DETR-WBA/rtdetr_pytorch/"
            "configs/rtdetr_improve/uavdetr_r18_visdrone.yml"
        ),
    )

    parser.add_argument("--resume", "-r", type=str, default=None)
    parser.add_argument("--tuning", "-t", type=str, default=None)

    parser.add_argument(
        "--test-only",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--val-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)

    parser.add_argument(
        "--output-dir",
        type=str,
        default=(
            "/home/ubuntu/WBA/RT-DETR-WBA/rtdetr_pytorch/"
            "output/UAV"
        ),
    )

    args = parser.parse_args()
    main(args)
