"""by lyuwenyu

Extended training entry:
- keeps model/training arguments in train.py
- keeps experiment-directory naming logic in src/misc/experiment.py
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import argparse

import src.misc.dist as dist
# Importing this package registers UAVHybridEncoder and SetCriterionInnerSIoU.
import src.zoo.rtdetr_improve  # noqa: F401,E402
from src.core import YAMLConfig
from src.solver import TASKS
from src.misc.experiment import resolve_experiment_dir


def main(args) -> None:

    # 初始化分布式训练
    dist.init_distributed()

    # 随机种子
    if args.seed is not None:
        dist.set_seed(args.seed)

    # resume 和 tuning 不能同时使用
    assert not all([args.tuning, args.resume]), \
        'Only support from_scratch or resume or tuning at one time'

    # ==========================================================
    # 解析本次实验实际保存目录
    #
    # 新训练：
    #   output/train + exp  -> output/train/exp
    #   如果 exp 已存在      -> output/train/exp2
    #   如果 exp2 也存在     -> output/train/exp3
    #
    # Resume：
    #   /xxx/exp/weights/epoch005.pth
    #   -> 强制继续保存到 /xxx/exp
    # ==========================================================
    run_dir = resolve_experiment_dir(
        base_dir=args.output_dir,
        name=args.name,
        resume=args.resume,
    )

    # ==========================================================
    # 构造命令行覆盖参数
    # ==========================================================
    overrides = {
        'resume': args.resume,
        'tuning': args.tuning,
        'use_amp': args.amp,
        'output_dir': str(run_dir),
    }

    # --------------------------
    # 训练轮数
    # 注意：RT-DETR源码中写的是 epoches
    # --------------------------
    if args.epochs is not None:
        overrides['epoches'] = args.epochs

    # --------------------------
    # checkpoint 保存间隔
    # --------------------------
    if args.checkpoint_step is not None:
        overrides['checkpoint_step'] = args.checkpoint_step

    # --------------------------
    # 日志输出间隔
    # --------------------------
    if args.log_step is not None:
        overrides['log_step'] = args.log_step

    # --------------------------
    # Train DataLoader
    # --------------------------
    train_loader_cfg = {}

    if args.batch_size is not None:
        train_loader_cfg['batch_size'] = args.batch_size

    if args.workers is not None:
        train_loader_cfg['num_workers'] = args.workers

    if train_loader_cfg:
        overrides['train_dataloader'] = train_loader_cfg

    # --------------------------
    # Val DataLoader
    # --------------------------
    val_loader_cfg = {}

    if args.val_batch_size is not None:
        val_loader_cfg['batch_size'] = args.val_batch_size

    if args.workers is not None:
        val_loader_cfg['num_workers'] = args.workers

    if val_loader_cfg:
        overrides['val_dataloader'] = val_loader_cfg

    # ==========================================================
    # 加载 YAML
    # ==========================================================
    cfg = YAMLConfig(
        args.config,
        **overrides
    )

    # ==========================================================
    # 打印当前关键配置
    # ==========================================================
    print("\n================ Training Config ================")
    print(f"Config          : {args.config}")
    print(f"Mode            : {'resume' if args.resume else ('tuning' if args.tuning else 'train')}")
    print(f"Name            : {run_dir.name}")
    print(f"Output dir      : {cfg.output_dir}")
    print(f"Epochs          : {cfg.epoches}")
    print(f"Train batch     : {cfg.yaml_cfg['train_dataloader']['batch_size']}")
    print(f"Val batch       : {cfg.yaml_cfg['val_dataloader']['batch_size']}")
    print(f"Train workers   : {cfg.yaml_cfg['train_dataloader']['num_workers']}")
    print(f"Val workers     : {cfg.yaml_cfg['val_dataloader']['num_workers']}")
    print(f"AMP             : {cfg.use_amp}")
    print(f"Seed            : {args.seed}")
    print(f"Checkpoint step : {cfg.checkpoint_step}")
    if args.resume:
        print(f"Resume          : {args.resume}")
    if args.tuning:
        print(f"Tuning          : {args.tuning}")
    print("=================================================\n")

    # 根据 task 创建 Solver
    solver = TASKS[cfg.yaml_cfg['task']](cfg)

    if args.test_only:
        solver.val()
    else:
        solver.fit()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # ==========================================================
    # 网络配置
    # ==========================================================
    parser.add_argument('--config', '-c',type=str,
        default='/home/ubuntu/WBA/RT-DETR-WBA/rtdetr_pytorch/configs/rtdetr_improve/uavdetr_r18_visdrone.yml',
        help='Model YAML config'
    )

    # ==========================================================
    # 训练控制
    # ==========================================================
    parser.add_argument('--epochs',type=int,
        default=200,
        help='Number of training epochs'
    )

    parser.add_argument('--batch-size',type=int,
        default=4,
        help='Training batch size'
    )

    parser.add_argument('--val-batch-size',type=int,
        default=4,
        help='Validation batch size'
    )

    parser.add_argument('--workers',type=int,
        default=8,
        help='Number of dataloader workers'
    )

    # ==========================================================
    # 实验保存目录
    #
    # 实际保存目录 = output-dir / name
    # ==========================================================
    parser.add_argument('--output-dir',type=str,
        default='/home/ubuntu/WBA/RT-DETR-WBA/rtdetr_pytorch/output/train/UAV',
        help='Root directory for training experiments'
    )

    parser.add_argument('--name',type=str,
        default='UAV-DETR_R18',
        help='Experiment name, e.g. exp, baseline, cfpr'
    )

    parser.add_argument(
        '--checkpoint-step',type=int,
        default=5,
        help='Save an extra checkpoint every N epochs'
    )

    parser.add_argument(
        '--log-step',type=int,
        default=None,
        help='Print training log every N iterations'
    )

    # ==========================================================
    # checkpoint
    # ==========================================================
    parser.add_argument(
        '--resume', '-r',type=str,
        default=None,
        help='Resume training from checkpoint. The original experiment directory is reused.'
    )

    parser.add_argument(
        '--tuning', '-t',type=str,
        default=None,
        help='Fine-tune from pretrained checkpoint'
    )

    # ==========================================================
    # 验证
    # ==========================================================
    parser.add_argument(
        '--test-only',action='store_true',
        default=False,
        help='Only run validation'
    )

    # ==========================================================
    # AMP
    # ==========================================================
    parser.add_argument(
        '--amp',action='store_true',
        default=False,
        help='Use automatic mixed precision'
    )

    # ==========================================================
    # Seed
    # ==========================================================
    parser.add_argument(
        '--seed',type=int,
        default=42,
        help='Random seed'
    )

    args = parser.parse_args()

    main(args)
