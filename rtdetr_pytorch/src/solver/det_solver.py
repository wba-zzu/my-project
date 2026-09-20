'''by lyuwenyu

Modified experiment output:
  output_dir/results.csv
  output_dir/log.txt
  output_dir/weights/best.pth
  output_dir/weights/last.pth
  output_dir/weights/epochXXX.pth

Also passes total_epochs to det_engine.py for YOLO-like terminal display.
'''

import csv
import time
import json
import datetime

import torch

from src.misc import dist
from src.data import get_coco_api_from_dataset

from .solver import BaseSolver
from .det_engine import train_one_epoch, evaluate


class DetSolver(BaseSolver):

    @staticmethod
    def _safe_float(value, default=''):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _load_best_map_from_csv(self, results_file):
        """Recover previous best AP50:95 when resuming in the same output dir."""
        best_map = -1.0
        if not results_file.exists():
            return best_map

        try:
            with results_file.open('r', newline='', encoding='utf-8') as f:
                for row in csv.DictReader(f):
                    value = row.get('metrics/mAP50-95(B)', '')
                    if value not in ('', None):
                        best_map = max(best_map, float(value))
        except Exception as exc:
            print(f'Warning: failed to read existing results.csv: {exc}')

        return best_map

    def fit(self, ):
        print("Start training")
        self.train()

        args = self.cfg

        n_parameters = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print('number of params:', n_parameters)

        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)
        best_stat = {'epoch': -1}

        # YOLO-like experiment directory.
        weights_dir = self.output_dir / 'weights'
        if dist.is_main_process():
            weights_dir.mkdir(parents=True, exist_ok=True)

        results_file = self.output_dir / 'results.csv'
        csv_fields = [
            'epoch',
            'train/loss',
            'train/loss_vfl',
            'train/loss_bbox',
            'train/loss_giou',
            'metrics/mAP50-95(B)',
            'metrics/mAP50(B)',
            'metrics/mAP75(B)',
            'metrics/mAP_small(B)',
            'metrics/mAP_medium(B)',
            'metrics/mAP_large(B)',
            'metrics/AR1(B)',
            'metrics/AR10(B)',
            'metrics/AR100(B)',
            'metrics/AR_small(B)',
            'metrics/AR_medium(B)',
            'metrics/AR_large(B)',
            'lr/backbone',
            'lr/encoder',
        ]

        if dist.is_main_process() and not results_file.exists():
            with results_file.open('w', newline='', encoding='utf-8') as f:
                csv.DictWriter(f, fieldnames=csv_fields).writeheader()

        # best.pth is selected by COCO AP@[0.50:0.95].
        best_map = (
            self._load_best_map_from_csv(results_file)
            if dist.is_main_process() else -1.0
        )

        start_time = time.time()

        for epoch in range(self.last_epoch + 1, args.epoches):
            if dist.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)

            train_stats = train_one_epoch(
                self.model,
                self.criterion,
                self.train_dataloader,
                self.optimizer,
                self.device,
                epoch,
                args.clip_max_norm,
                print_freq=args.log_step,
                ema=self.ema,
                scaler=self.scaler,
                total_epochs=args.epoches,
            )

            self.lr_scheduler.step()

            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                base_ds,
                self.device,
                self.output_dir,
            )

            coco_stats = test_stats.get('coco_eval_bbox', [])
            if len(coco_stats) < 12:
                coco_stats = list(coco_stats) + [0.0] * (12 - len(coco_stats))

            # COCO stats order:
            # 0 AP50:95, 1 AP50, 2 AP75, 3/4/5 AP S/M/L,
            # 6 AR1, 7 AR10, 8 ARmax, 9/10/11 AR S/M/L.
            map_50_95 = self._safe_float(coco_stats[0], 0.0)
            map_50 = self._safe_float(coco_stats[1], 0.0)
            map_75 = self._safe_float(coco_stats[2], 0.0)

            # Preserve the original best-stat summary.
            for k in test_stats.keys():
                if not test_stats[k]:
                    continue
                current = test_stats[k][0]
                if k in best_stat:
                    if current > best_stat[k]:
                        best_stat[k] = current
                        best_stat['epoch'] = epoch + 1
                else:
                    best_stat[k] = current
                    best_stat['epoch'] = epoch + 1
            print('best_stat: ', best_stat)

            # --------------------------------------------------
            # Weights
            # --------------------------------------------------
            if self.output_dir:
                state = self.state_dict(epoch)

                # Latest resumable checkpoint: overwritten every epoch.
                dist.save_on_master(state, weights_dir / 'last.pth')

                # Best checkpoint: AP50:95.
                if map_50_95 > best_map:
                    best_map = map_50_95
                    dist.save_on_master(state, weights_dir / 'best.pth')
                    if dist.is_main_process():
                        print(
                            f'New best: epoch {epoch + 1}, '
                            f'mAP50-95={map_50_95:.4f}, mAP50={map_50:.4f}'
                        )

                # Periodic snapshots.
                if args.checkpoint_step > 0 and (epoch + 1) % args.checkpoint_step == 0:
                    dist.save_on_master(
                        state,
                        weights_dir / f'epoch{epoch + 1:03d}.pth',
                    )

            # --------------------------------------------------
            # results.csv: one row per epoch
            # --------------------------------------------------
            csv_row = {
                'epoch': epoch + 1,
                'train/loss': self._safe_float(train_stats.get('loss', '')),
                'train/loss_vfl': self._safe_float(train_stats.get('loss_vfl', '')),
                'train/loss_bbox': self._safe_float(train_stats.get('loss_bbox', '')),
                'train/loss_giou': self._safe_float(train_stats.get('loss_giou', '')),
                'metrics/mAP50-95(B)': map_50_95,
                'metrics/mAP50(B)': map_50,
                'metrics/mAP75(B)': map_75,
                'metrics/mAP_small(B)': self._safe_float(coco_stats[3], 0.0),
                'metrics/mAP_medium(B)': self._safe_float(coco_stats[4], 0.0),
                'metrics/mAP_large(B)': self._safe_float(coco_stats[5], 0.0),
                'metrics/AR1(B)': self._safe_float(coco_stats[6], 0.0),
                'metrics/AR10(B)': self._safe_float(coco_stats[7], 0.0),
                'metrics/AR100(B)': self._safe_float(coco_stats[8], 0.0),
                'metrics/AR_small(B)': self._safe_float(coco_stats[9], 0.0),
                'metrics/AR_medium(B)': self._safe_float(coco_stats[10], 0.0),
                'metrics/AR_large(B)': self._safe_float(coco_stats[11], 0.0),
                'lr/backbone': self._safe_float(train_stats.get('backbone_lr', '')),
                'lr/encoder': self._safe_float(train_stats.get('encoder_lr', '')),
            }

            if dist.is_main_process():
                with results_file.open('a', newline='', encoding='utf-8') as f:
                    csv.DictWriter(f, fieldnames=csv_fields).writerow(csv_row)

            # Preserve original detailed JSON-lines log.
            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'test_{k}': v for k, v in test_stats.items()},
                'epoch': epoch,
                'n_parameters': n_parameters,
            }

            if self.output_dir and dist.is_main_process():
                with (self.output_dir / 'log.txt').open('a') as f:
                    f.write(json.dumps(log_stats) + '\n')

                # Preserve raw COCO evaluation data.
                if coco_evaluator is not None:
                    (self.output_dir / 'eval').mkdir(exist_ok=True)
                    if 'bbox' in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(
                                coco_evaluator.coco_eval['bbox'].eval,
                                self.output_dir / 'eval' / name,
                            )

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

    def val(self, ):
        self.eval()

        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)

        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(
            module,
            self.criterion,
            self.postprocessor,
            self.val_dataloader,
            base_ds,
            self.device,
            self.output_dir,
        )

        if self.output_dir and coco_evaluator is not None:
            (self.output_dir / 'eval').mkdir(exist_ok=True)
            if 'bbox' in coco_evaluator.coco_eval:
                dist.save_on_master(
                    coco_evaluator.coco_eval['bbox'].eval,
                    self.output_dir / 'eval' / 'val_latest.pth',
                )

        return test_stats
