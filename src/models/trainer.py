"""
训练器模块

负责训练循环、验证、早停 (Early Stopping) 和模型存档。
支持 DualStreamNet 和 SingleStreamBaseline 两种模型。
"""

import os
import time
import logging
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    早停机制

    监控验证集指标，当连续 patience 个 epoch 无改善时停止训练。

    参数:
        patience: 容忍的无改善 epoch 数
        min_delta: 最小改善阈值
        mode: 'min'（损失）或 'max'（指标如 F1）
    """

    def __init__(self, patience: int = 10, min_delta: float = 1e-4, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "min":
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


class Trainer:
    """
    模型训练器

    参数:
        model: 模型实例 (DualStreamNet 或 SingleStreamBaseline)
        criterion: 损失函数
        optimizer: 优化器
        device: 计算设备 ('cuda' 或 'cpu')
        checkpoint_dir: 模型检查点保存目录
        max_epochs: 最大训练轮数
        patience: 早停耐心值
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = "cuda",
        checkpoint_dir: str = "models/checkpoints",
        max_epochs: int = 100,
        patience: int = 15,
        accumulation_steps: int = 1,
        use_amp: bool = True,
    ):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_epochs = max_epochs
        self.accumulation_steps = max(1, accumulation_steps)
        self.use_amp = (
            use_amp
            and str(device).startswith("cuda")
            and torch.cuda.is_available()
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # 学习率调度器
        self.scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)

        # 早停
        self.early_stopping = EarlyStopping(patience=patience, mode="min")

        # 训练记录
        self.history = {"train_loss": [], "val_loss": [], "val_f1": [], "lr": []}
        self.best_val_loss = float("inf")

    def _compute_losses_and_probs(
        self,
        output: Dict[str, torch.Tensor],
        masks: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        统一处理 logits/prob 输出，返回损失与概率图。
        """
        if "logits" in output:
            logits = output["logits"]
            if logits.shape[2:] != masks.shape[2:]:
                logits = torch.nn.functional.interpolate(
                    logits, size=masks.shape[2:],
                    mode="bilinear", align_corners=False
                )
            losses = self.criterion(logits, masks)
            probs = torch.sigmoid(logits)
        else:
            probs = output["pred"]
            if probs.shape[2:] != masks.shape[2:]:
                probs = torch.nn.functional.interpolate(
                    probs, size=masks.shape[2:],
                    mode="bilinear", align_corners=False
                )
            logits = torch.logit(probs.clamp(1e-4, 1 - 1e-4))
            losses = self.criterion(logits, masks)

        return {"losses": losses, "probs": probs}

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """执行一个训练 epoch"""
        self.model.train()
        total_loss = 0.0
        total_bce = 0.0
        total_dice = 0.0
        num_batches = 0

        self.optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader, start=1):
            images = batch["image"].to(self.device, non_blocking=True)
            masks = batch["mask"].to(self.device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                output = self.model(images)
                computed = self._compute_losses_and_probs(output, masks)
                losses = computed["losses"]
                loss = losses["total"] / self.accumulation_steps

            self.scaler.scale(loss).backward()

            if step % self.accumulation_steps == 0 or step == len(train_loader):
                self.scaler.unscale_(self.optimizer)
                # 梯度裁剪，防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            total_loss += losses["total"].item()
            total_bce += losses["bce"].item()
            total_dice += losses["dice"].item()
            num_batches += 1

        return {
            "loss": total_loss / max(num_batches, 1),
            "bce": total_bce / max(num_batches, 1),
            "dice": total_dice / max(num_batches, 1),
        }

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """执行验证"""
        self.model.eval()
        total_loss = 0.0
        total_f1 = 0.0
        num_batches = 0

        for batch in val_loader:
            images = batch["image"].to(self.device, non_blocking=True)
            masks = batch["mask"].to(self.device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                output = self.model(images)
                computed = self._compute_losses_and_probs(output, masks)
                losses = computed["losses"]
                probs = computed["probs"]

            total_loss += losses["total"].item()

            # 像素级 F1-score
            pred_binary = (probs > 0.5).float()
            f1 = self._pixel_f1(pred_binary, masks)
            total_f1 += f1

            num_batches += 1

        return {
            "loss": total_loss / max(num_batches, 1),
            "f1": total_f1 / max(num_batches, 1),
        }

    def _pixel_f1(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """计算像素级 F1-Score"""
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)

        tp = (pred_flat * target_flat).sum().item()
        fp = (pred_flat * (1 - target_flat)).sum().item()
        fn = ((1 - pred_flat) * target_flat).sum().item()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        return f1

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> Dict:
        """
        完整训练流程

        返回:
            训练历史记录
        """
        logger.info(f"开始训练，最大 {self.max_epochs} 个 epoch")

        for epoch in range(1, self.max_epochs + 1):
            start_time = time.time()

            # 训练
            train_metrics = self.train_epoch(train_loader)
            # 验证
            val_metrics = self.validate(val_loader)

            # 调整学习率
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]

            # 记录历史
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_f1"].append(val_metrics["f1"])
            self.history["lr"].append(current_lr)

            elapsed = time.time() - start_time

            # 日志输出
            msg = (
                f"Epoch {epoch:3d}/{self.max_epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val F1: {val_metrics['f1']:.4f} | "
                f"LR: {current_lr:.6f} | "
                f"Time: {elapsed:.1f}s"
            )
            logger.info(msg)
            print(msg)

            # 保存最佳模型
            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self._save_checkpoint(epoch, val_metrics, is_best=True)

            # 每 10 个 epoch 保存一次
            if epoch % 10 == 0:
                self._save_checkpoint(epoch, val_metrics, is_best=False)

            # 早停检查
            if self.early_stopping(val_metrics["loss"]):
                logger.info(f"早停触发，在 epoch {epoch} 停止训练")
                print(f"早停触发，在 epoch {epoch} 停止训练")
                break

        return self.history

    def _save_checkpoint(
        self, epoch: int, metrics: Dict, is_best: bool = False
    ):
        """保存模型检查点"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": metrics,
            "history": self.history,
        }

        if is_best:
            path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, path)
            logger.info(f"保存最佳模型: {path}")
        else:
            path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
            torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """加载模型检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "history" in checkpoint:
            self.history = checkpoint["history"]
        logger.info(f"加载检查点: {path} (epoch {checkpoint.get('epoch', '?')})")
