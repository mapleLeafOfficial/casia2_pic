"""
鲁棒性基准测试套件

自动测试模型在各种图像降质条件下的性能稳定性：
- JPEG 压缩 (QF = 100, 90, 70, 50)
- 高斯噪声 (σ = 0, 0.5, 1.0, 2.0)
- 缩放变换 (0.5x, 0.8x, 1.0x, 1.2x, 1.5x, 2.0x)

按照技术规格要求：JPEG QF > 50 时，检测 AUC 下降应 < 10%。
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score
)
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ============== 图像降质操作 ==============

def apply_jpeg_compression(image: np.ndarray, quality: int) -> np.ndarray:
    """应用 JPEG 压缩"""
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded = cv2.imencode(".jpg", image, encode_param)
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def apply_gaussian_noise(image: np.ndarray, sigma: float) -> np.ndarray:
    """添加高斯噪声"""
    if sigma == 0:
        return image
    noise = np.random.normal(0, sigma * 255, image.shape).astype(np.float32)
    noisy = (image.astype(np.float32) + noise).clip(0, 255).astype(np.uint8)
    return noisy


def apply_scaling(image: np.ndarray, scale: float) -> np.ndarray:
    """缩放后恢复原始尺寸（模拟缩放攻击）"""
    if scale == 1.0:
        return image
    h, w = image.shape[:2]
    # 缩放
    new_h, new_w = int(h * scale), int(w * scale)
    scaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    # 恢复原始尺寸
    restored = cv2.resize(scaled, (w, h), interpolation=cv2.INTER_LINEAR)
    return restored


class RobustnessBenchmark:
    """
    自动化鲁棒性基准测试

    参数:
        model: 检测模型
        device: 推理设备
        output_dir: 结果输出目录
    """

    # 测试参数（来自 robustness-benchmark/spec.md）
    JPEG_QUALITIES = [100, 90, 70, 50]
    NOISE_SIGMAS = [0, 0.5, 1.0, 2.0]
    SCALE_FACTORS = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]

    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "cuda",
        output_dir: str = "outputs/benchmark",
        input_size: Tuple[int, int] = (512, 512),
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.input_size = input_size

        # ImageNet 标准化参数
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """图像预处理"""
        resized = cv2.resize(image, (self.input_size[1], self.input_size[0]))
        normalized = resized.astype(np.float32) / 255.0
        normalized = (normalized - self.mean) / self.std
        tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0).float()
        return tensor.to(self.device)

    @torch.no_grad()
    def predict(self, image: np.ndarray) -> np.ndarray:
        """单图推理，返回概率图"""
        tensor = self.preprocess(image)
        output = self.model(tensor)
        pred = output["pred"].squeeze().cpu().numpy()
        return pred

    def evaluate_dataset(
        self,
        images: List[np.ndarray],
        masks: List[np.ndarray],
        labels: List[int],
        attack_fn=None,
        attack_param=None,
        desc: str = "",
    ) -> Dict[str, float]:
        """
        在给定攻击条件下评估模型性能

        参数:
            images: 图像列表
            masks: Ground-truth 掩码列表
            labels: 图像级标签列表
            attack_fn: 攻击函数
            attack_param: 攻击参数

        返回:
            性能指标字典
        """
        all_preds = []
        all_labels = []
        pixel_preds = []
        pixel_truths = []

        for image, mask, label in tqdm(
            zip(images, masks, labels),
            total=len(images),
            desc=desc,
            leave=False,
        ):
            # 应用攻击
            if attack_fn is not None and attack_param is not None:
                attacked = attack_fn(image, attack_param)
            else:
                attacked = image

            # 推理
            prob_map = self.predict(attacked)

            # 图像级预测
            image_pred = float(prob_map.max())
            all_preds.append(image_pred)
            all_labels.append(label)

            # 像素级预测
            if mask is not None:
                mask_resized = cv2.resize(
                    mask, (prob_map.shape[1], prob_map.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )
                pixel_preds.extend(prob_map.flatten().tolist())
                pixel_truths.extend(mask_resized.flatten().tolist())

        metrics = {}

        # 图像级 AUC
        try:
            metrics["image_auc"] = roc_auc_score(all_labels, all_preds)
        except ValueError:
            metrics["image_auc"] = 0.0

        # 像素级指标
        if pixel_preds:
            pixel_preds_binary = [1 if p > 0.5 else 0 for p in pixel_preds]
            pixel_truths_binary = [1 if t > 0.5 else 0 for t in pixel_truths]

            metrics["pixel_f1"] = f1_score(pixel_truths_binary, pixel_preds_binary, zero_division=0)
            metrics["pixel_precision"] = precision_score(pixel_truths_binary, pixel_preds_binary, zero_division=0)
            metrics["pixel_recall"] = recall_score(pixel_truths_binary, pixel_preds_binary, zero_division=0)

            try:
                metrics["pixel_auc"] = roc_auc_score(pixel_truths_binary, pixel_preds)
            except ValueError:
                metrics["pixel_auc"] = 0.0

        return metrics

    def run_full_benchmark(
        self,
        images: List[np.ndarray],
        masks: List[np.ndarray],
        labels: List[int],
    ) -> Dict:
        """
        运行完整的鲁棒性基准测试

        返回:
            {
                "baseline": {...metrics...},
                "jpeg": {"100": {...}, "90": {...}, ...},
                "noise": {"0": {...}, "0.5": {...}, ...},
                "scale": {"0.5": {...}, "1.0": {...}, ...},
                "summary": {...}
            }
        """
        results = {}

        print("\n" + "=" * 60)
        print("鲁棒性基准测试")
        print("=" * 60)

        # 基线性能（无攻击）
        print("\n[基线] 无降质...")
        results["baseline"] = self.evaluate_dataset(
            images, masks, labels, desc="基线评估"
        )
        print(f"  Image AUC: {results['baseline'].get('image_auc', 0):.4f}")
        print(f"  Pixel F1:  {results['baseline'].get('pixel_f1', 0):.4f}")

        # JPEG 压缩测试
        results["jpeg"] = {}
        print("\n[JPEG 压缩测试]")
        for qf in self.JPEG_QUALITIES:
            print(f"  QF={qf}...", end=" ")
            results["jpeg"][str(qf)] = self.evaluate_dataset(
                images, masks, labels,
                attack_fn=apply_jpeg_compression,
                attack_param=qf,
                desc=f"JPEG QF={qf}",
            )
            auc = results["jpeg"][str(qf)].get("image_auc", 0)
            drop = results["baseline"].get("image_auc", 0) - auc
            print(f"AUC={auc:.4f} (下降 {drop:.4f})")

        # 高斯噪声测试
        results["noise"] = {}
        print("\n[高斯噪声测试]")
        for sigma in self.NOISE_SIGMAS:
            print(f"  σ={sigma}...", end=" ")
            results["noise"][str(sigma)] = self.evaluate_dataset(
                images, masks, labels,
                attack_fn=apply_gaussian_noise,
                attack_param=sigma,
                desc=f"Noise σ={sigma}",
            )
            auc = results["noise"][str(sigma)].get("image_auc", 0)
            print(f"AUC={auc:.4f}")

        # 缩放测试
        results["scale"] = {}
        print("\n[缩放变换测试]")
        for scale in self.SCALE_FACTORS:
            print(f"  ×{scale}...", end=" ")
            results["scale"][str(scale)] = self.evaluate_dataset(
                images, masks, labels,
                attack_fn=apply_scaling,
                attack_param=scale,
                desc=f"Scale ×{scale}",
            )
            auc = results["scale"][str(scale)].get("image_auc", 0)
            print(f"AUC={auc:.4f}")

        # 汇总
        baseline_auc = results["baseline"].get("image_auc", 0)
        jpeg_50_auc = results["jpeg"].get("50", {}).get("image_auc", 0)
        jpeg_drop = baseline_auc - jpeg_50_auc

        results["summary"] = {
            "baseline_image_auc": baseline_auc,
            "jpeg_qf50_auc_drop": jpeg_drop,
            "jpeg_qf50_pass": jpeg_drop < 0.10,  # 规格要求：下降 < 10%
            "timestamp": datetime.now().isoformat(),
        }

        print("\n" + "=" * 60)
        print(f"基线 AUC: {baseline_auc:.4f}")
        print(f"JPEG QF=50 AUC 下降: {jpeg_drop:.4f} ({'✅ 通过' if jpeg_drop < 0.10 else '❌ 未通过'})")
        print("=" * 60)

        # 保存结果
        report_path = self.output_dir / "benchmark_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n报告已保存至: {report_path}")

        return results
