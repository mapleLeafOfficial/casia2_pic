"""
ImageStandardizer 图像标准化模块

将不同格式、分辨率的输入图像统一调整至网络适配的张量尺寸，
确保输入数据的一致性。支持以下功能：

1. 尺寸标准化：统一缩放至目标分辨率 (默认 512x512)
2. 像素值归一化：[0, 255] → [0, 1]，可选 ImageNet 标准化
3. 通道标准化：确保 RGB 三通道输入
4. 分块模式 (Patch-wise)：对高分辨率图像进行滑窗分块处理
"""

import numpy as np
import cv2
import torch
from typing import Tuple, List, Optional, Dict


# ImageNet 预训练模型标准化参数
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ImageStandardizer:
    """
    图像预处理标准化器

    将各种格式、分辨率的输入图像统一转换为模型可接受的标准张量。

    参数:
        target_size: 目标输出尺寸 (H, W)，默认 (512, 512)
        normalize: 是否进行 ImageNet 标准化，默认 True
        keep_aspect_ratio: 是否保持宽高比（使用 padding），默认 False
        interpolation: 缩放插值方法，默认双线性插值
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (512, 512),
        normalize: bool = True,
        keep_aspect_ratio: bool = False,
        interpolation: int = cv2.INTER_LINEAR,
    ):
        self.target_size = target_size  # (H, W)
        self.normalize = normalize
        self.keep_aspect_ratio = keep_aspect_ratio
        self.interpolation = interpolation

        # ImageNet 标准化参数
        self.mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(IMAGENET_STD, dtype=np.float32).reshape(1, 1, 3)

    def __call__(
        self, image: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> Dict[str, torch.Tensor]:
        """
        对图像（及可选掩码）进行标准化处理

        参数:
            image: 输入图像，支持 BGR 或 RGB 格式，shape = (H, W, 3) 或 (H, W)
            mask: 可选的二值掩码，shape = (H, W)

        返回:
            字典包含：
                - image: 标准化后的张量 (3, target_H, target_W)
                - mask: 标准化后的掩码张量 (1, target_H, target_W)（如果提供）
                - original_size: 原始图像尺寸 (H, W)
                - scale_factor: 缩放比例 (scale_h, scale_w)
                - padding: 填充量 (top, bottom, left, right)（仅保持宽高比时有意义）
        """
        original_size = image.shape[:2]  # (H, W)
        result = {"original_size": original_size}

        # 步骤 1：确保 RGB 三通道
        image = self._ensure_rgb(image)

        # 步骤 2：调整尺寸
        if self.keep_aspect_ratio:
            image, padding, scale = self._resize_with_padding(image)
            result["padding"] = padding
            result["scale_factor"] = scale
        else:
            image = cv2.resize(
                image, (self.target_size[1], self.target_size[0]),
                interpolation=self.interpolation
            )
            scale_h = self.target_size[0] / original_size[0]
            scale_w = self.target_size[1] / original_size[1]
            result["scale_factor"] = (scale_h, scale_w)
            result["padding"] = (0, 0, 0, 0)

        # 步骤 3：像素值归一化到 [0, 1]
        image = image.astype(np.float32) / 255.0

        # 步骤 4：可选 ImageNet 标准化
        if self.normalize:
            image = (image - self.mean) / self.std

        # 步骤 5：转换为 PyTorch 张量 (H, W, 3) → (3, H, W)
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
        result["image"] = image_tensor

        # 处理掩码（如果提供）
        if mask is not None:
            mask = self._standardize_mask(mask, original_size)
            result["mask"] = mask

        return result

    def _ensure_rgb(self, image: np.ndarray) -> np.ndarray:
        """确保图像为 RGB 三通道格式"""
        if len(image.shape) == 2:
            # 灰度图 → RGB
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            # RGBA → RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif image.shape[2] == 3:
            # 假设输入是 BGR (OpenCV 默认)，转为 RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def _resize_with_padding(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int], Tuple[float, float]]:
        """
        保持宽高比缩放，不足部分用黑色填充

        返回:
            resized_image: 缩放并填充后的图像
            padding: (top, bottom, left, right) 填充像素数
            scale: (scale_h, scale_w)
        """
        h, w = image.shape[:2]
        target_h, target_w = self.target_size

        # 计算等比例缩放比
        scale = min(target_h / h, target_w / w)
        new_h = int(h * scale)
        new_w = int(w * scale)

        # 缩放
        resized = cv2.resize(
            image, (new_w, new_h), interpolation=self.interpolation
        )

        # 计算填充量
        pad_top = (target_h - new_h) // 2
        pad_bottom = target_h - new_h - pad_top
        pad_left = (target_w - new_w) // 2
        pad_right = target_w - new_w - pad_left

        # 应用填充（黑色像素）
        padded = cv2.copyMakeBorder(
            resized, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )

        return padded, (pad_top, pad_bottom, pad_left, pad_right), (scale, scale)

    def _standardize_mask(
        self, mask: np.ndarray, original_size: Tuple[int, int]
    ) -> torch.Tensor:
        """标准化掩码：缩放、二值化、转张量"""
        # 确保是单通道
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        if self.keep_aspect_ratio:
            h, w = original_size
            target_h, target_w = self.target_size
            scale = min(target_h / h, target_w / w)
            new_h = int(h * scale)
            new_w = int(w * scale)

            mask = cv2.resize(
                mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST
            )

            pad_top = (target_h - new_h) // 2
            pad_bottom = target_h - new_h - pad_top
            pad_left = (target_w - new_w) // 2
            pad_right = target_w - new_w - pad_left

            mask = cv2.copyMakeBorder(
                mask, pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_CONSTANT, value=0
            )
        else:
            mask = cv2.resize(
                mask, (self.target_size[1], self.target_size[0]),
                interpolation=cv2.INTER_NEAREST
            )

        # 二值化 + 转张量
        mask = (mask > 127).astype(np.float32)
        return torch.from_numpy(mask).unsqueeze(0)

    def inverse_transform(
        self,
        tensor: torch.Tensor,
        original_size: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """
        反标准化：将模型输出张量转回可显示的图像

        参数:
            tensor: 模型输出张量 (3, H, W) 或 (1, H, W)
            original_size: 原始图像尺寸，用于恢复原始分辨率

        返回:
            numpy 图像数组 (H, W, 3) 或 (H, W)，值区间 [0, 255]
        """
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)

        image = tensor.cpu().numpy()

        if image.shape[0] == 1:
            # 掩码/概率图
            image = image.squeeze(0)
            image = (image * 255).clip(0, 255).astype(np.uint8)
        else:
            # RGB 图像
            image = image.transpose(1, 2, 0)  # (3, H, W) → (H, W, 3)
            if self.normalize:
                image = image * self.std.squeeze() + self.mean.squeeze()
            image = (image * 255).clip(0, 255).astype(np.uint8)

        # 恢复原始尺寸
        if original_size is not None:
            image = cv2.resize(
                image, (original_size[1], original_size[0]),
                interpolation=cv2.INTER_LINEAR
            )

        return image


class PatchExtractor:
    """
    分块提取器：对高分辨率图像进行滑窗分块处理

    当输入图像远大于模型输入尺寸时，将图像分割为多个重叠的块 (Patch)，
    分别进行推理后再拼合结果。

    参数:
        patch_size: 单个块的尺寸 (H, W)，默认 (512, 512)
        stride: 滑窗步长 (stride_h, stride_w)，默认 (384, 384)
        standardizer: ImageStandardizer 实例（可选）
    """

    def __init__(
        self,
        patch_size: Tuple[int, int] = (512, 512),
        stride: Tuple[int, int] = (384, 384),
        standardizer: Optional[ImageStandardizer] = None,
    ):
        self.patch_size = patch_size
        self.stride = stride
        self.standardizer = standardizer or ImageStandardizer(
            target_size=patch_size, normalize=True
        )

    def extract_patches(
        self, image: np.ndarray
    ) -> List[Dict]:
        """
        从高分辨率图像中提取重叠的块

        参数:
            image: 输入图像 (H, W, 3)

        返回:
            块信息列表，每个元素包含:
                - patch: 标准化后的块张量
                - position: (y, x) 块在原图中的左上角坐标
                - size: (h, w) 块的原始尺寸
        """
        h, w = image.shape[:2]
        patch_h, patch_w = self.patch_size
        stride_h, stride_w = self.stride

        patches = []

        for y in range(0, max(1, h - patch_h + 1), stride_h):
            for x in range(0, max(1, w - patch_w + 1), stride_w):
                # 确保不超出边界
                y_end = min(y + patch_h, h)
                x_end = min(x + patch_w, w)
                y_start = max(0, y_end - patch_h)
                x_start = max(0, x_end - patch_w)

                patch = image[y_start:y_end, x_start:x_end]
                standardized = self.standardizer(patch)

                patches.append({
                    "patch": standardized["image"],
                    "position": (y_start, x_start),
                    "size": (y_end - y_start, x_end - x_start),
                })

        return patches

    def merge_predictions(
        self,
        patches: List[Dict],
        predictions: List[torch.Tensor],
        original_size: Tuple[int, int],
    ) -> np.ndarray:
        """
        将多个块的预测结果拼合为完整的预测图

        使用加权平均处理重叠区域，确保拼接处平滑过渡。

        参数:
            patches: extract_patches 返回的块信息
            predictions: 每个块对应的预测张量 (1, H, W)
            original_size: 原始图像尺寸 (H, W)

        返回:
            合并后的预测图 (H, W)，值区间 [0, 1]
        """
        h, w = original_size
        prediction_map = np.zeros((h, w), dtype=np.float32)
        weight_map = np.zeros((h, w), dtype=np.float32)

        for patch_info, pred in zip(patches, predictions):
            y, x = patch_info["position"]
            ph, pw = patch_info["size"]

            # 将预测缩放到块的原始尺寸
            pred_np = pred.squeeze().cpu().numpy()
            pred_resized = cv2.resize(pred_np, (pw, ph))

            # 累加预测值和权重
            prediction_map[y:y + ph, x:x + pw] += pred_resized
            weight_map[y:y + ph, x:x + pw] += 1.0

        # 加权平均（避免除零）
        weight_map = np.maximum(weight_map, 1e-6)
        prediction_map /= weight_map

        return prediction_map
