"""
结果可视化叠加模块

利用 OpenCV 实现检测结果的可视化：
1. 掩码叠加：将预测掩码半透明叠加到原图上
2. 热力图可视化：将置信度概率图转为彩色热力图
3. 边界绘制：在篡改区域边界画高亮轮廓
4. 并排对比图：原图 vs 检测结果
"""

import cv2
import numpy as np
from typing import Tuple, Optional


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.4,
) -> np.ndarray:
    """
    将二值掩码半透明叠加到原图上

    参数:
        image: 原始图像 (H, W, 3) BGR 格式
        mask: 二值掩码 (H, W)，0 = 真实, >0 = 篡改
        color: 掩码叠加颜色 (B, G, R)，默认红色
        alpha: 透明度 (0 = 完全透明, 1 = 完全不透明)

    返回:
        叠加后的图像 (H, W, 3)
    """
    result = image.copy()

    # 确保掩码是二值的
    binary_mask = (mask > 127).astype(np.uint8) if mask.max() > 1 else mask.astype(np.uint8)

    # 创建彩色叠加层
    overlay = np.zeros_like(image)
    overlay[:] = color

    # 在掩码区域进行 alpha 混合
    mask_3ch = np.stack([binary_mask] * 3, axis=-1)
    result = np.where(
        mask_3ch > 0,
        cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0),
        image,
    )

    return result


def create_heatmap(
    probability_map: np.ndarray,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    将概率图转为彩色热力图

    参数:
        probability_map: 概率图 (H, W)，值域 [0, 1]
        colormap: OpenCV 颜色映射，默认 JET

    返回:
        热力图 (H, W, 3) BGR
    """
    # 映射到 [0, 255]
    heatmap_gray = (probability_map * 255).clip(0, 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_gray, colormap)

    return heatmap_color


def overlay_heatmap(
    image: np.ndarray,
    probability_map: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    将置信度热力图叠加到原图上

    参数:
        image: 原始图像 (H, W, 3)
        probability_map: 概率图 (H, W)，值域 [0, 1]
        alpha: 热力图透明度
        colormap: 颜色映射

    返回:
        叠加后的图像
    """
    heatmap = create_heatmap(probability_map, colormap)

    # 确保尺寸匹配
    if heatmap.shape[:2] != image.shape[:2]:
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    return cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)


def draw_contours(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """
    在篡改区域边界画高亮轮廓

    参数:
        image: 原始图像
        mask: 二值掩码
        color: 轮廓颜色 (B, G, R)
        thickness: 轮廓线宽度

    返回:
        画有轮廓的图像
    """
    result = image.copy()

    # 二值化
    binary = (mask > 127).astype(np.uint8) * 255 if mask.max() > 1 else (mask * 255).astype(np.uint8)

    # 找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 画轮廓
    cv2.drawContours(result, contours, -1, color, thickness)

    return result


def create_comparison(
    original: np.ndarray,
    prediction: np.ndarray,
    mask_overlay: Optional[np.ndarray] = None,
    padding: int = 10,
) -> np.ndarray:
    """
    创建并排对比图

    参数:
        original: 原始图像
        prediction: 预测掩码（灰度或彩色）
        mask_overlay: 掩码叠加图（可选）
        padding: 图像间距

    返回:
        并排对比图
    """
    h, w = original.shape[:2]

    # 确保预测图与原图尺寸一致
    if prediction.shape[:2] != (h, w):
        prediction = cv2.resize(prediction, (w, h))

    # 如果预测是灰度图，转为三通道
    if len(prediction.shape) == 2:
        prediction = cv2.cvtColor(prediction, cv2.COLOR_GRAY2BGR)

    images = [original, prediction]
    if mask_overlay is not None:
        if mask_overlay.shape[:2] != (h, w):
            mask_overlay = cv2.resize(mask_overlay, (w, h))
        images.append(mask_overlay)

    # 拼接
    n = len(images)
    canvas_w = w * n + padding * (n - 1)
    canvas = np.ones((h, canvas_w, 3), dtype=np.uint8) * 255  # 白色间距

    for i, img in enumerate(images):
        x_start = i * (w + padding)
        canvas[:, x_start:x_start + w] = img

    return canvas


def generate_detection_report(
    probability_map: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    生成检测统计报告

    参数:
        probability_map: 概率图 (H, W)
        threshold: 二值化阈值

    返回:
        报告字典
    """
    binary_mask = (probability_map > threshold).astype(np.uint8)

    total_pixels = binary_mask.size
    tampered_pixels = binary_mask.sum()
    tampered_ratio = tampered_pixels / total_pixels

    # 连通域分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=8
    )

    # 去掉背景（label=0）
    regions = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        cx, cy = centroids[i]

        regions.append({
            "id": i,
            "area": int(area),
            "bbox": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
            "centroid": {"x": float(cx), "y": float(cy)},
            "mean_confidence": float(probability_map[labels == i].mean()),
        })

    # 按区域面积降序排列
    regions.sort(key=lambda r: r["area"], reverse=True)

    return {
        "is_tampered": bool(tampered_ratio > 0.01),
        "confidence": float(probability_map.max()),
        "tampered_ratio": float(tampered_ratio),
        "tampered_pixels": int(tampered_pixels),
        "total_pixels": int(total_pixels),
        "num_regions": len(regions),
        "regions": regions,
    }
