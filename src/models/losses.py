"""
复合损失函数模块

实现用于图像篡改检测的复合损失函数：
1. Weighted BCE Loss：加权二值交叉熵，处理正负样本不平衡
2. Dice Loss：基于区域重叠的损失，优化分割边界
3. CompositeLoss：二者加权组合

设计考量：
- 篡改区域通常只占整图的一小部分，导致严重的类别不平衡
- BCE 关注像素级分类精度，Dice 关注区域整体匹配度
- 二者互补：BCE 提供稳定的梯度信号，Dice 关注困难边界区域
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedBCELoss(nn.Module):
    """
    加权二值交叉熵损失

    为篡改像素（正样本）赋予更高权重，缓解类别不平衡问题。

    参数:
        pos_weight: 正样本权重，默认 None（自动计算）
        reduction: 聚合方式 ('mean', 'sum', 'none')
    """

    def __init__(
        self,
        pos_weight: float = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        参数:
            logits: 模型原始输出（未经 sigmoid），shape = (B, 1, H, W)
            targets: Ground-truth 掩码，shape = (B, 1, H, W)，值域 [0, 1]

        返回:
            标量损失值
        """
        if self.pos_weight is not None:
            # 使用指定的正样本权重
            weight = torch.ones_like(targets)
            weight[targets > 0.5] = self.pos_weight
        else:
            # 自动计算权重：基于当前 batch 中正负样本比例
            num_pos = targets.sum().clamp(min=1)
            num_neg = targets.numel() - num_pos
            auto_weight = num_neg / num_pos
            weight = torch.ones_like(targets)
            weight[targets > 0.5] = auto_weight.clamp(max=50.0)  # 限制最大权重

        # 使用 F.binary_cross_entropy_with_logits（数值稳定）
        loss = F.binary_cross_entropy_with_logits(
            logits, targets, weight=weight, reduction=self.reduction
        )
        return loss


class DiceLoss(nn.Module):
    """
    Dice 损失

    基于 Sørensen–Dice 系数，衡量预测区域与真实区域的重叠程度。
    Dice = 2 * |A ∩ B| / (|A| + |B|)

    对小目标区域特别有效，因为它不会被大量背景像素稀释。

    参数:
        smooth: 平滑因子，防止除零，默认 1.0
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        参数:
            logits: 模型原始输出（未经 sigmoid），shape = (B, 1, H, W)
            targets: Ground-truth 掩码，shape = (B, 1, H, W)

        返回:
            1 - Dice coefficient（越小越好）
        """
        # Sigmoid 激活
        probs = torch.sigmoid(logits)

        # 展平为一维
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        # Dice 系数
        intersection = (probs_flat * targets_flat).sum()
        dice_coeff = (
            (2.0 * intersection + self.smooth)
            / (probs_flat.sum() + targets_flat.sum() + self.smooth)
        )

        return 1.0 - dice_coeff


class CompositeLoss(nn.Module):
    """
    复合损失函数

    组合加权 BCE 和 Dice Loss：
        L = α * BCE + β * Dice

    参数:
        bce_weight: BCE 损失的权重系数 α，默认 0.5
        dice_weight: Dice 损失的权重系数 β，默认 0.5
        pos_weight: BCE 中正样本的权重
        dice_smooth: Dice Loss 的平滑因子
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        pos_weight: float = None,
        dice_smooth: float = 1.0,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

        self.bce_loss = WeightedBCELoss(pos_weight=pos_weight)
        self.dice_loss = DiceLoss(smooth=dice_smooth)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> dict:
        """
        参数:
            logits: 模型原始输出（未经 sigmoid），(B, 1, H, W)
            targets: Ground-truth 掩码 (B, 1, H, W)

        返回:
            字典包含:
                - total: 总损失
                - bce: BCE 分量
                - dice: Dice 分量
        """
        bce = self.bce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)

        total = self.bce_weight * bce + self.dice_weight * dice

        return {
            "total": total,
            "bce": bce.detach(),
            "dice": dice.detach(),
        }


if __name__ == "__main__":
    # 测试
    print("=" * 50)
    print("复合损失函数测试")
    print("=" * 50)

    criterion = CompositeLoss(bce_weight=0.5, dice_weight=0.5)

    # 模拟预测和标签
    logits = torch.randn(4, 1, 128, 128)
    # 模拟篡改掩码（约 10% 的像素为篡改区域）
    targets = (torch.rand(4, 1, 128, 128) > 0.9).float()

    losses = criterion(logits, targets)
    print(f"总损失: {losses['total']:.4f}")
    print(f"BCE 分量: {losses['bce']:.4f}")
    print(f"Dice 分量: {losses['dice']:.4f}")
    print(f"正样本比例: {targets.mean():.4f}")
