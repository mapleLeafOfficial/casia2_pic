"""
DualStreamNet - 双流融合网络架构

核心创新模型：同时利用 RGB 视觉特征和频域噪声残差特征，
实现高精度的图像篡改区域检测与定位。

架构:
    输入图像 (3, H, W)
          ├─→ RGB 流 (ResNet-18 编码器) ──────┐
          │                                     │  特征融合
          └─→ SRM 滤波 → 频域流 (ResNet-18) ──┘  (Channel Concat)
                                                │
                                          1x1 卷积降维
                                                │
                                          ASPP 模块
                                                │
                                        DeepLabV3+ 解码器
                                                │
                                        篡改概率图 (1, H, W)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Dict, Optional

from .baseline import ASPP, Decoder
from .srm_filters import SRMFilterModule


class DualStreamNet(nn.Module):
    """
    轻量化双流融合网络

    双流设计理念：
    - RGB 流捕获视觉语义不一致性（边界瑕疵、光照差异、几何失真）
    - 频域流捕获统计异常（传感器噪声残差、JPEG 压缩伪影、重采样痕迹）
    - 物理融合（通道拼接）确保轻量化

    参数:
        pretrained: 是否使用 ImageNet 预训练权重
        fusion_strategy: 融合策略 ('concat' 或 'add')
        num_classes: 输出通道数
    """

    def __init__(
        self,
        pretrained: bool = True,
        fusion_strategy: str = "concat",
        num_classes: int = 1,
    ):
        super().__init__()
        self.fusion_strategy = fusion_strategy

        # ========================
        # RGB 流：标准 ResNet-18
        # ========================
        rgb_resnet = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )

        self.rgb_layer0 = nn.Sequential(
            rgb_resnet.conv1, rgb_resnet.bn1, rgb_resnet.relu, rgb_resnet.maxpool
        )
        self.rgb_layer1 = rgb_resnet.layer1  # 64 ch, H/4
        self.rgb_layer2 = rgb_resnet.layer2  # 128 ch, H/8
        self.rgb_layer3 = rgb_resnet.layer3  # 256 ch, H/16
        self.rgb_layer4 = rgb_resnet.layer4  # 512 ch, H/32

        # ========================
        # 频域流：SRM + ResNet-18
        # ========================
        self.srm = SRMFilterModule(requires_grad=False)

        # SRM 输出 12 通道 → 适配到 64 通道
        self.freq_adapter = nn.Sequential(
            nn.Conv2d(12, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        freq_resnet = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )
        self.freq_layer1 = freq_resnet.layer1  # 64 ch
        self.freq_layer2 = freq_resnet.layer2  # 128 ch
        self.freq_layer3 = freq_resnet.layer3  # 256 ch
        self.freq_layer4 = freq_resnet.layer4  # 512 ch

        # ========================
        # 特征融合模块
        # ========================
        if fusion_strategy == "concat":
            # 拼接后 1024 通道 → 512 通道
            self.fusion_conv = nn.Sequential(
                nn.Conv2d(1024, 512, 1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
            )
            # 浅层融合：64 + 64 → 64
            self.low_level_fusion = nn.Sequential(
                nn.Conv2d(128, 64, 1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            )
        elif fusion_strategy == "add":
            # 元素相加，无需降维
            self.fusion_conv = nn.Identity()
            self.low_level_fusion = nn.Identity()
        else:
            raise ValueError(f"不支持的融合策略: {fusion_strategy}")

        # ========================
        # ASPP + 解码器
        # ========================
        self.aspp = ASPP(in_channels=512, out_channels=256)
        self.decoder = Decoder(
            low_level_channels=64,
            aspp_channels=256,
            num_classes=num_classes,
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        双流前向传播

        参数:
            x: 输入图像 (B, 3, H, W)

        返回:
            字典包含:
                - pred: 预测概率图 (B, 1, H, W)，经过 sigmoid
                - logits: 原始 logits (B, 1, H/4, W/4)
                - rgb_feat: RGB 流深层特征 (用于可视化/分析)
                - freq_feat: 频域流深层特征
        """
        input_size = x.shape[2:]

        # ---- RGB 流 ----
        rgb0 = self.rgb_layer0(x)         # (B, 64, H/4, W/4)
        rgb1 = self.rgb_layer1(rgb0)      # (B, 64, H/4, W/4)
        rgb2 = self.rgb_layer2(rgb1)      # (B, 128, H/8, W/8)
        rgb3 = self.rgb_layer3(rgb2)      # (B, 256, H/16, W/16)
        rgb4 = self.rgb_layer4(rgb3)      # (B, 512, H/32, W/32)

        # ---- 频域流 ----
        noise = self.srm(x)              # (B, 12, H, W)
        freq0 = self.freq_adapter(noise)  # (B, 64, H/4, W/4)
        freq1 = self.freq_layer1(freq0)   # (B, 64, H/4, W/4)
        freq2 = self.freq_layer2(freq1)   # (B, 128, H/8, W/8)
        freq3 = self.freq_layer3(freq2)   # (B, 256, H/16, W/16)
        freq4 = self.freq_layer4(freq3)   # (B, 512, H/32, W/32)

        # ---- 特征融合 ----
        if self.fusion_strategy == "concat":
            # 深层融合：第 4 阶段特征拼接 + 1x1 卷积降维
            fused_deep = self.fusion_conv(
                torch.cat([rgb4, freq4], dim=1)
            )  # (B, 512, H/32, W/32)

            # 浅层融合：第 1 阶段特征拼接 + 1x1 卷积降维
            fused_low = self.low_level_fusion(
                torch.cat([rgb1, freq1], dim=1)
            )  # (B, 64, H/4, W/4)
        else:
            # 元素相加
            fused_deep = rgb4 + freq4
            fused_low = rgb1 + freq1

        # ---- ASPP ----
        aspp_out = self.aspp(fused_deep)  # (B, 256, H/32, W/32)

        # ---- 解码器 ----
        logits = self.decoder(aspp_out, fused_low)  # (B, 1, H/4, W/4)

        # 上采样到原始输入尺寸
        pred = F.interpolate(
            logits, size=input_size, mode="bilinear", align_corners=False
        )

        return {
            "pred": torch.sigmoid(pred),
            "logits": logits,
            "rgb_feat": rgb4.detach(),
            "freq_feat": freq4.detach(),
        }

    def get_param_groups(self, lr: float = 1e-4) -> list:
        """
        获取分组学习率的参数组

        设计策略：
        - 预训练骨干：低学习率 (lr * 0.1)
        - SRM 滤波器：固定不训练
        - 新增模块（融合、ASPP、解码器）：标准学习率

        返回:
            参数组列表，可直接传入 optimizer
        """
        backbone_params = []
        new_params = []

        backbone_modules = [
            self.rgb_layer0, self.rgb_layer1, self.rgb_layer2,
            self.rgb_layer3, self.rgb_layer4,
            self.freq_layer1, self.freq_layer2,
            self.freq_layer3, self.freq_layer4,
        ]

        backbone_ids = set()
        for module in backbone_modules:
            for param in module.parameters():
                if param.requires_grad:
                    backbone_params.append(param)
                    backbone_ids.add(id(param))

        for param in self.parameters():
            if param.requires_grad and id(param) not in backbone_ids:
                new_params.append(param)

        return [
            {"params": backbone_params, "lr": lr * 0.1},  # 骨干：低学习率
            {"params": new_params, "lr": lr},              # 新模块：标准学习率
        ]


class DualStreamNetLite(DualStreamNet):
    """
    轻量化变体

    与 DualStreamNet 相同架构，但使用元素相加替代通道拼接，
    减少约 25% 的参数量，适合资源受限场景。
    """

    def __init__(self, pretrained: bool = True, num_classes: int = 1):
        super().__init__(
            pretrained=pretrained,
            fusion_strategy="add",
            num_classes=num_classes,
        )


if __name__ == "__main__":
    # 测试
    print("=" * 60)
    print("DualStreamNet 双流融合网络测试")
    print("=" * 60)

    # 标准版（通道拼接）
    model = DualStreamNet(pretrained=False, fusion_strategy="concat")
    dummy = torch.randn(2, 3, 512, 512)
    output = model(dummy)

    print(f"\n[Concat 版本]")
    print(f"  输入:  {dummy.shape}")
    print(f"  预测:  {output['pred'].shape}")
    print(f"  Logits: {output['logits'].shape}")
    print(f"  RGB 特征: {output['rgb_feat'].shape}")
    print(f"  频域特征: {output['freq_feat'].shape}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数: {trainable:,}")

    # 轻量版（元素相加）
    model_lite = DualStreamNetLite(pretrained=False)
    output_lite = model_lite(dummy)
    total_lite = sum(p.numel() for p in model_lite.parameters())
    print(f"\n[Add 版本]")
    print(f"  总参数量: {total_lite:,}")
    print(f"  参数减少: {(1 - total_lite/total_params) * 100:.1f}%")
