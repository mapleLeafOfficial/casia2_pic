"""
单流 ResNet-18 基准模型

作为双流网络的性能基准线 (baseline)。使用 ResNet-18 编码器 +
简化版 DeepLabV3+ 解码器，实现端到端的像素级篡改区域分割。

架构:
    输入图像 (3, H, W)
      → ResNet-18 编码器 (ImageNet 预训练)
      → ASPP 模块 (多尺度空洞卷积)
      → 解码器 (上采样 + 1x1 卷积)
      → 篡改概率图 (1, H, W)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Dict, Optional


class ASPP(nn.Module):
    """
    空洞空间金字塔池化 (Atrous Spatial Pyramid Pooling)

    使用不同膨胀率的空洞卷积捕捉多尺度上下文信息，
    是 DeepLabV3+ 解码器的核心组件。

    参数:
        in_channels: 输入通道数
        out_channels: 输出通道数
        atrous_rates: 空洞卷积膨胀率列表
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 256,
        atrous_rates: tuple = (6, 12, 18),
    ):
        super().__init__()

        modules = []

        # 1x1 卷积分支
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ))

        # 不同膨胀率的 3x3 空洞卷积分支
        for rate in atrous_rates:
            modules.append(nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, 3,
                    padding=rate, dilation=rate, bias=False
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ))

        # 全局平均池化分支
        modules.append(nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ))

        self.convs = nn.ModuleList(modules)

        # 融合投影层
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * (len(atrous_rates) + 2), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = []
        for conv in self.convs:
            out = conv(x)
            # 全局池化分支需要上采样回原始尺寸
            if out.shape[2:] != x.shape[2:]:
                out = F.interpolate(
                    out, size=x.shape[2:], mode="bilinear", align_corners=False
                )
            res.append(out)

        # 拼接所有分支并投影
        return self.project(torch.cat(res, dim=1))


class Decoder(nn.Module):
    """
    DeepLabV3+ 风格解码器

    将 ASPP 输出与浅层特征融合，逐步上采样到原图分辨率。

    参数:
        low_level_channels: 浅层特征通道数 (ResNet layer1 输出)
        aspp_channels: ASPP 输出通道数
        num_classes: 输出类别数 (篡改检测为 1)
    """

    def __init__(
        self,
        low_level_channels: int = 64,
        aspp_channels: int = 256,
        num_classes: int = 1,
    ):
        super().__init__()

        # 浅层特征降维
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        # 融合卷积
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(aspp_channels + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

        # 最终分类头
        self.classifier = nn.Conv2d(256, num_classes, 1)

    def forward(
        self, aspp_out: torch.Tensor, low_level_feat: torch.Tensor
    ) -> torch.Tensor:
        """
        参数:
            aspp_out: ASPP 模块输出 (B, 256, H/16, W/16)
            low_level_feat: 浅层特征 (B, 64, H/4, W/4)
        """
        # 浅层特征降维
        low_level = self.low_level_conv(low_level_feat)

        # 上采样 ASPP 输出到浅层特征尺寸
        aspp_up = F.interpolate(
            aspp_out, size=low_level.shape[2:],
            mode="bilinear", align_corners=False
        )

        # 拼接融合
        fused = torch.cat([aspp_up, low_level], dim=1)
        fused = self.fuse_conv(fused)

        return self.classifier(fused)


class SingleStreamBaseline(nn.Module):
    """
    单流 ResNet-18 基准模型

    使用 ImageNet 预训练的 ResNet-18 作为编码器，
    配合 ASPP + DeepLabV3+ 解码器进行像素级分割。

    参数:
        pretrained: 是否使用 ImageNet 预训练权重
        num_classes: 输出通道数 (默认 1，二分类分割)
    """

    def __init__(self, pretrained: bool = True, num_classes: int = 1):
        super().__init__()

        # 加载预训练 ResNet-18
        resnet = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )

        # 编码器各阶段
        self.layer0 = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )
        self.layer1 = resnet.layer1  # 输出: 64 通道, H/4
        self.layer2 = resnet.layer2  # 输出: 128 通道, H/8
        self.layer3 = resnet.layer3  # 输出: 256 通道, H/16
        self.layer4 = resnet.layer4  # 输出: 512 通道, H/32

        # ASPP 模块
        self.aspp = ASPP(in_channels=512, out_channels=256)

        # DeepLabV3+ 解码器
        self.decoder = Decoder(
            low_level_channels=64,
            aspp_channels=256,
            num_classes=num_classes,
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播

        参数:
            x: 输入图像张量 (B, 3, H, W)

        返回:
            字典包含:
                - pred: 预测概率图 (B, 1, H, W)，经过 sigmoid
                - logits: 原始 logits (B, 1, H/4, W/4)
        """
        input_size = x.shape[2:]

        # 编码器
        x0 = self.layer0(x)       # (B, 64, H/4, W/4)
        x1 = self.layer1(x0)      # (B, 64, H/4, W/4)
        x2 = self.layer2(x1)      # (B, 128, H/8, W/8)
        x3 = self.layer3(x2)      # (B, 256, H/16, W/16)
        x4 = self.layer4(x3)      # (B, 512, H/32, W/32)

        # ASPP
        aspp_out = self.aspp(x4)  # (B, 256, H/32, W/32)

        # 解码器
        logits = self.decoder(aspp_out, x1)  # (B, 1, H/4, W/4)

        # 上采样到原始输入尺寸
        pred = F.interpolate(
            logits, size=input_size, mode="bilinear", align_corners=False
        )

        return {
            "pred": torch.sigmoid(pred),
            "logits": logits,
        }


if __name__ == "__main__":
    # 快速测试
    model = SingleStreamBaseline(pretrained=False)
    dummy_input = torch.randn(2, 3, 512, 512)
    output = model(dummy_input)
    print(f"输入尺寸:  {dummy_input.shape}")
    print(f"预测尺寸:  {output['pred'].shape}")
    print(f"Logits:   {output['logits'].shape}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
