"""
频域特征提取模块 - SRM (Spatial Rich Model) 滤波器

SRM 滤波器最初用于图像隐写分析，能有效提取图像中的高频噪声残差。
在图像篡改检测中，篡改区域与原始区域的噪声模式存在差异，
SRM 滤波器能够放大这些微妙的不一致性。

本模块实现：
1. 三个固定的 SRM 滤波核（不可训练）
2. 一个可学习的滤波核（自适应学习）
3. 频域特征提取 CNN 分支
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def get_srm_kernels() -> np.ndarray:
    """
    生成 3 个经典 SRM 滤波核

    这些滤波器源自 Fridrich & Kodovsky 的隐写分析工作，
    专门用于提取图像的噪声残差和高频信息：

    - Filter 1: 一阶边缘检测核（水平差分）
    - Filter 2: 二阶拉普拉斯核（各向同性，无归一化）
    - Filter 3: 三阶 SQUARE 核（高阶统计特征）

    返回:
        shape = (3, 1, 5, 5) 的滤波核数组
    """

    # SRM Filter 1: 一阶差分（简化版 Edge 核）
    # 捕获相邻像素的一阶统计差异
    filter1 = np.array([
        [ 0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0],
        [ 0,  1, -2,  1,  0],
        [ 0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0],
    ], dtype=np.float32)

    # SRM Filter 2: 二阶拉普拉斯核
    # 检测图像平滑度的局部异常
    filter2 = np.array([
        [ 0,  0,  0,  0,  0],
        [ 0,  0,  1,  0,  0],
        [ 0,  1, -4,  1,  0],
        [ 0,  0,  1,  0,  0],
        [ 0,  0,  0,  0,  0],
    ], dtype=np.float32)

    # SRM Filter 3: 高阶 SQUARE 核
    # 捕获更复杂的噪声模式
    filter3 = np.array([
        [-1,  2, -2,  2, -1],
        [ 2, -6,  8, -6,  2],
        [-2,  8, -12, 8, -2],
        [ 2, -6,  8, -6,  2],
        [-1,  2, -2,  2, -1],
    ], dtype=np.float32) / 12.0  # 归一化

    # 堆叠为 (3, 1, 5, 5) - 3 个单通道 5x5 滤波器
    kernels = np.stack([filter1, filter2, filter3])
    return kernels.reshape(3, 1, 5, 5)


class SRMFilterModule(nn.Module):
    """
    SRM 滤波器模块

    包含 3 个固定的 SRM 滤波核和 1 个可学习的滤波核，
    将 RGB 图像的每个通道分别经过 4 个滤波器处理，
    输出 12 通道 (3 通道 × 4 滤波器) 的噪声残差特征图。

    参数:
        requires_grad: SRM 核是否可训练，默认 False（固定）
    """

    def __init__(self, requires_grad: bool = False):
        super().__init__()

        # 固定 SRM 滤波核 (3 个滤波器，每个单通道 5x5)
        srm_kernels = get_srm_kernels()  # (3, 1, 5, 5)
        srm_tensor = torch.from_numpy(srm_kernels).float()
        self.register_buffer("srm_kernels", srm_tensor)

        # 可学习滤波核 (1 个，初始化为小随机值)
        self.learnable_kernel = nn.Parameter(
            torch.randn(1, 1, 5, 5) * 0.01
        )

        # 对 SRM 核设置梯度开关
        if not requires_grad:
            self.srm_kernels.requires_grad_(False)

        # 后处理：BatchNorm + ReLU，归一化滤波输出
        self.bn = nn.BatchNorm2d(12)  # 3 RGB 通道 × 4 滤波器 = 12
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        对输入图像应用 SRM 滤波

        参数:
            x: 输入 RGB 图像 (B, 3, H, W)

        返回:
            噪声残差特征图 (B, 12, H, W)
        """
        B, C, H, W = x.shape
        assert C == 3, f"输入应为 3 通道 RGB，但得到 {C} 通道"

        # 合并所有滤波核: (3 固定 + 1 可学习) = (4, 1, 5, 5)
        all_kernels = torch.cat(
            [self.srm_kernels, self.learnable_kernel], dim=0
        )

        # 对每个 RGB 通道分别应用 4 个滤波器
        outputs = []
        for c in range(C):
            channel = x[:, c:c+1, :, :]  # (B, 1, H, W)
            filtered = F.conv2d(channel, all_kernels, padding=2)  # (B, 4, H, W)
            outputs.append(filtered)

        # 拼接所有通道的滤波结果: (B, 12, H, W)
        out = torch.cat(outputs, dim=1)

        # 归一化和激活
        out = self.bn(out)
        out = self.relu(out)

        return out


class FrequencyBranch(nn.Module):
    """
    频域分支完整实现

    SRM 滤波 → 通道适配 → ResNet-18 骨干网络

    参数:
        pretrained: ResNet-18 是否使用预训练权重
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()

        from torchvision import models

        # SRM 滤波层
        self.srm = SRMFilterModule(requires_grad=False)

        # 通道适配：12 通道 → 64 通道（匹配 ResNet 内部维度）
        self.channel_adapter = nn.Sequential(
            nn.Conv2d(12, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # ResNet-18 骨干网络（去掉初始的 conv1 和 maxpool）
        resnet = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )
        self.layer1 = resnet.layer1  # 64 → 64, H/4
        self.layer2 = resnet.layer2  # 64 → 128, H/8
        self.layer3 = resnet.layer3  # 128 → 256, H/16
        self.layer4 = resnet.layer4  # 256 → 512, H/32

    def forward(self, x: torch.Tensor) -> dict:
        """
        频域分支前向传播

        返回各层级的特征图，用于后续与 RGB 流融合。
        """
        # SRM 滤波
        noise_residual = self.srm(x)  # (B, 12, H, W)

        # 通道适配
        feat = self.channel_adapter(noise_residual)  # (B, 64, H/4, W/4)

        # 逐层提取特征
        f1 = self.layer1(feat)   # (B, 64, H/4, W/4)
        f2 = self.layer2(f1)     # (B, 128, H/8, W/8)
        f3 = self.layer3(f2)     # (B, 256, H/16, W/16)
        f4 = self.layer4(f3)     # (B, 512, H/32, W/32)

        return {
            "low_level": f1,  # 浅层特征，用于解码器
            "layer2": f2,
            "layer3": f3,
            "layer4": f4,     # 深层特征，输入 ASPP
        }


if __name__ == "__main__":
    # 测试 SRM 模块
    print("=" * 50)
    print("SRM 滤波器模块测试")
    print("=" * 50)

    srm = SRMFilterModule()
    dummy = torch.randn(2, 3, 256, 256)
    out = srm(dummy)
    print(f"输入: {dummy.shape}")
    print(f"输出: {out.shape}")
    print(f"SRM 参数量: {sum(p.numel() for p in srm.parameters()):,}")

    print("\n频域分支测试")
    freq = FrequencyBranch(pretrained=False)
    feats = freq(dummy)
    for name, feat in feats.items():
        print(f"  {name}: {feat.shape}")
    print(f"频域分支参数量: {sum(p.numel() for p in freq.parameters()):,}")
