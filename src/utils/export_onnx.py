"""
ONNX 模型导出与推理优化

将训练好的 PyTorch 模型导出为 ONNX 格式，
并通过 ONNX Runtime 进行推理加速。
"""

import os
import time
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def export_to_onnx(
    model: nn.Module,
    output_path: str = "models/exported/model.onnx",
    input_size: Tuple[int, int] = (512, 512),
    opset_version: int = 14,
    dynamic_batch: bool = True,
) -> str:
    """
    将 PyTorch 模型导出为 ONNX 格式

    参数:
        model: PyTorch 模型实例
        output_path: 输出文件路径
        input_size: 模型输入尺寸 (H, W)
        opset_version: ONNX 算子集版本
        dynamic_batch: 是否支持动态 batch size

    返回:
        导出的 ONNX 文件路径
    """
    # 确保输出目录存在
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    model = model.cpu()

    # 构造哑输入
    dummy_input = torch.randn(1, 3, input_size[0], input_size[1])

    # 动态维度配置
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        }

    # 导出
    print(f"正在导出 ONNX 模型到: {output_path}")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
    )

    # 验证导出的模型
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print(f"✅ ONNX 模型验证通过")
    except ImportError:
        print("⚠️ onnx 未安装，跳过模型验证")
    except Exception as e:
        print(f"❌ ONNX 模型验证失败: {e}")

    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"模型大小: {file_size:.1f} MB")

    return output_path


class ONNXInference:
    """
    ONNX Runtime 推理加速器

    参数:
        onnx_path: ONNX 模型文件路径
        providers: 推理后端 (默认优先使用 CUDA)
    """

    def __init__(
        self,
        onnx_path: str,
        providers: list = None,
    ):
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("请安装 onnxruntime: pip install onnxruntime-gpu")

        if providers is None:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self.session = ort.InferenceSession(onnx_path, providers=providers)

        # 获取输入/输出信息
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        input_shape = self.session.get_inputs()[0].shape
        logger.info(f"ONNX 模型已加载: {onnx_path}")
        logger.info(f"输入名称: {self.input_name}, 形状: {input_shape}")

        # ImageNet 标准化参数
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def preprocess(self, image: np.ndarray, target_size: Tuple[int, int] = (512, 512)) -> np.ndarray:
        """预处理图像为模型输入格式"""
        import cv2

        # BGR → RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(image_rgb, (target_size[1], target_size[0]))

        # 归一化
        normalized = resized.astype(np.float32) / 255.0
        normalized = (normalized - self.mean) / self.std

        # (H, W, 3) → (1, 3, H, W)
        return normalized.transpose(2, 0, 1)[np.newaxis, ...]

    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        使用 ONNX Runtime 推理

        返回:
            概率图 (H, W)
        """
        input_tensor = self.preprocess(image)
        result = self.session.run(
            [self.output_name],
            {self.input_name: input_tensor}
        )
        # Sigmoid 激活（如果模型输出是 logits）
        output = result[0].squeeze()
        if output.min() < 0:
            output = 1 / (1 + np.exp(-output))
        return output

    def benchmark_speed(
        self, image: np.ndarray, num_runs: int = 100, warmup: int = 10
    ) -> dict:
        """
        推理速度基准测试

        返回:
            包含平均耗时、吞吐量等指标的字典
        """
        input_tensor = self.preprocess(image)

        # 预热
        for _ in range(warmup):
            self.session.run([self.output_name], {self.input_name: input_tensor})

        # 计时
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            self.session.run([self.output_name], {self.input_name: input_tensor})
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        times = np.array(times) * 1000  # 转为毫秒

        return {
            "mean_ms": float(times.mean()),
            "std_ms": float(times.std()),
            "min_ms": float(times.min()),
            "max_ms": float(times.max()),
            "fps": float(1000 / times.mean()),
            "num_runs": num_runs,
        }


class ModelWrapper(nn.Module):
    """
    模型导出包装器

    将 DualStreamNet 的字典输出转为单个 Tensor 输出，
    以兼容 ONNX 导出要求。
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.model(x)
        return output["pred"]


if __name__ == "__main__":
    print("=" * 50)
    print("ONNX 模型导出工具")
    print("=" * 50)

    # 示例
    from src.models.dual_stream import DualStreamNet

    model = DualStreamNet(pretrained=False)
    wrapper = ModelWrapper(model)

    onnx_path = export_to_onnx(
        wrapper,
        output_path="models/exported/dual_stream.onnx",
        input_size=(512, 512),
    )

    print(f"\n导出完成: {onnx_path}")
