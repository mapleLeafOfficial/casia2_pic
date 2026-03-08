# 图像篡改检测系统 - 完整使用与训练教程

## 🚀 快速开始 (最简部署方案)

如果你想跳过复杂的配置，直接使用以下命令即可一键完成环境安装与启动：

```bash
# 给予执行权限并启动
chmod +x run_simple.sh
./run_simple.sh
```

该脚本会自动创建虚拟环境、安装依赖并启动后端服务。启动后，直接在浏览器打开 `src/frontend/index.html` 即可使用。

---

## 1. 准备工作：设置 Python 虚拟环境 (手动模式)

```bash
# 进入项目目录
cd ~/codework/hard/基于深度学习的图像篡改检测方法研究与应用

# 创建名为 .venv 的虚拟环境
python3 -m venv .venv

# 激活虚拟环境 (每次打开新终端时都需要执行)
source .venv/bin/activate

# 安装所有依赖
pip install -r requirements.txt
```

## 2. 准备数据集

按照以下结构准备你的 CASIA v2.0 或 NIST16 数据集：

```text
data/raw/
├── CASIA2/           <-- 从下载的归档文件中解压出 Au 和 Tp 文件夹
│   ├── Au/          # 真实图像
│   ├── Tp/          # 篡改图像
│   └── mask/        # CASIA 的篡改区域掩码 (可选)

└── NIST16/           <-- 下载 Nimble Challenge 2016
    ├── probe/       # 待检测的篡改图像
    ├── world/       # 真实图像背景
    └── mask/        # Ground-truth 掩码图
```

## 3. 运行模型训练 (重要: 需要 train.py)

*(注：系统目前缺少训练循环的入口脚本 `src/train.py`。如果你希望开始训练，我可以帮你生成该脚本。基于现有的 `Trainer` 类，这将非常简单。)*

假设 `src/train.py` 就绪后，你可以这样开始训练：

```bash
# 确保在项目根目录，且激活了虚拟环境
source .venv/bin/activate
export PYTHONPATH=.

# 启动训练
python src/train.py
```

为了方便，可以创建一个快捷运行脚本 `run_training.sh`：
```bash
#!/bin/bash
source .venv/bin/activate
export PYTHONPATH=.
python src/train.py
```
运行：`chmod +x run_training.sh && ./run_training.sh`

## 4. 获取预训练模型 (重要)

系统默认不包含大型权重文件。在启动推理服务前，建议运行以下辅助脚本来准备模型：

```bash
# 在激活环境的情况下运行
python src/utils/download_pretrained.py
```

该脚本会：
1. 创建 `models/checkpoints` 目录。
2. 演示如何加载骨架权重（ResNet-18）。
3. **提供行业内主流篡改检测模型（如 ManTra-Net, MVSS-Net）的下载链接。**

> [!TIP]
> 如果你有现成的权重文件（如 `best_model.pth`），请手动放入 `models/checkpoints/` 目录下。

## 5. 运行模型推理 (Web 后端)

服务将在 `http://0.0.0.0:5000` 启动。

## 5. 访问 Web 前端

用浏览器打开文件：
`src/frontend/index.html`

拖拽图片即可开始自动检测！

## 6. 运行基准测试与模型导出

验证模型在有损压缩或加噪条件下的鲁棒性：

```bash
export PYTHONPATH=.
python src/utils/robustness_benchmark.py
```

将 PyTorch 模型导出为加速优化的 ONNX 格式：

```bash
export PYTHONPATH=.
python src/utils/export_onnx.py
```
