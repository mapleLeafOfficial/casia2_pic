# Kaggle 免费 GPU 训练教程

本教程基于本项目，目标是在 Kaggle Notebook 的免费 GPU 环境中启动训练。

## 1. 在 Kaggle 新建 Notebook

1. 打开 https://www.kaggle.com/ 并登录。
2. 点击 `Code` -> `New Notebook`。
3. 右侧 `Settings` 中设置：
   - `Accelerator`: `GPU`
   - `Internet`: `On`（用于安装依赖或拉代码）

## 2. 准备项目代码

你可以任选一种方式：

1. Git 克隆（推荐）：

```python
!git clone <你的仓库地址> /kaggle/working/your-project
%cd /kaggle/working/your-project
```

2. 上传 zip 到 Notebook，再解压：

```python
!unzip -q /kaggle/input/<你的zip数据集>/<项目压缩包>.zip -d /kaggle/working/
%cd /kaggle/working/your-project
```

## 3. 准备数据集（CASIA2/NIST16）

1. 在 Notebook 右侧点击 `Add input`，添加你上传到 Kaggle Datasets 的数据集。
2. 记录数据路径，例如：
   - `/kaggle/input/casia2/CASIA2`
   - `/kaggle/input/nist16/NIST16`

## 4. 运行一键训练脚本

项目已提供脚本 [`kaggle_train.sh`](/d:/code-work/开题报告final/基于深度学习的图像篡改检测方法研究与应用/kaggle_train.sh)。

在 Notebook 单元执行：

```python
!chmod +x /kaggle/working/your-project/kaggle_train.sh
!PROJECT_DIR=/kaggle/working/your-project \
 CASIA_ROOT=/kaggle/input/casia2/CASIA2 \
 NIST_ROOT=/kaggle/input/nist16/NIST16 \
 MODEL=dual_stream_lite \
 BATCH_SIZE=4 \
 EPOCHS=5 \
 bash /kaggle/working/your-project/kaggle_train.sh
```

## 5. 常见问题

1. `CUDA unavailable`：
   - 检查 Notebook `Settings` 的 `Accelerator` 是否设为 `GPU`。
2. 数据读取失败：
   - 检查 `CASIA_ROOT` / `NIST_ROOT` 是否与 Kaggle Input 实际目录一致。
3. 显存不足：
   - 把 `BATCH_SIZE` 改小（如 `2`）。
   - 先用 `MODEL=dual_stream_lite`。
4. 训练时间不够：
   - 先用 `EPOCHS=1` 验证流程，再逐步增加。

## 6. 只做 GPU 连通性测试（可选）

```python
!python /kaggle/working/your-project/check_gpu.py
```

