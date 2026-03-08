# =============================================================================
# 基于深度学习的图像篡改检测系统 - Docker 基础镜像
# 基础环境：PyTorch 2.x + CUDA 11.8 + cuDNN 8
# =============================================================================

FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

LABEL maintainer="何荣博 <2231052349>"
LABEL description="图像篡改检测系统 - 深度学习推理与训练环境"

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# 设置工作目录
WORKDIR /app

# 安装系统依赖（OpenCV 需要 libgl1 等库）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制并安装 Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目代码
COPY . .

# 创建必要的数据目录
RUN mkdir -p /app/data/raw \
             /app/data/processed \
             /app/models/checkpoints \
             /app/models/exported \
             /app/logs \
             /app/outputs

# 暴露 Flask 后端端口
EXPOSE 5000

# 默认启动命令（可被 docker-compose 覆盖）
CMD ["python", "-m", "flask", "run", "--host=0.0.0.0"]
