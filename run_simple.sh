#!/bin/bash

# =================================================================
# 图像篡改检测系统 - 一键快速启动脚本 (非 Docker 版)
# =================================================================

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}>>> 正在初始化图像篡改检测系统环境...${NC}"

# 1. 检查 Python 环境
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}错误: 未找到 python3，请先安装 Python。${NC}"
    exit 1
fi

# 2. 创建并激活虚拟环境
VENV_DIR=".venv"
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}>>> 正在创建虚拟环境 $VENV_DIR...${NC}"
    python3 -m venv $VENV_DIR
fi

echo -e "${YELLOW}>>> 正在激活虚拟环境...${NC}"
source $VENV_DIR/bin/activate

# 3. 安装依赖
echo -e "${YELLOW}>>> 正在安装/更新依赖 (requirements.txt)...${NC}"
pip install --upgrade pip
pip install -r requirements.txt

# 4. 检查模型目录
mkdir -p models/checkpoints models/exported

# 5. 启动后端服务
echo -e "${GREEN}>>> 正在启动 Flask 后端服务器 (端口 5000)...${NC}"
export PYTHONPATH=$PYTHONPATH:.
python src/backend/app.py
