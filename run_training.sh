#!/bin/bash
# 激活虚拟环境并运行训练脚本
source .venv/bin/activate
export PYTHONPATH=.
python src/train.py
