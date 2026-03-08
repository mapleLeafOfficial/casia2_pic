#!/usr/bin/env bash
set -euo pipefail

# Kaggle notebook usage:
# !bash /kaggle/working/your-project/kaggle_train.sh

PROJECT_DIR="${PROJECT_DIR:-/kaggle/working/your-project}"
CASIA_ROOT="${CASIA_ROOT:-/kaggle/input/casia2/CASIA2}"
NIST_ROOT="${NIST_ROOT:-/kaggle/input/nist16/NIST16}"
MODEL="${MODEL:-dual_stream_lite}"
BATCH_SIZE="${BATCH_SIZE:-4}"
EPOCHS="${EPOCHS:-1}"

echo "[1/5] Project dir: ${PROJECT_DIR}"
cd "${PROJECT_DIR}"

echo "[2/5] Python and CUDA check"
python -c "import sys, torch; print(sys.version); print('torch=', torch.__version__); print('cuda_available=', torch.cuda.is_available()); print('cuda=', torch.version.cuda)"

echo "[3/5] Install project dependencies (keep Kaggle's preinstalled torch)"
grep -Evi '^(torch|torchvision|torchaudio)([<>=!~].*)?$' requirements.txt > /tmp/requirements-kaggle.txt
python -m pip install -U pip
python -m pip install -r /tmp/requirements-kaggle.txt

echo "[4/5] Run GPU smoke test"
python check_gpu.py || true

echo "[5/5] Start training"
export PYTHONPATH="${PROJECT_DIR}"
python src/train.py \
  --model "${MODEL}" \
  --casia-root "${CASIA_ROOT}" \
  --nist-root "${NIST_ROOT}" \
  --batch-size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --device cuda

