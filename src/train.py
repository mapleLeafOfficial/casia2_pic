import argparse
import logging
import random

import numpy as np
import torch
import torch.optim as optim

from src.data.datasets import create_dataloaders
from src.models.dual_stream import DualStreamNet, DualStreamNetLite
from src.models.baseline import SingleStreamBaseline
from src.models.losses import CompositeLoss
from src.models.trainer import Trainer

try:
    import albumentations as A
except ImportError:  # pragma: no cover - 运行环境缺少依赖时降级
    A = None

try:
    import cv2
except ImportError:  # pragma: no cover - 运行环境缺少依赖时降级
    cv2 = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="训练图像篡改检测模型")
    parser.add_argument("--model", type=str, default="dual_stream_lite", choices=["dual_stream", "dual_stream_lite", "baseline"])
    parser.add_argument("--casia-root", type=str, default="data/raw/CASIA2")
    parser.add_argument("--nist-root", type=str, default="data/raw/NIST16")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--target-size", type=int, default=512)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument("--keep-missing-mask-tampered", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_train_transform(target_size: int):
    if A is None or cv2 is None:
        logger.warning("Albumentations/OpenCV 未安装，训练将不使用数据增强。")
        return None

    return A.Compose([
        A.Resize(target_size, target_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.1,
            rotate_limit=15,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.4,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.15,
            p=0.3,
        ),
    ])


def main():
    args = parse_args()
    set_seed(args.seed)

    transform = None if args.no_augmentation else build_train_transform(args.target_size)

    logger.info("创建数据加载器...")
    loaders = create_dataloaders(
        casia_root=args.casia_root,
        nist_root=args.nist_root,
        target_size=(args.target_size, args.target_size),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        transform=transform,
        seed=args.seed,
        drop_tampered_without_mask=not args.keep_missing_mask_tampered,
    )

    if "train" not in loaders or len(loaders["train"]) == 0:
        logger.error("未能加载训练数据！请检查路径。")
        return
    if "val" not in loaders or len(loaders["val"]) == 0:
        logger.error("未能加载验证数据！请检查路径。")
        return

    logger.info(
        "数据集规模: train=%d, val=%d, 有效batch=%d",
        len(loaders["train"].dataset),
        len(loaders["val"].dataset),
        args.batch_size * max(1, args.accumulation_steps),
    )

    logger.info(f"初始化模型: {args.model}")
    if args.model == "dual_stream":
        model = DualStreamNet(pretrained=True)
    elif args.model == "dual_stream_lite":
        model = DualStreamNetLite(pretrained=True)
    else:
        model = SingleStreamBaseline(pretrained=True)

    criterion = CompositeLoss(bce_weight=0.5, dice_weight=0.5)
    
    if hasattr(model, 'get_param_groups'):
        optimizer = optim.AdamW(
            model.get_param_groups(lr=args.lr),
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=args.device,
        max_epochs=args.epochs,
        patience=args.patience,
        accumulation_steps=args.accumulation_steps,
        use_amp=not args.disable_amp,
    )

    logger.info("开始训练!")
    trainer.train(loaders["train"], loaders["val"])

if __name__ == "__main__":
    main()
