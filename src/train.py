import argparse
import logging
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim

from src.data.datasets import create_dataloaders
from src.data.standardizer import ImageStandardizer
from src.models.dual_stream import DualStreamNet, DualStreamNetLite
from src.models.baseline import SingleStreamBaseline
from src.models.losses import CompositeLoss
from src.models.trainer import Trainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="训练图像篡改检测模型")
    parser.add_argument("--model", type=str, default="dual_stream", choices=["dual_stream", "dual_stream_lite", "baseline"])
    parser.add_argument("--casia-root", type=str, default="data/raw/CASIA2")
    parser.add_argument("--nist-root", type=str, default="data/raw/NIST16")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 获取数据增强变换
    standardizer = ImageStandardizer(target_size=(512, 512), normalize=True)
    def transform(image, mask):
        res = standardizer(image, mask)
        # 简单将 image 从 [-x, x] 转换到 [0, 255] 以便下面处理，因为 standardizer 内部做了转换但是数据集要求 numpy
        # 实际情况你可以扩展这里添加 Albumentations
        # 这里为了简化，仅使用 standardizer 的尺寸调整
        return {"image": image, "mask": mask}

    logger.info("创建数据加载器...")
    loaders = create_dataloaders(
        casia_root=args.casia_root,
        nist_root=args.nist_root,
        batch_size=args.batch_size,
    )

    if "train" not in loaders or len(loaders["train"]) == 0:
        logger.error("未能加载训练数据！请检查路径。")
        return

    logger.info(f"初始化模型: {args.model}")
    if args.model == "dual_stream":
        model = DualStreamNet(pretrained=True)
    elif args.model == "dual_stream_lite":
        model = DualStreamNetLite(pretrained=True)
    else:
        model = SingleStreamBaseline(pretrained=True)

    criterion = CompositeLoss(bce_weight=0.5, dice_weight=0.5)
    
    if hasattr(model, 'get_param_groups'):
        optimizer = optim.AdamW(model.get_param_groups(lr=args.lr), weight_decay=1e-4)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=args.device,
        max_epochs=args.epochs,
    )

    logger.info("开始训练!")
    trainer.train(loaders["train"], loaders["val"])

if __name__ == "__main__":
    main()
