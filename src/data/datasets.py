"""
CASIA v2.0 和 NIST16 数据集加载脚本

支持的数据集：
- CASIA v2.0：包含拼接 (Splicing) 和复制移动 (Copy-Move) 篡改图像
- NIST16 (Nimble Challenge 2016)：包含多类型篡改图像及对应的参考掩码

数据集目录结构预期：
  data/raw/
  ├── CASIA2/
  │   ├── Au/          # 真实图像 (Authentic)
  │   ├── Tp/          # 篡改图像 (Tampered)
  │   └── mask/        # 篡改掩码 (如果可用)
  └── NIST16/
      ├── probe/       # 待检测图像
      ├── world/       # 原始参考图像
      └── mask/        # Ground-truth 掩码
"""

import os
import glob
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image


class CASIAv2Dataset(Dataset):
    """
    CASIA v2.0 数据集加载器

    CASIA v2.0 是图像篡改检测领域最广泛使用的基准数据集之一，
    包含 7,491 张真实图像和 5,123 张篡改图像（拼接 + 复制移动）。

    参数:
        root_dir: 数据集根目录路径 (例如 data/raw/CASIA2)
        transform: 可选的图像变换函数
        target_size: 目标输出尺寸 (H, W)，默认 (512, 512)
        split: 数据划分 ('train', 'val', 'test')
        train_ratio: 训练集比例，默认 0.7
        val_ratio: 验证集比例，默认 0.15
        seed: 随机种子，确保划分可复现
    """

    def __init__(
        self,
        root_dir: str,
        transform=None,
        target_size: Tuple[int, int] = (512, 512),
        split: str = "train",
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        seed: int = 42,
        drop_tampered_without_mask: bool = True,
    ):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.target_size = target_size
        self.split = split
        self.drop_tampered_without_mask = drop_tampered_without_mask

        # 收集图像路径和标签
        self.samples: List[Dict] = []
        self._load_samples()

        # 按照固定种子划分数据集
        self._split_dataset(train_ratio, val_ratio, seed)

    def _load_samples(self):
        """扫描数据集目录，收集所有图像路径"""
        au_dir = self.root_dir / "Au"  # 真实图像目录
        tp_dir = self.root_dir / "Tp"  # 篡改图像目录
        mask_dir = self.root_dir / "mask"  # 掩码目录（可选）
        skipped_tampered = 0

        # 支持的图像格式
        img_extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif")

        # 加载真实图像 (标签 = 0)
        if au_dir.exists():
            for ext in img_extensions:
                for img_path in sorted(au_dir.glob(ext)):
                    self.samples.append({
                        "image_path": str(img_path),
                        "mask_path": None,  # 真实图像没有掩码
                        "label": 0,  # 0 = 真实
                        "tamper_type": "authentic",
                    })

        # 加载篡改图像 (标签 = 1)
        if tp_dir.exists():
            for ext in img_extensions:
                for img_path in sorted(tp_dir.glob(ext)):
                    # 尝试匹配对应的掩码文件
                    mask_path = self._find_mask(img_path, mask_dir)
                    if (
                        self.drop_tampered_without_mask
                        and mask_dir.exists()
                        and mask_path is None
                    ):
                        skipped_tampered += 1
                        continue

                    # 根据文件名判断篡改类型
                    tamper_type = self._infer_tamper_type(img_path.name)

                    self.samples.append({
                        "image_path": str(img_path),
                        "mask_path": mask_path,
                        "label": 1,  # 1 = 篡改
                        "tamper_type": tamper_type,
                    })
        if skipped_tampered > 0:
            print(
                f"[CASIA] 跳过 {skipped_tampered} 张无掩码篡改样本 "
                f"(split={self.split}, root={self.root_dir})"
            )

    def _find_mask(self, img_path: Path, mask_dir: Path) -> Optional[str]:
        """
        尝试为篡改图像找到对应的 Ground-truth 掩码

        CASIA v2.0 的掩码命名规则可能不统一，此处尝试多种匹配策略。
        """
        if not mask_dir.exists():
            return None

        stem = img_path.stem
        # 常见的掩码命名模式
        candidates = [
            mask_dir / f"{stem}.tif",
            mask_dir / f"{stem}.jpg",
            mask_dir / f"{stem}.jpeg",
            mask_dir / f"{stem}.png",
            mask_dir / f"{stem}.bmp",
            mask_dir / f"{stem}_mask.tif",
            mask_dir / f"{stem}_mask.png",
            mask_dir / f"{stem}_gt.png",
            mask_dir / f"{stem}_gt.tif",
        ]

        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        return None

    def _infer_tamper_type(self, filename: str) -> str:
        """
        根据 CASIA v2.0 的文件命名规则推断篡改类型

        规则：
        - 文件名包含 'Sp' → Splicing（拼接）
        - 文件名包含 'CM' 或 'CMFD' → Copy-Move（复制移动）
        """
        filename_lower = filename.lower()
        if "sp" in filename_lower:
            return "splicing"
        elif "cm" in filename_lower:
            return "copy_move"
        else:
            return "unknown"

    def _split_dataset(self, train_ratio: float, val_ratio: float, seed: int):
        """固定种子划分训练/验证/测试集"""
        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(self.samples))

        n_train = int(len(indices) * train_ratio)
        n_val = int(len(indices) * val_ratio)

        if self.split == "train":
            selected = indices[:n_train]
        elif self.split == "val":
            selected = indices[n_train:n_train + n_val]
        elif self.split == "test":
            selected = indices[n_train + n_val:]
        else:
            raise ValueError(f"无效的 split 参数: {self.split}，应为 'train'/'val'/'test'")

        self.samples = [self.samples[i] for i in selected]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        """
        返回单个样本

        返回字典包含:
            - image: (3, H, W) 归一化后的 RGB 图像张量
            - mask: (1, H, W) 二值掩码张量 (0 = 真实, 1 = 篡改)
            - label: 整图标签 (0 或 1)
            - tamper_type: 篡改类型字符串
            - image_path: 原始图像路径
        """
        sample = self.samples[idx]

        # 读取图像 (BGR → RGB)
        image = cv2.imread(sample["image_path"])
        if image is None:
            raise FileNotFoundError(f"无法读取图像: {sample['image_path']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 读取掩码（如果存在）
        if sample["mask_path"] and os.path.exists(sample["mask_path"]):
            mask = cv2.imread(sample["mask_path"], cv2.IMREAD_GRAYSCALE)
            # 二值化处理：确保掩码只有 0 和 1
            mask = (mask > 127).astype(np.uint8)
        else:
            if sample["label"] == 0:
                # 真实图像：全零掩码
                mask = np.zeros(
                    (image.shape[0], image.shape[1]), dtype=np.uint8
                )
            else:
                # 篡改图像但无掩码：创建全 1 掩码（表示整图篡改，粗粒度标注）
                mask = np.ones(
                    (image.shape[0], image.shape[1]), dtype=np.uint8
                )

        # 调整尺寸
        image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)

        # 应用数据增强（如果有）
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # 转换为 PyTorch 张量
        # 图像：(H, W, 3) → (3, H, W)，归一化到 [0, 1]
        image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        # 掩码：(H, W) → (1, H, W)
        mask = torch.from_numpy(mask).unsqueeze(0).float()

        return {
            "image": image,
            "mask": mask,
            "label": torch.tensor(sample["label"], dtype=torch.long),
            "tamper_type": sample["tamper_type"],
            "image_path": sample["image_path"],
        }


class NIST16Dataset(Dataset):
    """
    NIST16 (Nimble Challenge 2016) 数据集加载器

    NIST16 包含多类型篡改操作（拼接、复制移动、擦除），
    并提供精确的像素级 Ground-truth 参考掩码。

    参数:
        root_dir: 数据集根目录路径 (例如 data/raw/NIST16)
        transform: 可选的图像变换函数
        target_size: 目标输出尺寸 (H, W)，默认 (512, 512)
        split: 数据划分 ('train', 'val', 'test')
        train_ratio: 训练集比例，默认 0.7
        val_ratio: 验证集比例，默认 0.15
        seed: 随机种子
    """

    def __init__(
        self,
        root_dir: str,
        transform=None,
        target_size: Tuple[int, int] = (512, 512),
        split: str = "train",
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        seed: int = 42,
    ):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.target_size = target_size
        self.split = split

        self.samples: List[Dict] = []
        self._load_samples()
        self._split_dataset(train_ratio, val_ratio, seed)

    def _load_samples(self):
        """扫描 NIST16 数据集目录"""
        probe_dir = self.root_dir / "probe"  # 待检测图像
        mask_dir = self.root_dir / "mask"    # Ground-truth 掩码
        world_dir = self.root_dir / "world"  # 原始参考图像

        img_extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif")

        # 加载篡改图像 (来自 probe 目录)
        if probe_dir.exists():
            for ext in img_extensions:
                for img_path in sorted(probe_dir.glob(ext)):
                    mask_path = self._find_nist_mask(img_path, mask_dir)
                    self.samples.append({
                        "image_path": str(img_path),
                        "mask_path": mask_path,
                        "label": 1,  # probe 中都是篡改图像
                        "tamper_type": "manipulated",
                    })

        # 加载真实图像 (来自 world 目录，如果存在)
        if world_dir.exists():
            for ext in img_extensions:
                for img_path in sorted(world_dir.glob(ext)):
                    self.samples.append({
                        "image_path": str(img_path),
                        "mask_path": None,
                        "label": 0,
                        "tamper_type": "authentic",
                    })

    def _find_nist_mask(self, img_path: Path, mask_dir: Path) -> Optional[str]:
        """
        查找 NIST16 对应的掩码文件

        NIST16 掩码通常与 probe 图像同名或有 '_mask' 后缀。
        """
        if not mask_dir.exists():
            return None

        stem = img_path.stem
        candidates = [
            mask_dir / f"{stem}.tif",
            mask_dir / f"{stem}.png",
            mask_dir / f"{stem}.bmp",
            mask_dir / f"{stem}_mask.tif",
            mask_dir / f"{stem}_mask.png",
            mask_dir / f"{stem}_mask.bmp",
        ]

        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        return None

    def _split_dataset(self, train_ratio: float, val_ratio: float, seed: int):
        """固定种子划分数据集"""
        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(self.samples))

        n_train = int(len(indices) * train_ratio)
        n_val = int(len(indices) * val_ratio)

        if self.split == "train":
            selected = indices[:n_train]
        elif self.split == "val":
            selected = indices[n_train:n_train + n_val]
        elif self.split == "test":
            selected = indices[n_train + n_val:]
        else:
            raise ValueError(f"无效的 split 参数: {self.split}")

        self.samples = [self.samples[i] for i in selected]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        """
        返回单个样本（格式与 CASIAv2Dataset 兼容）
        """
        sample = self.samples[idx]

        # 读取图像
        image = cv2.imread(sample["image_path"])
        if image is None:
            raise FileNotFoundError(f"无法读取图像: {sample['image_path']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 读取掩码
        if sample["mask_path"] and os.path.exists(sample["mask_path"]):
            mask = cv2.imread(sample["mask_path"], cv2.IMREAD_GRAYSCALE)
            mask = (mask > 127).astype(np.uint8)
        else:
            if sample["label"] == 0:
                mask = np.zeros(
                    (image.shape[0], image.shape[1]), dtype=np.uint8
                )
            else:
                mask = np.ones(
                    (image.shape[0], image.shape[1]), dtype=np.uint8
                )

        # 调整尺寸
        image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)

        # 数据增强
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # 转换为张量
        image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        mask = torch.from_numpy(mask).unsqueeze(0).float()

        return {
            "image": image,
            "mask": mask,
            "label": torch.tensor(sample["label"], dtype=torch.long),
            "tamper_type": sample["tamper_type"],
            "image_path": sample["image_path"],
        }


class CombinedTamperingDataset(Dataset):
    """
    组合数据集：将 CASIA v2.0 和 NIST16 合并为一个统一数据集

    用于跨数据集训练和评估，提升模型泛化能力。

    参数:
        datasets: 要合并的数据集列表
    """

    def __init__(self, datasets: List[Dataset]):
        super().__init__()
        self.datasets = datasets
        self.cumulative_sizes = []
        cumsum = 0
        for ds in datasets:
            cumsum += len(ds)
            self.cumulative_sizes.append(cumsum)

    def __len__(self) -> int:
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0

    def __getitem__(self, idx: int) -> Dict:
        # 定位样本所在的子数据集
        for i, size in enumerate(self.cumulative_sizes):
            if idx < size:
                if i > 0:
                    idx -= self.cumulative_sizes[i - 1]
                return self.datasets[i][idx]
        raise IndexError(f"索引 {idx} 超出范围")


def create_dataloaders(
    casia_root: Optional[str] = None,
    nist_root: Optional[str] = None,
    target_size: Tuple[int, int] = (512, 512),
    batch_size: int = 8,
    num_workers: int = 4,
    transform=None,
    seed: int = 42,
    drop_tampered_without_mask: bool = True,
) -> Dict[str, DataLoader]:
    """
    创建训练、验证和测试数据加载器的便捷函数

    参数:
        casia_root: CASIA v2.0 数据集根目录
        nist_root: NIST16 数据集根目录
        target_size: 图像目标尺寸
        batch_size: 批量大小
        num_workers: 数据加载工作线程数
        transform: 数据增强变换（仅用于训练集）
        seed: 随机种子

    返回:
        包含 'train', 'val', 'test' 三个 DataLoader 的字典
    """
    dataloaders = {}

    for split in ["train", "val", "test"]:
        datasets = []

        # 训练集使用数据增强，验证和测试集不使用
        current_transform = transform if split == "train" else None

        if casia_root and os.path.exists(casia_root):
            datasets.append(
                CASIAv2Dataset(
                    root_dir=casia_root,
                    transform=current_transform,
                    target_size=target_size,
                    split=split,
                    seed=seed,
                    drop_tampered_without_mask=drop_tampered_without_mask,
                )
            )

        if nist_root and os.path.exists(nist_root):
            datasets.append(
                NIST16Dataset(
                    root_dir=nist_root,
                    transform=current_transform,
                    target_size=target_size,
                    split=split,
                    seed=seed,
                )
            )

        if not datasets:
            print(f"警告：没有找到数据集，{split} DataLoader 将为空")
            continue

        # 合并数据集
        if len(datasets) == 1:
            combined = datasets[0]
        else:
            combined = CombinedTamperingDataset(datasets)

        dataloaders[split] = DataLoader(
            combined,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(num_workers > 0),
            drop_last=(split == "train"),
        )

    return dataloaders


if __name__ == "__main__":
    # 使用示例
    print("=" * 60)
    print("图像篡改检测数据集加载器 - 使用示例")
    print("=" * 60)

    # 示例：创建 CASIA v2.0 数据集
    # dataset = CASIAv2Dataset(
    #     root_dir="data/raw/CASIA2",
    #     target_size=(512, 512),
    #     split="train",
    # )
    # print(f"CASIA v2.0 训练集大小: {len(dataset)}")

    # 示例：一键创建所有数据加载器
    # loaders = create_dataloaders(
    #     casia_root="data/raw/CASIA2",
    #     nist_root="data/raw/NIST16",
    #     batch_size=8,
    # )
    # for name, loader in loaders.items():
    #     print(f"{name}: {len(loader.dataset)} 样本, {len(loader)} 批次")

    print("\n请将数据集放置在以下目录结构中：")
    print("  data/raw/CASIA2/Au/   - 真实图像")
    print("  data/raw/CASIA2/Tp/   - 篡改图像")
    print("  data/raw/CASIA2/mask/ - 掩码（可选）")
    print("  data/raw/NIST16/probe/ - 待检测图像")
    print("  data/raw/NIST16/world/ - 真实图像")
    print("  data/raw/NIST16/mask/  - Ground-truth 掩码")
