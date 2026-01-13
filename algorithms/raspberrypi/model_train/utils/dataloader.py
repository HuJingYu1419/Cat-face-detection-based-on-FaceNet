import os
import random
import torch
from torch.utils.data import Dataset, Subset
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import to_grayscale
from typing import Tuple
import numpy as np


class CatTripletDataset(Dataset):
    def __init__(self, root_dir: str, img_size: int = 48):
        """
        轻量化猫脸三元组数据集
        :param root_dir: 数据根目录
        :param img_size: 统一调整的图像尺寸 (默认48x48)
        """
        self.root_dir = os.path.join(root_dir, 'cat_faces')
        self.img_size = img_size

        # 使用更快的扫描方式收集样本
        self.samples = []
        self.cat_indices = {}  # 按类别索引的样本字典

        for cat_dir in sorted(os.listdir(self.root_dir)):
            cat_path = os.path.join(self.root_dir, cat_dir)
            if not os.path.isdir(cat_path):
                continue

            cat_idx = len(self.cat_indices)  # 自动分配类别ID
            self.cat_indices[cat_idx] = []

            for img_name in sorted(os.listdir(cat_path)):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(cat_path, img_name)
                    self.samples.append((img_path, cat_idx))
                    self.cat_indices[cat_idx].append(len(self.samples) - 1)

        # 优化后的数据增强流程 (灰度+轻量增强)
        self.transform = T.Compose([
            T.Lambda(lambda x: to_grayscale(x, num_output_channels=1)),  # 转为单通道灰度
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(p=0.3),  # 轻量数据增强
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5])  # 单通道归一化
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        获取单张图像和标签
        :return: (图像张量, 类别ID)
        """
        img_path, label = self.samples[idx]

        # 使用更快的图像加载方式
        try:
            with Image.open(img_path) as img:
                img = img.convert('RGB')  # 先转为RGB确保兼容性
                return self.transform(img), label
        except Exception as e:
            print(f"加载图像失败: {img_path}, 错误: {e}")
            # 返回空白图像作为占位符
            dummy_img = torch.zeros(1, self.img_size, self.img_size)
            return dummy_img, 0

    def get_triplet(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        在线生成三元组
        :return: (anchor, positive, negative)
        """
        _, anchor_label = self.samples[idx]

        # 优化查找速度 - 使用预存的类别索引
        positive_indices = self.cat_indices[anchor_label]
        positive_idx = random.choice([i for i in positive_indices if i != idx])

        # 从其他类别随机选择
        negative_label = random.choice([l for l in self.cat_indices.keys() if l != anchor_label])
        negative_idx = random.choice(self.cat_indices[negative_label])

        return self.__getitem__(idx)[0], self.__getitem__(positive_idx)[0], self.__getitem__(negative_idx)[0]

    def verify_dataset(self):
        """快速验证数据集完整性"""
        valid_samples = []
        for idx, (img_path, label) in enumerate(self.samples):
            if os.path.exists(img_path):
                valid_samples.append((img_path, label))
            else:
                print(f"警告: 缺失图像 {img_path}")
        self.samples = valid_samples
        self._rebuild_indices()

    def _rebuild_indices(self):
        """重建类别索引"""
        self.cat_indices = {}
        for idx, (_, label) in enumerate(self.samples):
            if label not in self.cat_indices:
                self.cat_indices[label] = []
            self.cat_indices[label].append(idx)


def split_dataset(dataset: Dataset, train_ratio: float = 0.8, seed: int = 42) -> Tuple[Subset, Subset]:
    """
    按类别划分数据集 (保持类别分布)
    :param dataset: CatTripletDataset实例
    :param train_ratio: 训练集比例
    :param seed: 随机种子
    :return: (训练集, 验证集)
    """
    random.seed(seed)

    # 按类别划分
    train_indices = []
    val_indices = []

    for cat_idx, indices in dataset.cat_indices.items():
        random.shuffle(indices)
        split_pos = int(len(indices) * train_ratio)
        train_indices.extend(indices[:split_pos])
        val_indices.extend(indices[split_pos:])

    random.shuffle(train_indices)
    random.shuffle(val_indices)

    return Subset(dataset, train_indices), Subset(dataset, val_indices)