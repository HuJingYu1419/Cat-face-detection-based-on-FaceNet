import os
import random
import torch
from torch.utils.data import Dataset, Subset
from PIL import Image
import torchvision.transforms as T


class CatTripletDataset(Dataset):
    def __init__(self, root_dir, img_size=64):
        self.root_dir = os.path.join(root_dir, 'cat_faces')
        self.cat_dirs = sorted([d for d in os.listdir(self.root_dir)
                                if os.path.isdir(os.path.join(self.root_dir, d))])
        self.transform = T.Compose([
            T.Resize(img_size),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        # 预加载文件路径加速访问
        self.samples = []
        for cat_idx, cat_dir in enumerate(self.cat_dirs):
            cat_path = os.path.join(self.root_dir, cat_dir)
            img_names = [f for f in os.listdir(cat_path) if f.endswith(('.jpg', '.png'))]
            self.samples.extend([(os.path.join(cat_path, f), cat_idx) for f in img_names])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """返回单张图像+标签（供在线采样使用）"""
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        return self.transform(img), label

    def get_triplet(self, idx):
        """在线生成Triplet的辅助方法"""
        _, anchor_label = self.samples[idx]

        # 查找同类别样本作为positive
        positive_indices = [i for i, (_, label) in enumerate(self.samples)
                            if label == anchor_label and i != idx]
        positive_idx = random.choice(positive_indices)

        # 查找不同类别样本作为negative
        negative_indices = [i for i, (_, label) in enumerate(self.samples)
                            if label != anchor_label]
        negative_idx = random.choice(negative_indices)

        anchor = self.__getitem__(idx)[0]
        positive = self.__getitem__(positive_idx)[0]
        negative = self.__getitem__(negative_idx)[0]

        return anchor, positive, negative


def split_dataset(dataset, train_ratio=0.8, seed=42):
    """按猫ID划分数据集（保持每只猫样本完整）"""
    random.seed(seed)

    # 获取所有唯一的猫ID（对应文件夹名）
    cat_ids = sorted(list({label for _, label in dataset.samples}))
    random.shuffle(cat_ids)  # 随机打乱

    # 按比例划分猫ID
    split_idx = int(len(cat_ids) * train_ratio)
    train_cats = set(cat_ids[:split_idx])
    val_cats = set(cat_ids[split_idx:])

    # 根据猫ID筛选样本索引
    train_indices = [i for i, (_, label) in enumerate(dataset.samples)
                     if label in train_cats]
    val_indices = [i for i, (_, label) in enumerate(dataset.samples)
                   if label in val_cats]

    return Subset(dataset, train_indices), Subset(dataset, val_indices)