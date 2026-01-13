import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import time
from model_train.models.facenet_pi import LiteFaceNet  # 需要从原项目导入模型定义


class QuantizedEmbeddingGenerator:
    def __init__(self, model_path):
        # 初始化量化模型
        self.model = self._load_quantized_model(model_path)
        self.model.eval()

        # 与训练一致的预处理（保持与train.py相同）
        self.transform = transforms.Compose([
            transforms.Resize(48),  # 量化模型输入尺寸为48x48
            transforms.Grayscale(),  # 量化模型使用单通道输入
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # 单通道归一化
        ])

    def _load_quantized_model(self, model_path):
        """加载量化模型"""
        checkpoint = torch.load(model_path, map_location='cpu')

        # 初始化原始模型结构
        model = LiteFaceNet(embedding_size=32)
        model.load_state_dict(checkpoint['state_dict'], strict=False)

        # 应用动态量化
        return torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear},
            dtype=torch.qint8
        )

    def get_embedding(self, img):
        """获取量化模型的embedding"""
        img_tensor = self.transform(img).unsqueeze(0)  # [1, 1, 48, 48]
        with torch.no_grad():
            embedding = self.model(img_tensor).numpy()[0]
        return embedding / np.linalg.norm(embedding)  # L2归一化


def build_database(model_path, data_dir, save_path):
    # 初始化
    generator = QuantizedEmbeddingGenerator(model_path)
    database = {}
    total_cats = len([d for d in os.listdir(data_dir)
                      if os.path.isdir(os.path.join(data_dir, d))])
    processed = 0

    print(f"\n开始处理 {total_cats} 只猫的特征提取...")
    start_time = time.time()

    for cat_dir in os.listdir(data_dir):
        cat_path = os.path.join(data_dir, cat_dir)
        if not os.path.isdir(cat_path):
            continue

        try:
            # 获取该猫的第一张图片
            img_path = os.path.join(cat_path, os.listdir(cat_path)[0])
            img = Image.open(img_path).convert('RGB')

            # 提取特征
            embedding = generator.get_embedding(img)
            database[cat_dir] = embedding
            processed += 1

            # 进度显示
            if processed % max(1, total_cats // 10) == 0 or processed == total_cats:
                elapsed = time.time() - start_time
                print(f"进度: {processed}/{total_cats} | 耗时: {elapsed:.1f}s | 当前处理: {cat_dir}")

        except Exception as e:
            print(f"处理失败 {cat_dir}: {str(e)}")
            continue

    # 保存数据库
    np.save(save_path, database)
    print(f"\n数据库已保存至 {save_path}")
    print(f"总样本数: {len(database)} | 特征维度: {next(iter(database.values())).shape}")


if __name__ == "__main__":
    # 配置路径
    QUANT_MODEL_PATH = "checkpoints/quantized_model_lite.pth"  # 量化模型
    DATA_DIR = "test_data/cat_faces/"
    SAVE_PATH = "cat_db_quant.npy"  # 新数据库文件名

    # 检查路径
    assert os.path.exists(QUANT_MODEL_PATH), f"模型不存在: {QUANT_MODEL_PATH}"
    assert os.path.exists(DATA_DIR), f"数据目录不存在: {DATA_DIR}"

    # 运行构建流程
    build_database(QUANT_MODEL_PATH, DATA_DIR, SAVE_PATH)