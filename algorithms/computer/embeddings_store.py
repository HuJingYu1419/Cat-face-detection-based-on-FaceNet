# 该脚本用于对指定文件夹下的猫咪图片进行特征提取，并构建一个包含猫咪名称与其对应特征向量的数据库。
# 生成的数据库将被保存为NumPy文件，供后续的猫咪识别任务使用。

import os
import numpy as np
import onnxruntime as ort  # 改用ONNX Runtime
from PIL import Image
from torchvision import transforms
import time

# 1. 定义与训练一致的预处理
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


# 2. 初始化ONNX模型
def init_onnx_model(model_path):
    sess = ort.InferenceSession(model_path)
    print(f"ONNX模型已加载，输入形状: {sess.get_inputs()[0].shape}")
    print(f"输出形状: {sess.get_outputs()[0].shape}")
    return sess


# 3. 生成特征数据库
def build_database(model_path, data_dir, save_path):
    # 初始化
    sess = init_onnx_model(model_path)
    database = {}
    total_cats = len([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
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

            # 预处理（与Inference.py完全一致）
            img_tensor = transform(img).unsqueeze(0)  # [1, 3, 64, 64]
            input_data = img_tensor.numpy()  # 转为NumPy供ONNX使用

            # 提取特征
            embedding = sess.run(["output"], {"input": input_data})[0]
            embedding = embedding / np.linalg.norm(embedding)  # L2归一化

            # 存入数据库
            database[cat_dir] = embedding
            processed += 1

            # 进度显示（每10%或至少每只猫）
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


# 主程序
if __name__ == "__main__":
    # 配置路径（与Inference.py使用相同的ONNX模型）
    MODEL_PATH = "models/facenet_model.onnx"  # 确保已导出ONNX模型
    DATA_DIR = "data/examples/profile"
    SAVE_PATH = "cat_db.npy"

    # 检查路径有效性
    assert os.path.exists(MODEL_PATH), f"ONNX模型不存在: {MODEL_PATH}"
    assert os.path.exists(DATA_DIR), f"数据目录不存在: {DATA_DIR}"

    # 运行构建流程
    build_database(MODEL_PATH, DATA_DIR, SAVE_PATH)