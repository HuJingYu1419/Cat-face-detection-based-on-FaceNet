# clean_data.py
import os
import cv2
from pathlib import Path


def clean_dataset():
    """清理损坏的图像文件"""
    image_dir = 'data/images/val'
    label_dir = 'data/labels/val'

    print("开始清理数据集...")

    # 支持的图像格式
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp'}

    removed_count = 0

    for img_file in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_file)
        label_file = Path(img_file).stem + '.txt'
        label_path = os.path.join(label_dir, label_file)

        # 检查文件格式
        file_ext = Path(img_file).suffix.lower()
        if file_ext not in supported_formats:
            print(f"移除不支持的格式: {img_file}")
            os.remove(img_path)
            if os.path.exists(label_path):
                os.remove(label_path)
            removed_count += 1
            continue

        # 检查图像是否可读
        try:
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError("无法读取图像")
            # 简单验证图像内容
            if img.size == 0:
                raise ValueError("空图像")
        except Exception as e:
            print(f"移除损坏图像: {img_file} - {e}")
            os.remove(img_path)
            if os.path.exists(label_path):
                os.remove(label_path)
            removed_count += 1
            continue

    print(f"清理完成，移除了 {removed_count} 个文件")
    return removed_count


if __name__ == "__main__":
    clean_dataset()