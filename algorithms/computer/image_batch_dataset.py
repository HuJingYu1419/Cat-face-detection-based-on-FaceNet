# 该脚本用于批量处理图像，应用随机增强方法，并将处理后的图像按照指定的Triple loss数据集组织方式保存到指定目录结构中。

import os
import cv2
import numpy as np
import random
from PIL import Image, ImageEnhance
import re


def horizontal_flip(image):
    """水平翻转图像"""
    return image.transpose(Image.FLIP_LEFT_RIGHT)


def random_brightness_contrast(image):
    """
    随机调整亮度和对比度
    亮度调整范围: 0.8-1.2
    对比度调整范围: 0.9-1.1
    """
    # 随机亮度调整
    enhancer = ImageEnhance.Brightness(image)
    brightness_factor = random.uniform(0.8, 1.2)
    image = enhancer.enhance(brightness_factor)

    # 随机对比度调整
    enhancer = ImageEnhance.Contrast(image)
    contrast_factor = random.uniform(0.9, 1.1)
    image = enhancer.enhance(contrast_factor)

    return image


def random_rotation(image):
    """随机旋转图像，角度范围: -10到10度"""
    angle = random.uniform(-10, 10)
    return image.rotate(angle, resample=Image.BILINEAR, fillcolor=(128, 128, 128))


def extract_cat_number(filename):
    """从文件名中提取编号"""
    match1 = re.search(r'(\d{1})', filename)
    match2=re.search(r'(\d{2})', filename)
    match3 = re.search(r'(\d{3})', filename)
    match4=re.search(r'(\d{4})', filename)
    match5=re.search(r'(\d{5})', filename)
    if match5:
        return match5.group(1)
    elif match4:
        return match4.group(1)
    elif match3:
        return match3.group(1)
    elif match2:
        return match2.group(1)
    elif match1:
        return match1.group(1)
    else:
         return "None"  # 表示无编号


def process_image(input_path, output_dir, image_name):
    """
    处理单张图像并按照项目要求的结构保存
    """
    # 从文件名中提取三位数编号
    cat_number = extract_cat_number(image_name)
    cat_dir = os.path.join(output_dir, f"cat_{cat_number}")

    # 确保猫的目录存在
    os.makedirs(cat_dir, exist_ok=True)

    # 打开原始图像
    original_image = Image.open(input_path)

    # 随机选择要应用的增强方法(至少选1种，最多3种)
    augmentations = [
        horizontal_flip,
        random_brightness_contrast,
        random_rotation
    ]
    selected_augmentations = random.sample(augmentations, k=random.randint(1, 3))

    # 保存原始图像(编号1)
    original_output_name = f"{cat_number}_1.jpg"
    original_output_path = os.path.join(cat_dir, original_output_name)
    original_image.save(original_output_path)

    # 应用选定的增强方法并保存处理后的图像(编号2)
    processed_image = original_image.copy()
    for aug in selected_augmentations:
        processed_image = aug(processed_image)

    processed_output_name = f"{cat_number}_2.jpg"
    processed_output_path = os.path.join(cat_dir, processed_output_name)
    processed_image.save(processed_output_path)

    print(f"Processed: {image_name} -> {cat_number}_1.jpg and {cat_number}_2.jpg in {cat_dir}")


def batch_process_images(input_dir, output_dir):
    """批量处理目录中的所有图像"""
    # 创建cat_faces目录
    cat_faces_dir = os.path.join(output_dir, "cat_faces")
    os.makedirs(cat_faces_dir, exist_ok=True)

    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                input_path = os.path.join(root, file)
                process_image(input_path, cat_faces_dir, file)


if __name__ == "__main__":
    # 设置输入输出目录
    input_directory = r""  # 替换为您的输入目录
    output_directory = r""  # 替换为您想要的输出目录

    # 处理所有图像
    batch_process_images(input_directory, output_directory)