#!/usr/bin/env python3

import argparse
import os
import sys
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image
from torchvision import transforms
import torch
from inference.raspberrypi.Pi_detector import PiCatMonitorLite


def main():
    # 创建ArgumentParser对象
    formatter = lambda prog: argparse.HelpFormatter(prog, max_help_position=36)
    desc = '''
    树莓派猫脸识别监控系统 - 专为树莓派优化的实时监控
    使用YOLO检测猫脸并用量化FaceNet模型进行识别比对
    '''

    parser = argparse.ArgumentParser(description=desc, formatter_class=formatter)

    # 模型和数据库参数
    parser.add_argument('-m', '--model',
                        help='量化FaceNet模型路径',
                        default='models/facenet_lite.pth',
                        metavar='<file>')

    parser.add_argument('-d', '--database-folder',
                        help='包含猫脸图片的文件夹路径，用于生成特征数据库',
                        metavar='<folder>')

    parser.add_argument('-y', '--yolo-model',
                        help='YOLO模型路径',
                        default='models/yolo_model.pt',
                        metavar='<file>')

    parser.add_argument('-c', '--camera',
                        help='摄像头设备ID (树莓派通常为0)',
                        type=int,
                        default=0,
                        metavar='<id>')

    parser.add_argument('-t', '--threshold',
                        help='相似度阈值 (默认: 0.4)',
                        type=float,
                        default=0.4,
                        metavar='<float>')

    parser.add_argument('-di', '--detection-interval',
                        help='YOLO检测间隔帧数 (默认: 5)',
                        type=int,
                        default=5,
                        metavar='<frames>')

    parser.add_argument('-ds', '--display-scale',
                        help='显示缩放比例 (默认: 0.5)',
                        type=float,
                        default=0.5,
                        metavar='<scale>')

    parser.add_argument('--cpu-governor',
                        help='CPU性能模式 (performance/ondemand)',
                        choices=['performance', 'ondemand'],
                        default='performance',
                        metavar='<mode>')

    # 解析参数
    args = parser.parse_args()

    # 验证文件存在性
    if not os.path.exists(args.model):
        print(f"错误: 模型文件不存在: {args.model}")
        sys.exit(1)

    if not os.path.exists(args.yolo_model):
        print(f"错误: YOLO模型文件不存在: {args.yolo_model}")
        print("请下载YOLOv8n模型或指定正确的路径")
        sys.exit(1)

    # 处理特征数据库
    db_path, is_temp_db = handle_database_folder(args.database_images, args.model)

    print("=" * 60)
    print("树莓派猫脸识别监控系统")
    print("=" * 60)
    print(f"模型路径: {args.model}")
    print(f"数据库路径: {db_path}")
    print(f"YOLO模型: {args.yolo_model}")
    print(f"摄像头设备: {args.camera}")
    print(f"相似度阈值: {args.threshold}")
    print(f"检测间隔: {args.detection_interval} 帧")
    print(f"显示缩放: {args.display_scale}")
    print(f"CPU模式: {args.cpu_governor}")
    print("=" * 60)

    try:
        # 初始化树莓派监控器
        monitor = PiCatMonitorLite(
            model_path=args.model,
            db_path=db_path,
            yolo_model_path=args.yolo_model
        )

        # 设置监控参数
        monitor.detection_interval = args.detection_interval
        monitor.display_scale = args.display_scale

        # 设置CPU性能模式
        if args.cpu_governor == 'performance':
            os.system("echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor")
            print("已启用CPU性能模式")

        print("\n启动监控中...")
        print("按 'q' 键退出监控")
        print("-" * 40)

        # 启动监控
        monitor.start_monitoring()

    except KeyboardInterrupt:
        print("\n监控已由用户中断")
    except Exception as e:
        print(f"监控过程中出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理临时数据库文件
        if is_temp_db and os.path.exists(db_path):
            os.remove(db_path)
            print(f"已删除临时数据库文件: {db_path}")

        # 恢复CPU模式
        if args.cpu_governor == 'performance':
            os.system("echo ondemand | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor")
            print("已恢复CPU省电模式")

        print("监控系统已关闭")


def handle_database_folder(folder_path, model_path):
    """
    处理特征数据库图片，生成临时的量化npy文件
    返回: (数据库路径, 是否为临时文件)
    """
    default_db_path = 'data/examples/embeddings/cat_db_quant.npy'

    # 获取文件夹中的所有图片文件
    image_paths = get_image_files_from_folder(folder_path)

    if not image_paths:
        print(f"错误: 文件夹中没有找到图片文件: {folder_path}")
        print(f"使用默认特征数据库: {default_db_path}")
        return default_db_path, False

    # 检查默认数据库是否存在
    if not os.path.exists(default_db_path):
        print(f"警告: 默认数据库不存在 {default_db_path}")

    # 创建临时数据库
    print(f"根据提供的 {len(image_paths)} 张图片生成临时量化特征数据库...")

    # 创建临时npy文件
    temp_db_path = tempfile.mktemp(suffix='.npy', prefix='temp_cat_db_quant_')

    try:
        # 构建量化数据库
        build_quantized_database_from_images(model_path, image_paths, temp_db_path)

        print(f"临时量化特征数据库已生成: {temp_db_path}")
        return temp_db_path, True

    except Exception as e:
        print(f"生成临时数据库时出错: {e}")
        print("使用默认数据库")
        return default_db_path, False


def get_image_files_from_folder(folder_path):
    """
    从文件夹中获取所有图片文件
    """
    folder = Path(folder_path)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

    image_paths = []
    for ext in image_extensions:
        image_paths.extend(folder.glob(f'*{ext}'))
        image_paths.extend(folder.glob(f'*{ext.upper()}'))

    # 转换为字符串路径并排序
    image_paths = sorted([str(path) for path in image_paths])

    print(f"在文件夹中找到 {len(image_paths)} 张图片:")
    for img_path in image_paths:
        print(f"  - {os.path.basename(img_path)}")

    return image_paths

def build_quantized_database_from_images(model_path, image_paths, output_path):
    """
    直接从图片列表构建量化特征数据库
    """
    # 定义量化模型的预处理转换
    transform = transforms.Compose([
        transforms.Resize(48),  # 量化模型输入尺寸
        transforms.Grayscale(),  # 单通道输入
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # 单通道归一化
    ])

    # 加载量化模型
    try:
        from model_train.models.facenet_pi import LiteFaceNet

        checkpoint = torch.load(model_path, map_location='cpu')
        model = LiteFaceNet(embedding_size=32)
        model.load_state_dict(checkpoint['state_dict'], strict=False)

        # 应用动态量化
        quantized_model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        quantized_model.eval()

    except ImportError:
        print("错误: 无法导入LiteFaceNet模型定义")
        raise
    except Exception as e:
        print(f"加载量化模型失败: {e}")
        raise

    database = {}

    print(f"开始处理 {len(image_paths)} 张图片的量化特征提取...")

    for i, img_path in enumerate(image_paths):
        if not os.path.exists(img_path):
            print(f"警告: 图片不存在，跳过: {img_path}")
            continue

        try:
            # 加载图片
            img = Image.open(img_path).convert('RGB')

            # 预处理
            img_tensor = transform(img).unsqueeze(0)  # [1, 1, 48, 48]

            # 提取量化特征
            with torch.no_grad():
                embedding = quantized_model(img_tensor).numpy()[0]
            embedding = embedding / np.linalg.norm(embedding)  # L2归一化

            # 使用文件名作为猫的ID（去掉扩展名）
            cat_id = os.path.splitext(os.path.basename(img_path))[0]
            database[cat_id] = embedding

            print(f"已处理: {cat_id} ({i + 1}/{len(image_paths)})")

        except Exception as e:
            print(f"处理图片失败 {img_path}: {str(e)}")
            continue

    # 保存数据库
    np.save(output_path, database)
    print(f"量化数据库已保存，包含 {len(database)} 个特征向量")

    return database


if __name__ == "__main__":
    main()