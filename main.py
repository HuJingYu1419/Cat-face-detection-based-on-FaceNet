#!/usr/bin/env python3

import argparse
import os
import sys
import tempfile
import numpy as np
from PIL import Image
from torchvision import transforms
import onnxruntime as ort
from pathlib import Path
from algorithms.computer.image_detector import ImageCatProcessor
from algorithms.computer.video_detector import VideoCatProcessor
from algorithms.computer.monitor_detector import LightweightCatMonitor


def main():
    # 创建ArgumentParser对象
    formatter = lambda prog: argparse.HelpFormatter(prog, max_help_position=36)
    desc = '''
    猫脸识别系统 - 支持图片、视频和实时监控处理
    使用YOLO检测猫脸并用FaceNet模型进行识别比对
    '''

    parser = argparse.ArgumentParser(description=desc, formatter_class=formatter)

    # 添加参数
    parser.add_argument('-i', '--input-image',
                        help='输入图片文件',
                        nargs='?',  # 表示参数是可选的
                        const='data/examples/profile/Loki.jpg',  # 当有选项但没有值时使用这个值
                        metavar='<file>')

    parser.add_argument('-v', '--input-video',
                        help='输入视频文件',
                        nargs='?',  # 表示参数是可选的
                        const='data/examples/video/loki.mp4',  # 当有选项但没有值时使用这个值
                        metavar='<file>')

    parser.add_argument('-mv', '--monitor-video',
                        help='启用实时监控模式',
                        action='store_true')

    parser.add_argument('-c', '--camera',
                        help='摄像头设备ID或RTSP地址 (默认: 0)',
                        default='0',
                        metavar='<id/url>')

    parser.add_argument('-o', '--output',
                        help='输出路径',
                        default='detection_result',
                        metavar='<path>')

    parser.add_argument('-m', '--model',
                        help='ONNX模型路径',
                        default='models/facenet_model.onnx',
                        metavar='<file>')

    parser.add_argument('-d', '--database-folder',
                        help='包含猫脸图片的文件夹路径，用于生成特征数据库',
                        metavar='<folder>')
    
    parser.add_argument('-db', '--database-path',
                        help='特征数据库文件路径 (优先于数据库文件夹)',
                        default='data/examples/embeddings/cat_db.npy',
                        metavar='<file>')

    parser.add_argument('-y', '--yolo-model',
                        help='YOLO模型路径',
                        default='models/catface_yolo.pt',
                        metavar='<file>')

    parser.add_argument('-t', '--threshold',
                        help='相似度阈值 (默认: 0.4)',
                        type=float,
                        default=0.4,
                        metavar='<float>')

    # 解析参数
    args = parser.parse_args()

    # 验证参数 - 检查用户是否在命令行中使用了这些选项
    has_image = any(arg in sys.argv for arg in ['-i', '--input-image'])
    has_video = any(arg in sys.argv for arg in ['-v', '--input-video'])
    has_monitor = args.monitor_video

    input_modes = sum([has_image, has_video, has_monitor])

    if input_modes == 0:
        parser.error("必须指定输入模式: 使用 -i (图片), -v (视频) 或 -mv (实时监控)")

    if input_modes > 1:
        parser.error("不能同时指定多个输入模式")

    # 处理特征数据库
    db_path, is_temp_db = handle_database_folder(args.database_folder, args.model)

    # 确保输出目录存在（如果不是监控模式）
    if not args.monitor_video:
        os.makedirs(args.output, exist_ok=True)

    try:
        if has_image:
            # 图片处理模式
            process_image(args.input_image, args.output, args.model,
                          db_path, args.yolo_model, args.threshold)

        elif has_video:
            # 视频处理模式
            process_video(args.input_video, args.output, args.model,
                          db_path, args.yolo_model, args.threshold)

        elif args.monitor_video:
            # 实时监控模式
            process_monitor(args.camera, args.model,
                            db_path, args.yolo_model, args.threshold)

    except Exception as e:
        print(f"处理过程中出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    finally:
        # 清理临时数据库文件
        if is_temp_db and os.path.exists(db_path):
            os.remove(db_path)
            print(f"已删除临时数据库文件: {db_path}")


def handle_database_folder(folder_path, model_path):
    """
    处理特征数据库文件夹，生成临时的npy文件
    返回: (数据库路径, 是否为临时文件)
    """
    default_db_path = 'data/examples/embeddings/cat_db.npy'

    # 如果没有提供文件夹，使用默认数据库
    if not folder_path:
        print(f"使用默认特征数据库: {default_db_path}")
        return default_db_path, False

    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"错误: 文件夹不存在: {folder_path}")
        print(f"使用默认特征数据库: {default_db_path}")
        return default_db_path, False

    # 检查默认数据库是否存在
    if not os.path.exists(default_db_path):
        print(f"警告: 默认数据库不存在 {default_db_path}")

    # 获取文件夹中的所有图片文件
    image_paths = get_image_files_from_folder(folder_path)

    if not image_paths:
        print(f"错误: 文件夹中没有找到图片文件: {folder_path}")
        print(f"使用默认特征数据库: {default_db_path}")
        return default_db_path, False

    # 创建临时数据库
    print(f"根据文件夹中的 {len(image_paths)} 张图片生成临时特征数据库...")

    # 创建临时npy文件
    temp_db_path = tempfile.mktemp(suffix='.npy', prefix='temp_cat_db_')

    try:
        # 构建数据库
        build_database_from_images(model_path, image_paths, temp_db_path)

        print(f"临时特征数据库已生成: {temp_db_path}")
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


def build_database_from_images(model_path, image_paths, output_path):
    """
    直接从图片列表构建特征数据库
    """
    # 定义预处理转换
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 初始化ONNX模型
    sess = ort.InferenceSession(model_path)
    database = {}

    print(f"开始处理 {len(image_paths)} 张图片的特征提取...")

    for i, img_path in enumerate(image_paths):
        if not os.path.exists(img_path):
            print(f"警告: 图片不存在，跳过: {img_path}")
            continue

        try:
            # 加载图片
            img = Image.open(img_path).convert('RGB')

            # 预处理
            img_tensor = transform(img).unsqueeze(0)  # [1, 3, 64, 64]
            input_data = img_tensor.numpy()

            # 提取特征
            embedding = sess.run(["output"], {"input": input_data})[0]
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
    print(f"数据库已保存，包含 {len(database)} 个特征向量")

    return database


# 以下函数保持不变...
def process_image(input_path, output_path, model_path, db_path, yolo_model_path, threshold):
    """
    处理单张图片
    """
    print(f"开始处理图片: {input_path}")
    print(f"使用特征数据库: {db_path}")

    # 初始化图片处理器
    processor = ImageCatProcessor(
        model_path=model_path,
        db_path=db_path,
        yolo_model_path=yolo_model_path
    )

    # 生成输出文件名
    basename = os.path.splitext(os.path.basename(input_path))[0]
    output_file = os.path.join(output_path, f"{basename}_processed.jpg")

    # 处理图片
    cat_id, similarity, processed_img = processor.process_image(
        input_path, output_file, threshold
    )

    # 输出结果
    print("\n" + "=" * 50)
    print("图片处理结果:")
    print("=" * 50)

    if cat_id != "no_cat":
        print(f"识别到猫: {cat_id}")
        print(f"相似度: {similarity:.3f}")
        print(f"置信度: {similarity:.2%}")
    else:
        print("未检测到猫")

    print(f"输出文件: {output_file}")
    print("=" * 50)


def process_video(input_path, output_path, model_path, db_path, yolo_model_path, threshold):
    """
    处理视频文件
    """
    print(f"开始处理视频: {input_path}")
    print(f"使用特征数据库: {db_path}")

    # 初始化视频处理器
    processor = VideoCatProcessor(
        model_path=model_path,
        db_path=db_path,
        yolo_model_path=yolo_model_path
    )

    # 生成输出文件名
    basename = os.path.splitext(os.path.basename(input_path))[0]
    output_file = os.path.join(output_path, f"{basename}_processed.mp4")

    # 处理视频
    print("开始视频处理，请稍候...")
    results = processor.process_video(input_path, output_file, threshold)

    # 生成并显示报告
    report = processor.generate_report()
    print("\n" + "=" * 50)
    print("视频处理完成!")
    print("=" * 50)
    print(report)
    print(f"输出视频: {output_file}")
    print("=" * 50)


def process_monitor(camera_input, model_path, db_path, yolo_model_path, threshold):
    """
    处理实时监控
    """
    print(f"启动实时监控模式...")
    print(f"使用特征数据库: {db_path}")

    # 解析摄像头输入
    try:
        # 尝试将输入转换为整数（设备ID）
        camera_id = int(camera_input)
        print(f"使用摄像头设备: {camera_id}")
    except ValueError:
        # 如果是字符串，可能是RTSP地址或URL
        camera_id = camera_input
        print(f"使用视频流: {camera_input}")

    # 初始化监控处理器
    monitor = LightweightCatMonitor(
        model_path=model_path,
        db_path=db_path,
        yolo_model_path=yolo_model_path,
        camera_id=camera_id
    )

    # 启动监控
    try:
        monitor.start_monitoring(threshold=threshold)
    except KeyboardInterrupt:
        print("\n监控已由用户中断")
    except Exception as e:
        print(f"监控过程中出错: {e}")
    finally:
        print("监控系统已关闭")


if __name__ == "__main__":
    main()