
from ultralytics import YOLO
import time


def train_cat_detector():
    """YOLO训练函数（增强进度打印）"""
    # 加载预训练模型
    model = YOLO('../../../config/yolov8n.pt')

    print("开始训练猫脸检测模型")
    print("=" * 60)
    print("训练配置:")
    print(f"  模型: yolov8n.pt")
    print(f"  轮次: 50")
    print(f"  批次大小: 8")
    print(f"  设备: CPU")
    print("=" * 60)

    # 开始训练
    results = model.train(
        data='../../../config/cat_data.yaml',
        epochs=50,
        imgsz=640,
        batch=8,
        patience=15,
        device=0,
        project='results',
        name='cat_detection',
        verbose=True  # 这是关键，启用详细输出
    )

    print("=" * 60)
    print("训练完成！")
    print(f"最佳模型保存在: results/cat_detection/weights/best.pt")


if __name__ == "__main__":
    start_time = time.time()
    train_cat_detector()
    end_time = time.time()

    # 计算并打印总时间
    total_time = end_time - start_time
    minutes = int(total_time // 60)
    seconds = int(total_time % 60)
    print(f"总训练时间: {minutes}分{seconds}秒")