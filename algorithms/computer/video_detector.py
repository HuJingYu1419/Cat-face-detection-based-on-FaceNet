import cv2
import numpy as np
import onnxruntime as ort
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
import time
from sklearn.metrics.pairwise import cosine_similarity
from ultralytics import YOLO
import os


class VideoCatProcessor:
    def __init__(self, model_path: str, db_path: str = "cat_db.npy", yolo_model_path: str = None):
        """
        视频猫脸处理系统（基于本地视频文件）

        参数:
            model_path: ONNX模型路径
            db_path: 预存embedding数据库路径
            yolo_model_path: 自定义YOLO模型路径（可选）
        """
        # 初始化识别模型和数据库
        self.recog_sess = ort.InferenceSession(model_path)
        self.db = np.load(db_path, allow_pickle=True).item()

        # 初始化YOLOv8检测器 - 使用自定义模型或默认模型
        if yolo_model_path and os.path.exists(yolo_model_path):
            self.yolo = YOLO(yolo_model_path)  # 使用自定义模型
            print(f"使用自定义YOLO模型: {yolo_model_path}")
        else:
            self.yolo = YOLO('yolov8n.pt')  # 使用默认模型
            print("使用默认YOLO模型: yolov8n.pt")

        self.tracker = None

        if not self.db:
            raise ValueError("Embedding database is empty!")

        # 性能统计
        self.frame_count = 0
        self.processing_time = 0
        self.detected_frames = 0
        self.last_match = ("unknown", 0.0)
        self.last_roi = None
        self.results = []

    def preprocess_for_recognition(self, img: np.ndarray) -> np.ndarray:
        """识别模型的预处理"""
        img = cv2.resize(img, (64, 64))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = (img / 255.0 - 0.5) / 0.5
        return img.transpose(2, 0, 1).astype(np.float32)

    def detect_with_yolo(self, frame: np.ndarray):
        """使用YOLO检测猫脸"""
        results = self.yolo(frame, classes=0, verbose=False)  # class 原yolo模型中15是猫，自定义设为0
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()

        if len(boxes) > 0:
            best_idx = np.argmax(confs)
            x1, y1, x2, y2 = map(int, boxes[best_idx])
            return frame[y1:y2, x1:x2], (x1, y1, x2, y2)
        return None, None

    def get_embedding(self, img: np.ndarray) -> np.ndarray:
        """提取单张图像的embedding"""
        input_data = self.preprocess_for_recognition(img)[np.newaxis, ...]
        return self.recog_sess.run(["output"], {"input": input_data})[0]

    def recognize_cat(self, roi: np.ndarray, threshold: float = 0.6) -> Tuple[str, float]:
        """识别单个ROI中的猫"""
        emb = self.get_embedding(roi)
        similarities = {
            cat_id: cosine_similarity(emb, db_emb.reshape(1, -1))[0][0]
            for cat_id, db_emb in self.db.items()
        }
        best_match = max(similarities.items(), key=lambda x: x[1])
        return (best_match[0], best_match[1]) if best_match[1] > threshold else ("unknown", best_match[1])

    def process_video(self, video_path: str, output_path: str, threshold: float = 0.4) -> List[Tuple[int, str, float]]:
        """
        处理本地视频文件

        参数:
            video_path: 输入视频路径
            output_path: 输出视频路径
            threshold: 相似度阈值

        返回:
            识别结果列表(帧索引, 猫ID, 相似度)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")

        # 获取视频信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 创建输出视频
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # 初始化跟踪器
        tracker = None
        detection_interval = 5  # 每n帧进行一次YOLO检测
        frame_idx = 0
        last_second = -1
        embeddings = []

        print(f"\n开始处理视频: {video_path}")
        print(f"总帧数: {total_frames}, FPS: {fps:.1f}, 分辨率: {width}x{height}")
        print("=" * 50)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            start_time = time.time()
            output_frame = frame.copy()
            current_second = int(time.time() % 60)
            result = (frame_idx, "no_cat", 0.0)  # 默认无猫

            try:
                roi, bbox = None, None

                # 检测逻辑
                if frame_idx % detection_interval == 0 or tracker is None:
                    roi, bbox = self.detect_with_yolo(frame)
                    if bbox:
                        try:
                            tracker = cv2.legacy.TrackerKCF_create()
                        except AttributeError:
                            tracker = cv2.TrackerKCF.create()
                        tracker.init(frame, (bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]))
                        self.last_roi = bbox
                else:
                    success, box = tracker.update(frame)
                    if success:
                        x, y, w, h = map(int, box)
                        roi = frame[y:y + h, x:x + w]
                        self.last_roi = (x, y, x + w, y + h)

                # 识别逻辑
                if roi is not None and roi.size > 0:
                    self.detected_frames += 1
                    cat_id, similarity = self.recognize_cat(roi, threshold)
                    self.last_match = (cat_id, similarity)
                    result = (frame_idx, cat_id, similarity)

                    # 标注结果
                    x1, y1, x2, y2 = self.last_roi
                    cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    display_text = f"{cat_id} ({similarity:.2f})" if similarity > threshold else "Unknown"
                    cv2.putText(output_frame, display_text,
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 255, 0), 2)

                    if current_second != last_second:
                        embeddings.append(self.get_embedding(roi))

                # 每秒汇总结果
                if current_second != last_second and embeddings:
                    similarities = defaultdict(float)
                    for emb in embeddings:
                        for cat_id, db_emb in self.db.items():
                            sim = cosine_similarity(emb, db_emb.reshape(1, -1))[0][0]
                            similarities[cat_id] += sim

                    for cat_id in similarities:
                        similarities[cat_id] /= len(embeddings)

                    best_match = max(similarities.items(), key=lambda x: x[1])
                    print(f"第{current_second}秒汇总: {best_match[0]} ({best_match[1]:.2f})")
                    embeddings = []
                    last_second = current_second

                # 写入输出视频
                out.write(output_frame)
                self.results.append(result)

            except Exception as e:
                print(f"处理第{frame_idx}帧时出错: {e}")
                # 即使出错也写入原始帧
                out.write(output_frame)
                self.results.append((frame_idx, "error", 0.0))

            # 更新性能统计
            self.processing_time += time.time() - start_time
            self.frame_count += 1

            # 打印进度
            if frame_idx % 30 == 0:  # 每30帧显示一次进度
                progress = (frame_idx / total_frames) * 100
                print(f"进度: {progress:.1f}% ({frame_idx}/{total_frames}) | "
                      f"检测到猫脸: {self.detected_frames} 帧")

            frame_idx += 1

            # 按Q键可提前退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("用户中断处理")
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        # 性能摘要
        if self.frame_count > 0:
            avg_time = self.processing_time / self.frame_count
            print("\n处理完成!")
            print(f"总处理帧数: {self.frame_count}")
            print(f"检测到猫脸的帧数: {self.detected_frames} ({self.detected_frames / self.frame_count:.1%})")
            print(f"平均每帧处理时间: {avg_time * 1000:.1f}ms")
            print(f"处理速度: {self.frame_count / self.processing_time:.1f} FPS")

        return self.results

    def generate_report(self):
        """生成处理报告"""
        if not self.results:
            return "无处理结果"

        cat_counts = Counter([r[1] for r in self.results if r[1] not in ["no_cat", "error"]])
        total_frames = len(self.results)
        cat_frames = sum(cat_counts.values())

        report = [
            "=" * 50,
            "视频处理报告",
            "=" * 50,
            f"总帧数: {total_frames}",
            f"检测到猫脸的帧数: {cat_frames} ({cat_frames / total_frames:.1%})",
            f"平均处理时间: {self.processing_time / self.frame_count * 1000:.1f}ms/帧",
            "",
            "猫脸识别统计:"
        ]

        for cat_id, count in cat_counts.most_common():
            report.append(f"  {cat_id}: {count} 帧 ({count / total_frames:.1%})")

        return "\n".join(report)


# 使用示例
if __name__ == "__main__":
    # 初始化处理器 - 指定自定义YOLO模型
    processor = VideoCatProcessor(
        model_path="checkpoints/best_model2.onnx",
        db_path="cat_db2.npy",
        yolo_model_path="best2.pt"  # 添加这行
    )

    # 输入输出路径
    input_video = "loki.mp4"
    output_video = "output/processed_video.mp4"

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_video), exist_ok=True)

    try:
        # 处理视频
        print("开始视频处理...")
        results = processor.process_video(input_video, output_video)

        # 生成并打印报告
        report = processor.generate_report()
        print(report)

        print(f"\n处理后的视频已保存至: {output_video}")

    except Exception as e:
        print(f"处理视频时出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()