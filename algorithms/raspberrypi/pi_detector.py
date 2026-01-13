import torch
import cv2
import numpy as np
import time
import threading
from queue import Queue
import os
from typing import Dict, List, Tuple
from picamera2 import Picamera2
from sklearn.metrics.pairwise import cosine_similarity
from ultralytics import YOLO
from collections import defaultdict
from model_train.models.facenet_pi import LiteFaceNet

class PiCatMonitorLite:
    def __init__(self, model_path: str, db_path: str = "cat_db.npy", yolo_model_path: str = None):
        # 初始化数据库和摄像头
        self.db = np.load(db_path, allow_pickle=True).item()
        self.picam2 = None
        self.frame_queue = Queue(maxsize=3)
        self.stop_event = threading.Event()

        # 性能统计
        self.frame_count = 0
        self.processing_time = 0
        self.detected_frames = 0
        self.last_match = ("unknown", 0.0)
        self.last_roi = None
        self.display_scale = 0.5  # 显示缩放比例

        # 初始化模型
        self.model = self._load_quantized_model(model_path)
        self.model.eval()
        torch.backends.quantized.engine = 'qnnpack'  # ARM优化

        # 初始化YOLOv8检测器
        if yolo_model_path and os.path.exists(yolo_model_path):
            self.yolo = YOLO(yolo_model_path)  # 使用本地模型文件
        else:
            # 如果没有提供本地路径，尝试默认位置
            default_yolo_path = os.path.expanduser("~/.cache/ultralytics/hub/yolov8n.pt")
            if os.path.exists(default_yolo_path):
                self.yolo = YOLO(default_yolo_path)
            else:
                # 如果都没有，创建目录并等待用户手动放置
                os.makedirs(os.path.dirname(default_yolo_path), exist_ok=True)
                print(f"请将 yolov8n.pt 文件复制到: {default_yolo_path}")
                self.yolo = None  # 设置为None，避免后续错误

        self.tracker = None
        self.detection_interval = 5  # 每5帧进行一次YOLO检测

    def _load_quantized_model(self, model_path):
        """加载量化模型"""
        checkpoint = torch.load(model_path, map_location='cpu')

        model = LiteFaceNet(embedding_size=32)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        return torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        ).eval()

    def get_embedding(self, frame):
        """提取特征向量"""
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (48, 48))
        tensor = torch.from_numpy(resized).float()
        tensor = (tensor / 127.5) - 1.0
        tensor = tensor.unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            return self.model(tensor).numpy()[0]

    def detect_with_yolo(self, frame: np.ndarray):
        """使用YOLO检测猫脸"""
        results = self.yolo(frame, classes=15, verbose=False)  # class 15是猫
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()

        if len(boxes) > 0:
            best_idx = np.argmax(confs)
            x1, y1, x2, y2 = map(int, boxes[best_idx])
            return frame[y1:y2, x1:x2], (x1, y1, x2, y2)
        return None, None

    def recognize_cat(self, roi: np.ndarray, threshold: float = 0.4) -> Tuple[str, float]:
        """识别猫"""
        emb = self.get_embedding(roi)
        similarities = {
            cat_id: cosine_similarity([emb], [db_emb])[0][0]
            for cat_id, db_emb in self.db.items()
        }
        best_match = max(similarities.items(), key=lambda x: x[1])
        return (best_match[0], best_match[1]) if best_match[1] > threshold else ("unknown", best_match[1])

    def camera_capture(self):
        """摄像头采集线程"""
        try:
            self.picam2 = Picamera2()
            config = self.picam2.create_preview_configuration(
                main={"size": (640, 480), "format": "RGB888"}
            )
            self.picam2.configure(config)
            self.picam2.start()

            while not self.stop_event.is_set():
                frame = self.picam2.capture_array()
                if not self.frame_queue.full():
                    self.frame_queue.put(frame)
                time.sleep(0.05)

        except Exception as e:
            print(f"摄像头错误: {e}")
        finally:
            if self.picam2:
                self.picam2.stop()

    def process_frames(self):
        """帧处理线程（整合YOLOv8和跟踪器）"""
        cv2.namedWindow('Pi Cat Monitor', cv2.WINDOW_NORMAL)
        frame_idx = 0
        last_second = -1
        embeddings = []

        while not self.stop_event.is_set():
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop_event.set()
                break

            if self.frame_queue.empty():
                time.sleep(0.01)
                continue

            frame = self.frame_queue.get()
            output_frame = frame.copy()
            current_second = int(time.time() % 60)

            try:
                start_time = time.time()
                roi, bbox = None, None

                # 检测逻辑
                if frame_idx % self.detection_interval == 0 or self.tracker is None:
                    roi, bbox = self.detect_with_yolo(frame)
                    if bbox:
                        try:
                            self.tracker = cv2.legacy.TrackerKCF_create()
                        except AttributeError:
                            self.tracker = cv2.TrackerKCF.create()
                        self.tracker.init(frame, (bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]))
                        self.last_roi = bbox
                else:
                    success, box = self.tracker.update(frame)
                    if success:
                        x, y, w, h = map(int, box)
                        roi = frame[y:y + h, x:x + w]
                        self.last_roi = (x, y, x + w, y + h)

                # 识别逻辑
                if roi is not None and roi.size > 0:
                    self.detected_frames += 1
                    cat_id, similarity = self.recognize_cat(roi)
                    self.last_match = (cat_id, similarity)

                    # 标注结果
                    x1, y1, x2, y2 = self.last_roi
                    cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    display_text = f"{cat_id} ({similarity:.2f})" if similarity > 0.4 else "Unknown"
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
                            sim = cosine_similarity([emb], [db_emb])[0][0]
                            similarities[cat_id] += sim

                    for cat_id in similarities:
                        similarities[cat_id] /= len(embeddings)

                    best_match = max(similarities.items(), key=lambda x: x[1])
                    print(f"第{current_second}秒汇总: {best_match[0]} ({best_match[1]:.2f})")
                    embeddings = []
                    last_second = current_second

                # 显示处理结果
                small_frame = cv2.resize(output_frame,
                                         (int(output_frame.shape[1] * self.display_scale),
                                          int(output_frame.shape[0] * self.display_scale)))
                cv2.imshow('Pi Cat Monitor', small_frame)

                # 更新统计
                self.frame_count += 1
                self.processing_time += time.time() - start_time
                frame_idx += 1

            except Exception as e:
                print(f"处理错误: {e}")

        cv2.destroyAllWindows()

    def start_monitoring(self):
        """启动监控系统"""
        os.system("echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor")

        camera_thread = threading.Thread(target=self.camera_capture)
        process_thread = threading.Thread(target=self.process_frames)

        camera_thread.daemon = True
        process_thread.daemon = True

        camera_thread.start()
        process_thread.start()

        try:
            while not self.stop_event.is_set():
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.stop_event.set()

        camera_thread.join(timeout=1)
        process_thread.join(timeout=1)
        os.system("echo ondemand | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor")

        if self.frame_count > 0:
            print(f"\n平均处理时间: {self.processing_time / self.frame_count * 1000:.1f}ms/帧")
            print(f"检测率: {self.detected_frames / self.frame_count:.1%}")


if __name__ == "__main__":
    monitor = PiCatMonitorLite(
        model_path="checkpoints/quantized_model_lite.pth",
        db_path="cat_db_quant.npy",
        yolo_model_path="/home/hujingyu/cat_monitor_lite/yolov8n.pt"  # 指定自定义路径
    )
    try:
        monitor.start_monitoring()
    except Exception as e:
        print(f"监控错误: {e}")
    finally:
        cv2.destroyAllWindows()