import cv2
import numpy as np
import onnxruntime as ort
from collections import defaultdict
from typing import Dict, List, Tuple
import time
from sklearn.metrics.pairwise import cosine_similarity
import threading
from queue import Queue
from ultralytics import YOLO
import os


class LightweightCatMonitor:
    def __init__(self, model_path: str, db_path: str = "cat_db.npy", yolo_model_path: str = "yolo_model.pt",camera_id=0):
        """
        轻量级实时猫脸监控系统

        参数:
            model_path: ONNX模型路径
            db_path: 预存embedding数据库路径
            yolo_model_path: YOLO模型路径（可选）
        """
        # 初始化识别模型和数据库
        self.recog_sess = ort.InferenceSession(model_path)
        self.db = np.load(db_path, allow_pickle=True).item()
        self.camera_id=camera_id

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

        # 视频处理队列
        self.frame_queue = Queue(maxsize=30)
        self.stop_event = threading.Event()

        # 性能统计
        self.frame_count = 0
        self.processing_time = 0
        self.detected_frames = 0
        self.last_match = ("unknown", 0.0)
        self.last_roi = None

    def preprocess_for_recognition(self, img: np.ndarray) -> np.ndarray:
        """识别模型的预处理"""
        img = cv2.resize(img, (64, 64))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = (img / 255.0 - 0.5) / 0.5
        return img.transpose(2, 0, 1).astype(np.float32)

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

    def camera_capture(self,camera_id):
        """摄像头采集线程"""
        if isinstance(camera_id, str) and camera_id.startswith(('rtsp://', 'http://')):
            cap = cv2.VideoCapture(camera_id)
        else:
            cap = cv2.VideoCapture(int(camera_id))  # 确保转换为整数
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 15)

        if not cap.isOpened():
            raise ValueError(f"无法打开摄像头: {camera_id}")

        print(f"摄像头采集已启动 (设备: {camera_id})...")
        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                print("摄像头读取失败")
                break

            if not self.frame_queue.full():
                self.frame_queue.put(frame)
            else:
                time.sleep(0.01)

        cap.release()
        print("摄像头采集已停止")

    def process_frames(self, threshold: float = 0.6):
        """轻量化的帧处理线程"""
        cv2.namedWindow('Cat Monitor - Live View', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Cat Monitor - Live View', 800, 600)

        print("实时监控已启动...")
        detection_interval = 5  # 每n帧进行一次YOLO检测
        frame_idx = 0
        tracker = None
        last_second = -1
        embeddings = []

        while not self.stop_event.is_set() or not self.frame_queue.empty():
            if self.frame_queue.empty():
                time.sleep(0.01)
                continue

            frame = self.frame_queue.get()
            current_second = int(time.time() % 60)
            output_frame = frame.copy()

            try:
                start_time = time.time()
                roi, bbox = None, None

                # 检测逻辑
                if frame_idx % detection_interval == 0 or tracker is None:
                    roi, bbox = self.detect_with_yolo(frame)
                    if bbox:
                            try:
                                # 优先尝试legacy方式
                                tracker = cv2.legacy.TrackerKCF_create()
                            except AttributeError:
                                # 回退到新API方式
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

                # 显示实时画面
                cv2.imshow('Cat Monitor - Live View', output_frame)

                # 更新性能统计
                self.processing_time += time.time() - start_time
                self.frame_count += 1
                frame_idx += 1

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.stop_event.set()
                    break

            except Exception as e:
                print(f"处理帧时出错: {e}")
                continue

        cv2.destroyAllWindows()
        print("实时监控已停止")

    def start_monitoring(self, threshold: float = 0.6):
        """启动监控系统"""
        camera_thread = threading.Thread(target=self.camera_capture, args=(self.camera_id,))
        process_thread = threading.Thread(target=self.process_frames, args=(threshold,))

        camera_thread.daemon = True
        process_thread.daemon = True

        camera_thread.start()
        process_thread.start()

        print(f"监控系统已启动，摄像头: {self.camera_id}，阈值: {threshold}，按q键停止...")
        try:
            while not self.stop_event.is_set():
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.stop_event.set()

        camera_thread.join(timeout=1)
        process_thread.join(timeout=1)

        # 打印性能摘要
        if self.frame_count > 0:
            avg_time = self.processing_time / self.frame_count
            print("\n性能摘要:")
            print(f"总处理帧数: {self.frame_count}")
            print(f"检测到猫脸的帧数: {self.detected_frames} ({self.detected_frames / self.frame_count:.1%})")
            print(f"平均每帧处理时间: {avg_time * 1000:.1f}ms")
            print(f"处理速度: {self.frame_count / self.processing_time:.1f} FPS")


if __name__ == "__main__":
    monitor = LightweightCatMonitor(
        "../Learning code-acurate/checkpoints/best_model.onnx",
        "cat_db.npy",
        "yolov8n.pt",  # 添加yolo模型路径
        0
    )
    try:
        monitor.start_monitoring(threshold=0.6)  # 使用默认摄像头
    except Exception as e:
        print(f"监控系统出错: {e}")
    finally:
        cv2.destroyAllWindows()