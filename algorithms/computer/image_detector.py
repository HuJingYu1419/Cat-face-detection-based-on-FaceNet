import cv2
import numpy as np
import onnxruntime as ort
from typing import Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
from ultralytics import YOLO
import os


class ImageCatProcessor:
    def __init__(self, model_path: str, db_path: str = "cat_db.npy", yolo_model_path: str = None):
        """
        单张图片猫脸处理系统

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

        if not self.db:
            raise ValueError("Embedding database is empty!")

    def preprocess_for_recognition(self, img: np.ndarray) -> np.ndarray:
        """识别模型的预处理"""
        img = cv2.resize(img, (64, 64))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = (img / 255.0 - 0.5) / 0.5
        return img.transpose(2, 0, 1).astype(np.float32)

    def detect_with_yolo(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[tuple]]:
        """使用YOLO检测猫脸"""
        results = self.yolo(image, classes=0, verbose=False)  # class 原yolo模型中15是猫，自定义设为0

        if results and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()

            best_idx = np.argmax(confs)
            x1, y1, x2, y2 = map(int, boxes[best_idx])
            roi = image[y1:y2, x1:x2]
            return roi, (x1, y1, x2, y2)

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

    def process_image(self, image_path: str, output_path: Optional[str] = None,
                      threshold: float = 0.4) -> Tuple[Optional[str], float, Optional[np.ndarray]]:
        """
        处理单张图片

        参数:
            image_path: 输入图片路径
            output_path: 输出图片路径（可选，为None则不保存）
            threshold: 相似度阈值

        返回:
            (猫ID, 相似度, 标注后的图片)
        """
        # 读取图片
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图片: {image_path}")

        # 复制原始图片用于标注
        output_image = image.copy()

        # 检测猫脸
        roi, bbox = self.detect_with_yolo(image)

        cat_id = "no_cat"
        similarity = 0.0

        if roi is not None and roi.size > 0:
            # 识别猫
            cat_id, similarity = self.recognize_cat(roi, threshold)

            # 在图片上标注结果
            x1, y1, x2, y2 = bbox
            cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            display_text = f"{cat_id} ({similarity:.2f})" if similarity > threshold else "Unknown"
            cv2.putText(output_image, display_text,
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)

        # 保存输出图片
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, output_image)

        return cat_id, similarity, output_image

    def process_image_from_array(self, image: np.ndarray, output_path: Optional[str] = None,
                                 threshold: float = 0.4) -> Tuple[Optional[str], float, Optional[np.ndarray]]:
        """
        直接从numpy数组处理图片

        参数:
            image: 输入图片数组
            output_path: 输出图片路径（可选）
            threshold: 相似度阈值

        返回:
            (猫ID, 相似度, 标注后的图片)
        """
        # 复制原始图片用于标注
        output_image = image.copy()

        # 检测猫脸
        roi, bbox = self.detect_with_yolo(image)

        cat_id = "no_cat"
        similarity = 0.0

        if roi is not None and roi.size > 0:
            # 识别猫
            cat_id, similarity = self.recognize_cat(roi, threshold)

            # 在图片上标注结果
            x1, y1, x2, y2 = bbox
            cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            display_text = f"{cat_id} ({similarity:.2f})" if similarity > threshold else "Unknown"
            cv2.putText(output_image, display_text,
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)

        # 保存输出图片
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, output_image)

        return cat_id, similarity, output_image


# 使用示例
if __name__ == "__main__":
    # 初始化处理器
    processor = ImageCatProcessor(
        model_path="checkpoints/best_model2.onnx",
        db_path="cat_db2.npy",
        yolo_model_path="best2.pt"
    )

    # 处理单张图片
    input_image = "test_cat.jpg"
    output_image = "output/processed_image.jpg"

    try:
        # 处理图片
        cat_id, similarity, processed_img = processor.process_image(input_image, output_image)

        print(f"识别结果: {cat_id} (相似度: {similarity:.2f})")

        if cat_id != "no_cat":
            print(f"图片中检测到猫: {cat_id}, 相似度: {similarity:.2f}")
        else:
            print("图片中未检测到猫")

        print(f"处理后的图片已保存至: {output_image}")

        # 显示图片（可选）
        cv2.imshow("Processed Image", processed_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"处理图片时出错: {e}")
        import traceback

        traceback.print_exc()