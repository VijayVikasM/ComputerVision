
import cv2
from ultralytics import YOLO
from src.config import MODEL_PATH, CONFIDENCE_THRESHOLD

def load_yolo_model():
    """
    Load the YOLO model specified in the configuration.
    """
    return YOLO(MODEL_PATH)

def detect_objects(frame, model):
    """
    Perform object detection on a single frame using YOLO.
    """
    results = model.track(frame, stream=True)
    detections = []

    for result in results:
        classes_names = result.names  # Class names
        for box in result.boxes:
            if box.conf[0] > CONFIDENCE_THRESHOLD:
                # Get coordinates, class, and confidence
                [x1, y1, x2, y2] = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cls = int(box.cls[0])
                class_name = classes_names[cls]
                confidence = box.conf[0]
                detections.append({
                    "box": [x1, y1, x2, y2],
                    "class": class_name,
                    "confidence": confidence,
                    "cls_id": cls
                })
    return detections
