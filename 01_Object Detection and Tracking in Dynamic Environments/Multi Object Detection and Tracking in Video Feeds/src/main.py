
import cv2
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.detection import load_yolo_model, detect_objects
from src.utils import get_class_color
from src.config import DATA_PATH

def main():
    # Load the YOLO model
    model = load_yolo_model()

    # Load video
    video_path = os.path.join(DATA_PATH, "example_video.mp4")
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect objects
        detections = detect_objects(frame, model)

        # Display detections
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            class_name = det["class"]
            confidence = det["confidence"]
            color = get_class_color(det["cls_id"])

            # Draw the rectangle and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f'{class_name} {confidence:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Show frame
        cv2.imshow("YOLO Detection", frame)

        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
