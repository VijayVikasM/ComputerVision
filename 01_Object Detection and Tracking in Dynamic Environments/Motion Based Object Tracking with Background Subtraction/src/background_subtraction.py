# src/background_subtraction.py

import cv2
from src.config import BG_SUBTRACTION_METHOD

def create_background_subtractor():
    if BG_SUBTRACTION_METHOD == "MOG2":
        return cv2.createBackgroundSubtractorMOG2()
    elif BG_SUBTRACTION_METHOD == "KNN":
        return cv2.createBackgroundSubtractorKNN()
    else:
        raise ValueError("Invalid background subtraction method.")

def apply_background_subtraction(video_path):
    cap = cv2.VideoCapture(video_path)
    bg_subtractor = create_background_subtractor()
    frames = []  # Initialize an empty list to store frames

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        fg_mask = bg_subtractor.apply(frame)
        frames.append(fg_mask)  # Collect processed frames

        # Display frames (optional)
        cv2.imshow("Frame", frame)
        cv2.imshow("Foreground Mask", fg_mask)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return frames  # Return the list of frames, even if empty
