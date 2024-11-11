# src/utils.py
import cv2
import os
from src.config import OUTPUT_PATH

def save_video(frames, output_filename, fps=20, frame_size=(640, 480)):
    """
    Save a series of frames as a video.
    """
    output_path = os.path.join(OUTPUT_PATH, output_filename)  # Define the full path
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, frame_size)
    
    for frame in frames:
        out.write(frame)
    
    out.release()
    print(f"Video saved to {output_path}")
