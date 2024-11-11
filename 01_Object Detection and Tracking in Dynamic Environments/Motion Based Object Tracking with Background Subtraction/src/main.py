# src/main.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import save_video
from src.background_subtraction import apply_background_subtraction
from src.config import DATA_PATH

def main():
    video_path = os.path.join(DATA_PATH, "example_video.avi")
    frames = []  # Collect frames if you want to save the video later

    # Assuming apply_background_subtraction returns processed frames
    processed_frames = apply_background_subtraction(video_path)  
    frames.extend(processed_frames)  # Collect frames from processing

    save_video(frames, "output_video.avi")  # Save collected frames to a video

if __name__ == "__main__":
    main()
