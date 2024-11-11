
import os

# Paths
DATA_PATH = os.path.join("data", "raw")
OUTPUT_PATH = os.path.join("data", "processed")
MODEL_PATH = "yolov8s.pt"  # Default model (can be changed by user)

# Detection Parameters
CONFIDENCE_THRESHOLD = 0.6  # Confidence threshold for detection

# Ensure directories exist
os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(OUTPUT_PATH, exist_ok=True)
