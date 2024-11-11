# src/config.py
import os

# Paths
DATA_PATH = os.path.join("data", "raw")
OUTPUT_PATH = os.path.join("data", "processed")  # Directory for output files

# Ensure the output path exists
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Parameters for Background Subtraction
BG_SUBTRACTION_METHOD = "MOG2"  # Options: "MOG2" or "KNN"
