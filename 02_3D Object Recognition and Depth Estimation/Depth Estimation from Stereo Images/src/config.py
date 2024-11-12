# src/config.py
import os
import re

# Paths
DATA_PATH = os.path.join("data", "chess1")
OUTPUT_PATH = os.path.join("data", "output")
CALIBRATION_FILE = os.path.join(DATA_PATH, "calib.txt")  # Calibration file path

# Ensure directories exist
os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Load calibration parameters from calib.txt
def load_calibration():
    calibration_params = {}
    with open(CALIBRATION_FILE, 'r') as f:
        for line in f:
            key, value = line.strip().split('=', 1)
            calibration_params[key.strip()] = value.strip()
    return calibration_params

# Parse focal length, baseline, and doffs from calibration
calibration = load_calibration()

# Extract focal length from cam0 matrix
cam0_matrix = calibration.get("cam0")
focal_length = None
if cam0_matrix:
    # Extract the first element in cam0 matrix, which is the focal length (f)
    match = re.match(r"\[(\d+\.\d+)", cam0_matrix)  
    if match:
        focal_length = float(match.group(1))

# Use default focal length if none found
focal_length = focal_length if focal_length else 1758.23  # Default from example if missing
print(focal_length)
# Baseline in meters (convert from mm)
baseline = float(calibration.get("baseline", 111.53)) / 1000

# Optional parameters
doffs = float(calibration.get("doffs", 0))  # Offset
NUM_DISPARITIES = int(calibration.get("ndisp", 290))  # Number of disparities
BLOCK_SIZE = 15  # Block size for stereo matching (set as per requirements)
