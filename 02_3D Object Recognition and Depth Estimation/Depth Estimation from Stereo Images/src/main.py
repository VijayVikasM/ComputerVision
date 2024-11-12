
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import DATA_PATH, OUTPUT_PATH
from src.depth_estimation import compute_disparity_map
from src.utils import load_stereo_images, save_disparity_map

def main():
    # Define file paths for left and right images
    left_image_path = os.path.join(DATA_PATH, "left_image.png")
    right_image_path = os.path.join(DATA_PATH, "right_image.png")
    
    # Load stereo images
    left_image, right_image = load_stereo_images(left_image_path, right_image_path)
    
    # Compute disparity map
    disparity_map = compute_disparity_map(left_image, right_image)
    save_disparity_map(disparity_map, os.path.join(OUTPUT_PATH, "disparity_map.png"))

if __name__ == "__main__":
    main()
