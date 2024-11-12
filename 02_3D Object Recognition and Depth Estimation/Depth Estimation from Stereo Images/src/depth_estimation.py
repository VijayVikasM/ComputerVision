# src/depth_estimation.py
import cv2
import numpy as np
from src.config import NUM_DISPARITIES, BLOCK_SIZE, doffs

def compute_disparity_map(left_image, right_image):
    """
    Compute the disparity map from stereo images.
    """
    stereo = cv2.StereoBM_create(numDisparities=NUM_DISPARITIES, blockSize=BLOCK_SIZE)
    disparity = stereo.compute(left_image, right_image).astype(np.float32) / 16.0
    return disparity

