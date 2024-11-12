# src/depth_estimation.py
import cv2
import numpy as np
from src.config import MIN_DISPARITY, NUM_DISPARITIES, BLOCK_SIZE, DISPARITY_ALGORITHM

def compute_disparity_map(left_image, right_image):
    """
    Compute the disparity map from stereo images using StereoBM or StereoSGBM.
    """
    if DISPARITY_ALGORITHM == "BM":
        stereo = cv2.StereoBM_create(numDisparities=NUM_DISPARITIES, blockSize=BLOCK_SIZE)
    elif DISPARITY_ALGORITHM == "SGBM":
        stereo = cv2.StereoSGBM_create(
            minDisparity=MIN_DISPARITY,
            numDisparities=NUM_DISPARITIES,
            blockSize=BLOCK_SIZE,
            P1=8 * 3 * BLOCK_SIZE ** 2,   # Parameter controlling disparity smoothness (small smoothness penalty)
            P2=32 * 3 * BLOCK_SIZE ** 2,  # Parameter controlling disparity smoothness (large smoothness penalty)
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32
        )
    else:
        raise ValueError("Unsupported disparity algorithm specified in config.py. Choose 'BM' or 'SGBM'.")

    # Calculate disparity and normalize
    disparity = stereo.compute(left_image, right_image).astype(np.float32) / 16.0
    return disparity

