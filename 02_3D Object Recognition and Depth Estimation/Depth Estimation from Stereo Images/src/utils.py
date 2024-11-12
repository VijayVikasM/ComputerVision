import cv2
import os
import numpy as np

def load_stereo_images(left_image_path, right_image_path):
    """
    Load left and right images in grayscale.
    """
    left_image = cv2.imread(left_image_path, cv2.IMREAD_GRAYSCALE)
    right_image = cv2.imread(right_image_path, cv2.IMREAD_GRAYSCALE)
    return left_image, right_image

def save_disparity_map(disparity_map, output_path):
    """
    Save the disparity map as an image.
    """
    disparity_map_normalized = cv2.normalize(disparity_map, None, 0, 255, cv2.NORM_MINMAX)
    disparity_map_normalized = np.uint8(disparity_map_normalized)
    cv2.imwrite(output_path, disparity_map_normalized)
    print(f"Disparity map saved to {output_path}")
