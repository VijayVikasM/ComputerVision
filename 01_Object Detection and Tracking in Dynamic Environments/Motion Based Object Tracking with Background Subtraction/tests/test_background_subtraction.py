
import cv2
from src.background_subtraction import create_background_subtractor

def test_create_background_subtractor():
    bg_subtractor = create_background_subtractor()
    assert bg_subtractor is not None
    assert isinstance(bg_subtractor, (cv2.BackgroundSubtractorMOG2, cv2.BackgroundSubtractorKNN))
