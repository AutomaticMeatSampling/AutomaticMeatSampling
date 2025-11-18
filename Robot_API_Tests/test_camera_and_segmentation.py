import cv2
from takepicture import take_photo
import os
from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parent.parent  # Repo/
seg_folder = repo_root / "Segmentation_Algorithms"
sys.path.append(str(seg_folder))

import select_sample_point
import segment_ld

"""
Test Script to test incorporating segmentation functions into Robot_API_Tests folder along with take_photo function
"""

SCRIPT_DIR = Path(__file__).resolve()
def rel_path(path):
    filepath = SCRIPT_DIR.parent / path
    folder = os.path.dirname(filepath)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)
    return filepath


if __name__ == "__main__":

    predictor = segment_ld.load_sam_model()

    num_marbling_pts=1
    num_muscle_pts=0

    img_path = rel_path("test_photo.png")
    take_photo(img_path)

    muscle_points, marbling_points = select_sample_point.segment_and_select_points(img_path, predictor, num_marbling_pts=num_marbling_pts, num_muscle_pts=num_muscle_pts)