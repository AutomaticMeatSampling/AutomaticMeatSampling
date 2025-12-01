from PyQt5.QtCore import QThread, pyqtSignal
import sys
from pathlib import Path

#repo_root = Path(__file__).resolve().parent.parent  # Repo/
#seg_folder = repo_root / "Segmentation_Algorithms"
#sys.path.append(str(seg_folder))

#import select_sample_point
#import segment_ld

class ImageWorker(QThread):
    result_ready = pyqtSignal(object)
    img_path = "image_processing_img.png"
    max_num_muscle = 3
    max_num_marbling = 3

    def __init__(self, num_muscle_pts, num_marbling_pts):
        super().__init__()
        self.predictor = segment_ld.load_sam_model()
        self.num_marbling_pts = num_marbling_pts
        self.num_muscle_pts = num_muscle_pts
        
    def set_num_muscle_pts(self, num_muscle_pts):
        if num_muscle_pts > ImageWorker.max_num_muscle:
            return -1
        self.num_muscle_pts = num_muscle_pts
        return self.num_muscle_pts

    def set_num_marbling_pts(self, num_marbling_pts):
        if num_marbling_pts > ImageWorker.max_num_marbling:
            return -1
        self.num_marbling_pts = num_marbling_pts
        return self.num_marbling_pts

    def run(self):
        muscle_points, marbling_points = select_sample_point.segment_and_select_points(ImageWorker.img_path, self.predictor, num_marbling_pts=self.num_marbling_pts, num_muscle_pts=self.num_muscle_pts, show=False)
        self.result_ready.emit({"muscle_points": muscle_points,
                                "marbling_points": marbling_points})