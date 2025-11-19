from PyQt5.QtCore import QThread, pyqtSignal
import sys
from pathlib import Path
import serial
import time
from ImageWorker import ImageWorker

repo_root = Path(__file__).resolve().parent.parent  # Repo/
seg_folder = repo_root / "Robot_API_Tests"
sys.path.append(str(seg_folder))

from pydexarm import Dexarm
from takepicture import take_photo

class RobotWorker(QThread):
    progress = pyqtSignal(str)
    steps = ["PHOTO", "TIP_PICKUP", "SOLVENT_PICKUP", "SAMPLE_COLLECTION", "SAMPLE_DROPOFF", "TIP_DISPOSAL"]

    def __init__(self, sample_mode, main_window):
        super().__init__()
        self.complete_success = False
        self._stop_requested = False
        self.sample_mode = sample_mode
        self.main_window = main_window

        # Setup connction to robot here
        print("Setting up robot connections...")
        # self.dexarm = Dexarm(port="COM3")
        # self.ser_micro = serial.Serial(port='COM4', baudrate=115200, timeout=0.1)

    def run(self):
        self.progress.emit("ROBOT_START")
        print(f"Running robot worker process with sample mode: {self.sample_mode}")

        for step in RobotWorker.steps:
            if self._stop_requested:
                self.finish()
                return
            
            self.progress.emit(step)

            match step:
                case "PHOTO":
                    self.collect_photo()
                case "TIP_PICKUP":
                    time.sleep(2)
                case "SOLVENT_PICKUP":
                    time.sleep(2)
                case "SAMPLE_COLLECTION":
                    if self.sample_mode == "Automatic":
                        timeout = 10
                    else:
                        timeout = 100
                    if not self.main_window.image_coords_ready_event.wait(timeout=timeout):
                        self.progress.emit("Error: Image coord result not received in time")
                        self.finish()
                        return
                    
                    muscle_pts = self.main_window.selected_points["muscle_points"]
                    marbling_pts = self.main_window.selected_points["marbling_points"]
                    self.progress.emit("Received coordinates!!!")

                    # Clear even for next use
                    self.main_window.image_coords_ready_event.clear()
                    time.sleep(2)
                case "SAMPLE_DROPOFF":
                    time.sleep(2)
                case "TIP_DISPOSAL":
                    time.sleep(2)
                case _:
                    print("Unknown step what???")

        self.complete_success = True
        self.finish()

    def finish(self):
        if self.complete_success:
            self.progress.emit("ROBOT_STOP_SUCCESS")
        else:
            self.progress.emit("ROBOT_STOP")
        self._stop_requested = False

    
    def stop(self):
        # Set flag to be handled by main run function
        if self.isRunning():
            self._stop_requested = True

    def collect_photo(self):
        # Do steps to move to photo position

        # Take photo
        # take_photo(ImageWorker.img_path)

        if self.sample_mode == "Manual":
            self.progress.emit("START_MANUAL_POINT_SELECTION")
        elif self.sample_mode == "Automatic":
            self.progress.emit("START_AUTOMATIC_POINT_SELECTION")

        time.sleep(2)