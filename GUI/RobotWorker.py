from PyQt5.QtCore import QThread, pyqtSignal
import sys
from pathlib import Path
import serial
import time
from ImageWorker import ImageWorker
import cv2
import os

repo_root = Path(__file__).resolve().parent.parent  # Repo/
seg_folder = repo_root / "Robot_API_Tests"
sys.path.append(str(seg_folder))

from pydexarm import Dexarm
from takepicture import take_photo

SCRIPT_DIR = Path(__file__).resolve()
def rel_path(path):
    filepath = SCRIPT_DIR.parent / path
    folder = os.path.dirname(filepath)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)
    return filepath

class RobotWorker(QThread):
    progress = pyqtSignal(str)
    steps = ["TIP_PICKUP", "SOLVENT_PICKUP", "MOVE_TO_XY_SAMPLE_COORD", "SAMPLE_DROPOFF", "TIP_DISPOSAL"]
    #"TIP_PICKUP", "SOLVENT_PICKUP", "MOVE_TO_XY_SAMPLE_COORD", "MOVE_DOWN_TO_SAMPLE_COORD", "SAMPLE_DROPOFF", "TIP_DISPOSAL"

    def __init__(self, sample_mode, main_window, curr_tip=1, max_tip_num=6, curr_vial=1, max_vial_num=3, use_robot_simulator=False, use_simulated_sample=False):
        super().__init__()

        # ------------------------
        # Simulator configurations
        # ------------------------
        self.use_robot_simulator = use_robot_simulator
        self.use_simulated_sample = use_simulated_sample

        #--------------------------------
        # Sample Collection Configuration
        #--------------------------------
        self.sample_mode = sample_mode
        self.total_num_samples = 1 # Assume 1 sample at start, will be updated with image processing results

        #--------------------------------------
        # Track of current pipette tip and vial
        #--------------------------------------
        self.curr_tip = curr_tip
        self.max_tip_num = max_tip_num
        self.curr_vial = curr_vial
        self.max_vial_num = max_vial_num

        #---------------
        # Status/Flags
        #---------------
        self.complete_success = False
        self._stop_requested = False
        self.image_processing_complete = False
        self.last_completed_step = None
        
        self.main_window = main_window

        #--------------------------------
        # Setup connection to robot ports
        #--------------------------------
        print("Setting up robot connections...")
        if not self.use_robot_simulator:
            
            if os.name == "nt":#Windows
                self.dexarm = Dexarm(port="COM6")
                self.ser_micro = serial.Serial(port='COM4', baudrate=115200, timeout=0.1)
            else: #Mac/Linux
                self.dexarm = Dexarm(port="/dev/cu.usbmodem207B396A36311")#COM3
                self.ser_micro = serial.Serial(port='/dev/cu.usbmodem142101', baudrate=115200, timeout=0.1)

    def run(self):
        self.progress.emit("ROBOT_START")
        print(f"Running robot worker process with sample mode: {self.sample_mode}")

        # --------------------------------------------------------------------------------
        # STEP 0: Initial step of getting photo, sets off automatic/manual coord selection
        # --------------------------------------------------------------------------------
        self.collect_photo()
        self.last_completed_step = "PHOTO"

        # Start sample process for each sample pt...
        num_samples_collected = 0
        while num_samples_collected < self.total_num_samples:
            num_samples_collected += 1
            self.progress.emit(f"Starting robot movement for sample {num_samples_collected}")

            # ------------------------------------
            # Error Checking: Check tip/vial count
            # ------------------------------------
            if self.isOverMaxTipNum() or self.isOverMaxVialNum():
                self.progress.emit("Not enough tips or vials to complete remaining samples")
                self.finish()

            # -------------------------------------------------------
            # Loop through sample collection steps for each sample pt
            # -------------------------------------------------------
            for step in RobotWorker.steps:

                # ----------------------------------
                # Watch for STOP interrupt between steps
                # ----------------------------------
                if self._stop_requested:
                    self.finish()
                    return
                
                self.progress.emit(step)

                # ------------------------------------------
                # SIMULATED ROBOT MOVEMENT
                # ------------------------------------------
                if self.use_robot_simulator:
                    time.sleep(2)
                    break
                

                match step:

                    # ------------------------------------------------------------------
                    # STEP 1: (a) Move to pipette tip area and (b) Pick up a pipette tip
                    # ------------------------------------------------------------------
                    case "TIP_PICKUP":
                        self.dexarm.move_to_pipette_tip(self.curr_tip)
                        self.curr_tip += 1

                    # --------------------------------------------------------
                    # STEP 2: (a) Move to solvent tube and (b) Collect solvent
                    # --------------------------------------------------------
                    case "SOLVENT_PICKUP":
                        self.dexarm.step_3_move_to_solvent()

                    # ---------------------------------------------------------------
                    # STEP 3: (a) Calculate robot coordinate, may need to wait for coords from main_window (auto/manual) & (b) move to calculated location
                    # ---------------------------------------------------------------
                    case "MOVE_TO_XY_SAMPLE_COORD":

                        # If first time here, may need to wait for results to appear
                        if not self.image_processing_complete:
                            ret = self.wait_for_coord_results()
                            if ret == -1:
                                self.finish()
                                return
                            
                        # ********* TODO: ADD FUNC TO MOVE TO sample XY location (Breanna) ***************
                        ## temporary adjustment for data measurement
                        current_pipette_tip_points = self.sample_list[num_samples_collected - 1]
                        for i in range(0, len(current_pipette_tip_points)):
                            self.dexarm.move_to_point_position(current_pipette_tip_points[i][0], current_pipette_tip_points[i][1])
                            self.dexarm.move_down_meat(self.ser_micro)
                        # for i in range (0, len(self.muscle_pts)):
                        #     self.dexarm.move_to_point_position(self.muscle_pts[i][0], self.muscle_pts[i][1])
                        #     self.dexarm.move_down_meat(self.ser_micro)
                        # for i in range (0, len(self.marbling_pts)):
                        #     self.dexarm.move_to_point_position(self.marbling_pts[i][0], self.marbling_pts[i][1])
                        #     self.dexarm.move_down_meat(self.ser_micro)
                        # self.dexarm.move_to_point_position(self.muscle_pts[0][0],self.muscle_pts[0][1])

                            
                    # ----------------------------------------
                    # STEP 4: Move downward to sample and hold
                    # ----------------------------------------
                    case "MOVE_DOWN_TO_SAMPLE_COORD":
                        self.dexarm.move_down_meat(self.ser_micro)

                    # ------------------------------------------------
                    # STEP 5: (a) Move to vial and (b) Dispense liquid
                    # ------------------------------------------------
                    case "SAMPLE_DROPOFF":
                        self.dexarm.step_5_dispense_sample(self.curr_vial)
                        self.curr_vial += 1

                    # ------------------------------------------------------------
                    # STEP 6: (a) Move to dispose cup and (b) Drop off pipette tip
                    # ------------------------------------------------------------
                    case "TIP_DISPOSAL":
                        self.dexarm.step_6_move_to_dispose_cup()

                    # UNKNOWN case, should never happen
                    case _:
                        print("UNKNOWN STEP")
                
                # -------------------------------------
                # UPDATE STATUS of last completed step
                # -------------------------------------
                self.last_completed_step = step

            # After collecting sample, go home
            self.dexarm.go_new_home()

        self.complete_success = True
        self.finish()

    def finish(self):
        if self.complete_success:
            self.progress.emit("ROBOT_STOP_SUCCESS")
        else:
            # If not completed successuflly need to do some clean up steps
            self.progress.emit("ROBOT_STOP")
        self._stop_requested = False

        if not self.use_robot_simulator:
            self.dexarm.close()
        self.ser_micro.close()

    
    def stop(self):
        # Set flag to be handled by main run function
        if self.isRunning():
            self._stop_requested = True

    def collect_photo(self):
        """
        Collect photo and start coordinate selection from main thread (automatic or manual)
        """
        # Move to photo position:
        if not self.use_robot_simulator:
            photograph_offset_y = 30
            photograph_offset_z = 110
            self.dexarm.go_new_home()
            self.dexarm.move_to_photograph_position(photograph_offset_y, photograph_offset_z)
            time.sleep(4)

            #move_to(0, self.dexarm.y_home+photograph_offset_y, photograph_offset_z)

             
            
            # TODO might need to wait a bit before img is taken - to stabilize
        else:
            # Imitate robot movement to photo position
            time.sleep(2)

        # Take photo
        if self.use_simulated_sample: 
            simulated_img = cv2.imread(rel_path("simulated_img.png"))
            print(simulated_img)
            cv2.imwrite(ImageWorker.img_path, simulated_img)
        else:
            take_photo(ImageWorker.img_path)

        if self.sample_mode == "Manual":
            self.progress.emit("START_MANUAL_POINT_SELECTION")
        elif self.sample_mode == "Automatic":
            self.progress.emit("START_AUTOMATIC_POINT_SELECTION")

    def wait_for_coord_results(self):
        """ Wait for results for specific timeout amount """

        timeout = 10 if self.sample_mode == "Automatic" else 100
        if not self.main_window.image_coords_ready_event.wait(timeout=timeout):
            self.progress.emit("Error: Image coord result not received in time")
            return -1
        
        # Clear event
        self.main_window.image_coords_ready_event.clear()
        
        # Update progress/status
        self.progress.emit("Robot Worker Received Coordinates!!!")
        self.image_processing_complete = True

        # ----------------------------
        # Parse selected sample points
        # ----------------------------
        self.muscle_pts = self.main_window.selected_points["muscle_points"]
        self.marbling_pts = self.main_window.selected_points["marbling_points"]

        # Each entry is a list of coords ---> For one pipette tip
        self.sample_list = [self.muscle_pts, self.marbling_pts]

        # Edit later, need something in the GUI
        # Total num samples = number of pipettes
        self.total_num_samples = 2 #len(self.muscle_pts) + len(self.marbling_pts)

        # Check that total number of points is greater than 0 / Additional check of STOP interrupt
        if self.total_num_samples < 1 or self._stop_requested:
            self.progress.emit("Error: total number of samples is less than 1") if self.total_num_samples < 1 else self.progress.emit("STOP_INTERRUPT")
            return -1

    def isOverMaxTipNum(self):
        """ Check current tip within bounds """
        return self.curr_tip > self.max_tip_num
    
    def isOverMaxVialNum(self):
        """ Check current vial within bounds """
        return self.curr_vial > self.max_vial_num