#Imports
from pydexarm import Dexarm
import time
import cv2
import os
from takepicture import take_photo
import serial
import serial.tools.list_ports
import threading
import sys
from pathlib import Path

#Import other folders' code
repo_root = Path(__file__).resolve().parent.parent  # Repo/
seg_folder = repo_root / "Segmentation_Algorithms"
sys.path.append(str(seg_folder))

# Image processing definitions
use_img_processing = False

if use_img_processing:
    import select_sample_point
    import segment_ld

processing_done = threading.Event() # image processing done signal

result_holder = {}

def image_processing_thread(predictor, img_path, num_marbling_pts=1, num_muscle_pts=0):
    print("[IMAGE PROCESSING] Starting image processing...")
    processing_done.clear()
    start = time.time()

    # Begin image processing code
    muscle_points, marbling_points = select_sample_point.segment_and_select_points(img_path, predictor, num_marbling_pts=num_marbling_pts, num_muscle_pts=num_muscle_pts, show=False)

    result_holder["muscle_points"] = muscle_points
    result_holder["marbling_points"] = marbling_points

    end_time = time.time() - start

    print(f"[IMAGE PROCESSING] Processing Complete at {end_time} seconds !!!!")

    processing_done.set()

##########################################################
##########################################################

def main():

    ######################################################
    ######################################################
    # Initialization
    ######################################################
    ######################################################

    pipette_pin = 1

    # Load predictor only one time
    if use_img_processing:
        predictor = segment_ld.load_sam_model()

    # Detect OS for COM port determination
    if os.name == "nt":#Windows
        dexarm = Dexarm(port="COM3")
        ser_micro = serial.Serial(port='COM4', baudrate=115200, timeout=0.1)
    else: #Mac/Linux
        dexarm = Dexarm(port="/dev/cu.usbmodem207B396A36311")#COM3
        ser_micro = serial.Serial(port='/dev/cu.usbmodem14301', baudrate=115200, timeout=0.1)

    num_marbling_pts = 1
    num_muscle_pts = 0

    dexarm.report_coordinates()
    print(dexarm.get_module_type())

    # Initalize pins to be output pins
    # Leftmost Pin: 17
    # Right Pin: 18
    dexarm._send_cmd("M42 P17 M1\r")
    dexarm._send_cmd("M42 P18 M1\r")

    ######################################################
    ######################################################
    # Step 1: Go to photograph position and take photo
    ######################################################
    ######################################################

    #   For considering pipette to camera offset
    photograph_offset_y = -40
    photograph_offset_z = 110

    dexarm.go_home()
    dexarm.move_to_photograph_position(photograph_offset_y, photograph_offset_z)

    img_path = "meat_sample.png"
    if use_img_processing:
        time.sleep(2)
        take_photo(img_path)
        vision_thread = threading.Thread(
            target=image_processing_thread,
            args=(predictor, img_path, num_marbling_pts, num_muscle_pts,),
            daemon=True
        )
        vision_thread.start()

    ######################################################
    ######################################################
    # Step 2: Collect pipette tip
    ######################################################
    ######################################################

    dexarm.fast_move_to(None, None, 120)
    dexarm.move_to_pipette_tip(pipette_pin)

    ######################################################
    ######################################################
    # Step 3: Collect solvent
    ######################################################
    ######################################################

    dexarm.step_3_move_to_solvent()

    ######################################################
    ######################################################
    # Step 4: Move to sample location
    ######################################################
    ######################################################

    if use_img_processing:
        processing_done.wait()
        marbling_points = result_holder["marbling_points"]
        muscle_points = result_holder["muscle_points"]
        # TODO calculate actual movement stuff here
        print(marbling_points)
        print(muscle_points)

    sample_loc_x = -7 #in pixels? 
    sample_loc_y = 185

    dexarm.move_to_point_position(sample_loc_x, sample_loc_y) #this has to be in pixels
    dexarm.move_down_meat(ser_micro)

    ######################################################
    ######################################################
    # Step 5: Dispense liquid
    ######################################################
    ######################################################

    dexarm.step_5_dispense_sample(2)

    ######################################################
    ######################################################
    # Step 6: Drop off pipette
    ######################################################
    ######################################################
    
    dexarm.step_6_move_to_dispose_cup()


# TEST IMAGE STUFF
    # photograph_offset_y = -40
    # photograph_offset_z = 100

    # dexarm.go_home()
    # dexarm.move_to(0, dexarm.y_home+photograph_offset_y, photograph_offset_z)

    # time.sleep(2)

    # take_photo("test_photo.png")

    ######################################################
    ######################################################
    # Final Step: Close the serial port
    ######################################################
    ######################################################
    
    dexarm.close()
    pass

if __name__ == '__main__':
    main()
    # for i in range(1,7):
    #     main(i)

