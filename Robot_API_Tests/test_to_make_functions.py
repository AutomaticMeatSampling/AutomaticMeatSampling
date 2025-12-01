from pydexarm import Dexarm
import time
import cv2
from takepicture import take_photo
import serial
import serial.tools.list_ports
import os

# Note for taking pictures, change your directory to where you want them saved
#set pipette offset to zero if you want to center the camera

# Initialize webcam (0 = default camera)
cam = cv2.VideoCapture(0)

def main():

    if os.name == "nt":#Windows
        dexarm = Dexarm(port="COM3")
    else: #Mac/Linux
        dexarm = Dexarm(port="/dev/cu.usbmodem207B396A36311")#COM3

    # Step 1: At initiation, always go home first

    dexarm.go_home()
    dexarm.move_to_photograph_position()

    # Capture one frame AT PHOTOGRAPHING POINT
    time.sleep(5)
    take_photo("photograph_posi_img.png")

    #report position at photograph
    dexarm.report_coordinates()

    # Step 2: Movement (see code chunk of variables above)
    dexarm.move_to_point_position(1620, 558)
    
    

    # Final Step: Close the serial port
    dexarm.close()
    pass

if __name__ == '__main__':
    main()
