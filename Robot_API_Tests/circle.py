from pydexarm import Dexarm
import time
import cv2 

# Initialize webcam (0 = default camera)
cam = cv2.VideoCapture(0)

def main():
    dexarm = Dexarm(port="/dev/cu.usbmodem207B396A36311")#COM3
    y_home = 300;#DO NOT CHANGE
    
    # Step 1: At initiation, always go home first

    dexarm.go_home()
    
    for x in range(10):
        #dexarm.move_to(100, y_home, 0)

        dexarm._send_cmd("G2 X100 Y400 R50\r")#this is NOT x100 y100


        #dexarm.move_inward_to_target(100, y_home+100)
        #dexarm.move_to(100, y_home+100, 0)
        time.sleep(2)

        # Capture one frame
        ret, frame = cam.read()

        if ret:      
            cv2.imwrite(f"captured_image{x}.png", frame)       
        else:
            print("Failed to capture image.")

        dexarm.go_home()

    # Final Step: Close the serial port
    cam.release() 
    dexarm.close()
    pass

if __name__ == '__main__':
    main()





