from pydexarm import Dexarm
import time
import cv2 
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

    y_home = 300;#DO NOT CHANGE

    ## INPUT HERE
    dot_x_pixel = 1620;#test value  1774, 1862, 1494, 810, 878
    dot_y_pixel = 558;#test value   842, 584,   530, 758, 482
    ## INPUT HERE

    #55 pix x 55 pix y, dimensions: 2372 × 1582

    #########################

    # SEMI CONSTANTS (change if image size does)

    #CAMERA_WIDTH_PX = 1920  #so the dimensions of this change sometimes. make it so it reads the image to get the dimensions from that first
    #CAMERA_HEIGHT_PX = 1080
    CAMERA_WIDTH_PX = 2372
    CAMERA_HEIGHT_PX = 1582

    grid_pixels_x = 55#39
    grid_pixel_y = 55#39

     # CONSTANTS 
    
    INCHES_TO_MM = 25.4 #true constant
    grid_inches = 0.25 #inches
    grid_mm = grid_inches*INCHES_TO_MM

    photograph_offset = 40 #y offset (was -40)

    #########################

    

    x_camera_center_offset_pixel = CAMERA_WIDTH_PX/2
    y_camera_center_offset_pixel = CAMERA_HEIGHT_PX/2 #POSITIVE IS NEGATIVE and vice versa for additional offsets

    x_mm_per_pixel = grid_mm/grid_pixels_x
    y_mm_per_pixel = grid_mm/grid_pixel_y

    pipette_offset_x = 0#-5*x_mm_per_pixel
    pipette_offset_y = 0#609*y_mm_per_pixel #606

    pixel_move_x_mm = ((dot_x_pixel - x_camera_center_offset_pixel)*x_mm_per_pixel) + pipette_offset_x;
    pixel_move_y_mm = -((dot_y_pixel - y_camera_center_offset_pixel)*y_mm_per_pixel) - pipette_offset_y;

    if dot_x_pixel > x_camera_center_offset_pixel:
        rotation_mode = 'CW'
    else:
        rotation_mode = 'CCW'

    # Step 1: At initiation, always go home first

    for x in range(10):

        dexarm.go_home()
        dexarm.move_to(0, y_home+photograph_offset, 150)#photographing position
        #note: maximum to move downwards like that is y = -70, z = -80

        # Capture one frame AT PHOTOGRAPHING POINT
        time.sleep(5)
        ret, frame = cam.read()

        if ret:      
            cv2.imwrite(f"photograph_posi_img{x}.png", frame)       
        else:
            print("Failed to capture image.")

        #report position at photograph
        x_curr, y_curr, z_curr, *_ = dexarm.get_current_position()
        print(f" actual X: {x_curr}") 
        print(f" actual Y: {y_curr}") 
        print(f" actual Z : {z_curr}")

        # Step 2: Movement
        dexarm.fast_move_to(0, y_home+photograph_offset, 0)
        dexarm.move_inward_to_target(pixel_move_x_mm, y_home + photograph_offset + pixel_move_y_mm, -80, rotation_mode) 
       
        # Capture one frame AT MOVED POINT
        time.sleep(5)
        ret, frame = cam.read()

        if ret:      
            cv2.imwrite(f"point_posi_img{x}.png", frame)       
        else:
            print("Failed to capture image.")

        dexarm.go_home()

        x_curr, y_curr, z_curr, *_ = dexarm.get_current_position()
        print(f" actual X: {x_curr}") 
        print(f" actual Y: {y_curr}") 
        print(f" actual Z : {z_curr}")

    # Final Step: Close the serial port
    dexarm.close()
    pass

if __name__ == '__main__':
    main()
