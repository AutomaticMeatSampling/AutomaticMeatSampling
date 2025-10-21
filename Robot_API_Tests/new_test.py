from pydexarm import Dexarm
import time

def main():
    dexarm = Dexarm(port="/dev/cu.usbmodem207B396A36311")#COM3
    y_home = 300;#DO NOT CHANGE


    #55 pix x 55 pix y, dimensions: 2372 × 1582

    #########################

    # CONSTANTS 
    
    INCHES_TO_MM = 25.4
    CAMERA_WIDTH_PX = 2372
    CAMERA_HEIGHT_PX = 1582

    grid_inches = 0.25 #inches
    grid_mm = grid_inches*INCHES_TO_MM

    photograph_offset = 40 #y offset (was -40)

    



    #########################

    grid_pixels_x = 55
    grid_pixel_y = 55

    x_camera_center_offset_pixel = CAMERA_WIDTH_PX/2
    y_camera_center_offset_pixel = CAMERA_HEIGHT_PX/2 #POSITIVE IS NEGATIVE and vice versa for additional offsets

    x_mm_per_pixel = grid_mm/grid_pixels_x
    y_mm_per_pixel = grid_mm/grid_pixel_y

    ## INPUT HERE
    dot_x_pixel = 1610;#test value
    dot_y_pixel = 252;#test value
    ## INPUT HERE

    pipette_offset_x = 0*x_mm_per_pixel
    pipette_offset_y = 606*y_mm_per_pixel

    pixel_move_x_mm = ((dot_x_pixel - x_camera_center_offset_pixel)*x_mm_per_pixel) + pipette_offset_x;
    pixel_move_y_mm = -((dot_y_pixel - y_camera_center_offset_pixel)*y_mm_per_pixel) - pipette_offset_y;

    if dot_x_pixel > x_camera_center_offset_pixel:
        rotation_mode = 'CW'
    else:
        rotation_mode = 'CCW'

    # Step 1: At initiation, always go home first

    dexarm.go_home()
    dexarm.move_to(0, y_home+photograph_offset, 150)#photographing position

    #note: maximum to move downwards like that is y = -70, z = -80

    dexarm.fast_move_to(0, y_home+photograph_offset, 0)
    dexarm.move_inward_to_target(pixel_move_x_mm, y_home + photograph_offset + pixel_move_y_mm, -80, rotation_mode) #need to check if value is left or right to say clockwise or counterclockwise

    # Final Step: Close the serial port
    dexarm.close()
    pass

if __name__ == '__main__':
    main()