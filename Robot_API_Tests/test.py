from pydexarm import Dexarm
import time
import cv2
from takepicture import take_photo
import serial
import serial.tools.list_ports

def main():
    dexarm = Dexarm(port="COM3")
    ser_micro = serial.Serial(port='COM4', baudrate=115200, timeout=0.1)

    # Step 1: At initiation, always go home first
    # dexarm.go_home()
    
    # dexarm.move_to(0, 200, 0)
    # dexarm.fast_move_to(50, 200, 0) # x, y, z coords
    # dexarm.fast_move_to(x=50, y=200, z=-20)

    # x_curr, y_curr, z_curr, *_ = dexarm.get_current_position()
    # print(f" actual X: {x_curr}") 
    # print(f" actual Y: {y_curr}") 
    # print(f" actual Z : {z_curr}")
    # dexarm.fast_move_to(None, None, 120)
    # dexarm.go_home()
    # dexarm.fast_move_to(None, None, 120)
    x_curr, y_curr, z_curr, *_ = dexarm.get_current_position()
    print(f" actual X: {x_curr}") 
    print(f" actual Y: {y_curr}") 
    print(f" actual Z : {z_curr}")
    print(dexarm.get_module_type())

    # dexarm.step_5_dispense_sample(1)
    

    # Initalize pins to be output pins
    # Leftmost Pin: 17
    # Right Pin: 18
    dexarm._send_cmd("M42 P17 M1\r")
    dexarm._send_cmd("M42 P18 M1\r")


    # Step 1: Go to photograph position and take photo
    photograph_offset_y = -40
    photograph_offset_z = 110

    dexarm.go_home()
    dexarm.move_to(0, dexarm.y_home+photograph_offset_y, photograph_offset_z)

    # time.sleep(3)

    # Step 2: Collect pipette tip
    dexarm.fast_move_to(None, None, 120)
    dexarm.move_to_pipette_tip(1)

    # time.sleep(3)

    # x_curr, y_curr, z_curr, *_ = dexarm.get_current_position()
    # print(f" actual X: {x_curr}") 
    # print(f" actual Y: {y_curr}") 
    # print(f" actual Z : {z_curr}")

    dexarm.step_3_move_to_solvent()


    # Step 4: Move to temporary sample location
    sample_loc_x = -7
    sample_loc_y = 185
    dexarm.move_to(0,187, 45)
    # time.sleep(3)
    dexarm.move_down_meat(ser_micro)


    # Step 5: Dispense liquid
    dexarm.step_5_dispense_sample(1)

    # Step 6: Drop off pipette
    dexarm.step_6_move_to_dispose_cup()

    # dexarm.go_home()

    # dexarm.toggle_gpio_pin(8)


# TEST IMAGE STUFF
    # photograph_offset_y = -40
    # photograph_offset_z = 100

    # dexarm.go_home()
    # dexarm.move_to(0, dexarm.y_home+photograph_offset_y, photograph_offset_z)

    # time.sleep(2)

    # take_photo("test_photo.png")




    # Final Step: Close the serial port
    dexarm.close()
    pass

if __name__ == '__main__':
    main()
