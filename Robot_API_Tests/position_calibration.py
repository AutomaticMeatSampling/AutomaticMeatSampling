from pydexarm import Dexarm
import time
import cv2
# from takepicture import take_photo
import serial
import serial.tools.list_ports

if __name__ == '__main__':
    dexarm = Dexarm(port="COM6")
    ser_micro = serial.Serial(port='COM4', baudrate=115200, timeout=0.1)

    # Find Pipette Tip Position
    print('Turn off dexarm and move to position above pipette tip')
    print('Turn on dexarm once you have positioned it above pipette tip')
    x = input('Enter Y and press enter when done')
    if x == 'Y':
        x_curr, y_curr, z_curr, *_ = dexarm.get_current_position()
        print(f" actual X: {x_curr}")
        print(f" actual Y: {y_curr}")
        print(f" actual Z : {z_curr}")
