from pydexarm import Dexarm
import serial
import serial.tools.list_ports
import time

def id_serial_ports():
    # Identify COM ports for microcontroller and Rotrics arm
    returned_txt = ''
    msg_list = ['R\r', 'M1112\r']
    ser_micro_find = None
    dexarm = None
    ports = serial.tools.list_ports.comports()
    print('Searching for COM ports')

    for port in ports:
        found_port = False
        for msg in msg_list:
            try:
                ser = serial.Serial(port=port.device, baudrate=115200, timeout=0.1)
                time.sleep(1)
                ser.flushInput()
                ser.write(bytes(msg, 'utf-8'))
                returned_txt = ser.readline()
                ser.flushInput()

                if returned_txt == b'R here \n':
                    ser_micro_find = ser
                    print(f'Found microcontroller at {port.device}')
                    found_port = True
                    ser.close()
                elif returned_txt == b'M1112\r\n':
                    ser.close()
                    dexarm = Dexarm(port=port.device)
                    print(f'Found Rotrics Dexarm at {port.device}')
                    found_port = True
                else:
                    ser.close()
            except serial.SerialException as e:
                print(f'Error opening serial port: {e}')

            except Exception as e:
                print(f'Unexpected error during serial port initialization: {e}')

            if found_port:
                break
    return ser_micro_find, dexarm


def ser_close(ser_micro, dexarm):
    ser_list = [ser_micro, dexarm]
    for ser in ser_list:
        try:
            ser.close()
        except AttributeError as e:
            print(f'Error closing port: {e}')


def move_down(ser_micro, dexarm, z_pos, step_size):
    connect = True
    received_msg = ''
    ser_micro.flushInput()
    ser_micro.write(bytes('D', 'utf-8'))
    dexarm.move_to(None, None, z_pos, mode='G0')
    move_step = 0.1
    print('Checking Connection')
    while connect:
        received_msg = ser_micro.readline()
        ser_micro.flushInput()
        dexarm.move_to(x=None, y=None, z=z_pos, e=None, feedrate=2000, mode='G1', wait=True)
        z_pos = z_pos - move_step
        if received_msg == b'DISCONNECTED\r\n':
            connect = False
            print('Disconnected')

    # Maybe need to pause for 5 seconds over here


# Main
if __name__ == '__main__':
    ser_micro, dexarm = id_serial_ports()
    # ser_micro = serial.Serial(port='COM4', baudrate=115200, timeout=0.1)
    # dexarm = Dexarm(port="COM6")
    ser_micro.open()

    dexarm.move_down_meat(ser_micro)
    # move_down(ser_micro, dexarm, z_home, step_size)
    # for i in range(0, 10):
    #    dexarm.move_to(None, None, 40)
    #    move_down(ser_micro, dexarm)  # move dexarm down until microcontroller signals broken contact
    #    time.sleep(5)

    ser_close(ser_micro, dexarm)
