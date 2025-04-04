from pydexarm import Dexarm
import time

def main():
    dexarm = Dexarm(port="COM3")

    # Step 1: At initiation, always go home first
    dexarm.go_home()
    
    dexarm.move_to(0, 200, 0)
    dexarm.fast_move_to(50, 200, 0) # x, y, z coords
    dexarm.fast_move_to(x=50, y=200, z=-20)





    # Final Step: Close the serial port
    dexarm.close()
    pass

if __name__ == '__main__':
    main()