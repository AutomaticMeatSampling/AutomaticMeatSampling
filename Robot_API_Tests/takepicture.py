import cv2 
import time
import os

# Initialize webcam (0 = default camera)
# cam = cv2.VideoCapture(0)

# Capture one frame
# ret, frame = cam.read()

# if ret:      
#     cv2.imwrite("captured_image.png", frame)
# else:
#     print("Failed to capture image.")

# cam.release() 

def take_photo(save_path):

    if os.name == "nt":#Windows
        cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    else: #Mac/Linux
        cam = cv2.VideoCapture(0)

    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 3264)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 2448)

    time.sleep(1)

    # Warm up the camera (important for high-res!)
    for _ in range(5):
        print("Photo!")
        cam.read()

    # Capture one frame
    ret, frame = cam.read()

    if ret:        
        cv2.imwrite(save_path, frame)  
    else:
        print("Failed to capture image.")

    cam.release() 
