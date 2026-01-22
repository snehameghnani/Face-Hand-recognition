import cv2

# Try to open the default webcam (index 0)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not detected! Make sure it's connected or not in use.")
else:   
    ret, frame = cap.read()
    if ret:
        print("Camera is working! Frame captured successfully.")
    else:
        print("Camera opened, but frame not captured.")
    cap.release()

 

