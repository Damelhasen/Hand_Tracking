import cv2


webcam = cv2.VideoCapture(0)
print("Webcam Found")

while  True :
    ret,frame = webcam.read()
    if ret == True:
       
        cv2.imshow("Webcam",frame)
        key =cv2.waitKey(1)
        if key == ord('q'):
            print("Webcam Stopped")
            break
    
         