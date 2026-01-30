import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

webcam = cv2.VideoCapture(0)
print("Webcam Found")

while  True :
    ret,frame = webcam.read()
    if ret == True:
        #applying hand tracking
        cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        results = mp_hands.Hands.process(frame)
        cv2.imshow("Webcam",frame)
        
        #draw stuff on hands
        cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame,hand_landmarks,connections = mp_hands.HAND_CONNECTIONS)
        
        key =cv2.waitKey(1)
        if key == ord('q'):
            print("Webcam Stopped")
            break
    
         