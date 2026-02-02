import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

position1 = mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP 
webcam = cv2.VideoCapture(0)
print("Webcam Found")

# Create a single Hands instance outside the loop
with mp_hands.Hands(
    model_complexity=0,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
) as hands:
    while True:
        

        ret, frame = webcam.read()
        if ret:
            # Convert frame to RGB for processing
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Apply hand tracking
            results = hands.process(frame_rgb)
            
            # Convert back to BGR for display
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            
            # Draw hand landmarks if detected
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame_bgr,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
                    
                    # get index finger tip normalized coordinates and convert to pixel coords
                    h, w, _ = frame_bgr.shape
                    idx_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP] 
                    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                    rndthumb = round((thumb_tip.y *10),2)
                    rndindex = round((idx_tip.y *10),2)
                    print(rndindex, rndthumb)
                        
                    #position = (int(idx_tip.x * w), int(idx_tip.y * h))
                    #print(position)
                   
                    
            
            cv2.imshow("Webcam", frame_bgr)
            
            key = cv2.waitKey(1)
            if key == ord('q'):
                print("Webcam Stopped")
                break

webcam.release()
cv2.destroyAllWindows()