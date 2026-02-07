import pygame 
from pynput.keyboard import Key, Controller
import cv2
import mediapipe as mp
import pickle
import numpy as np
import time
pygame.init() 

keyboard = Controller()

screen_width = 1280
screenheight = 720
win = pygame.display.set_mode((screen_width, screenheight)) 
pygame.display.set_caption("Hand Tracking Game") 

x = 200
y = 200

width = 40
height = 40

vel = 10
run = True

############### Hand tracking code################

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: "up", 1: "down", 2: "right", 3: "left"}

while run:
	predicted_character = None
	
	data_aux = []
	x_ = []
	y_ = []

	ret, frame = cap.read()
	frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	results = hands.process(frame_rgb)
	if results.multi_hand_landmarks:
		for hand_landmarks in results.multi_hand_landmarks:
			for i in range(len(hand_landmarks.landmark)):
				hand_x = hand_landmarks.landmark[i].x
				hand_y = hand_landmarks.landmark[i].y
				x_.append(hand_x)
				y_.append(hand_y)

			for i in range(len(hand_landmarks.landmark)):
				hand_x = hand_landmarks.landmark[i].x
				hand_y = hand_landmarks.landmark[i].y
				data_aux.append(hand_x - min(x_))
				data_aux.append(hand_y - min(y_))

		prediction = model.predict([np.asarray(data_aux)])
		predicted_character = labels_dict[int(prediction[0])]

	pygame.time.delay(10)

	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			run = False

	keys = pygame.key.get_pressed()

	if keys[pygame.K_LEFT] and x > 0:
		x += vel

	if keys[pygame.K_RIGHT] and x < screen_width:
		x -= vel

	if keys[pygame.K_UP] and y > 0:
		y -= vel

	if keys[pygame.K_DOWN] and y < screenheight:
		y += vel

	if predicted_character == "down" and y < screenheight:
		y += vel
		print("down")
		 
		
	elif predicted_character == "up" and y > 0:
		y -= vel
		print("up")
		
	elif predicted_character == "left" and x > 0:
		x -= vel
		print("left")
		
	elif predicted_character == "right" and x < screen_width:
		x += vel
		print("right")
		
    
	win.fill((0, 0, 0))
	pygame.draw.rect(win, (255, 0, 0), (x, y, width, height))
	pygame.display.update()

cap.release()
pygame.quit()