from main import load_model
import numpy as np
import cv2
from time import sleep
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils

IMAGE_HEIGHT = 460
IMAGE_WIDTH = 680
NUM_LANDMARKS = 21

model, initPos, labels, annotations = load_model()

def print_landmarks(positions, image):
    for landmark in positions:
        x = landmark[0]
        y = landmark[1]

        pixelCoordinates = mp_drawing._normalized_to_pixel_coordinates(x, y, IMAGE_WIDTH, IMAGE_HEIGHT)

        cv2.circle(image, pixelCoordinates, 5, (0, 255, 0), -1)
    
    #return image

blankScreen = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), np.uint8)
currentPositions = np.empty((NUM_LANDMARKS, 3))
for gesture in range(0, len(model)):
    input("Press enter to see " + annotations[labels[gesture]])
    currentPositions = initPos[gesture]
    print_landmarks(currentPositions, blankScreen)
    for frame in model[gesture]:
        sleep(1)
        cv2.imshow("model", blankScreen)
        key = cv2.waitKey(1000)
        
        blankScreen = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), np.uint8)

        for landmark in range(0, NUM_LANDMARKS):
            xDist = frame[landmark][0]
            yDist = frame[landmark][1]
            zDist = frame[landmark][2]

            currentPositions[landmark][0] += xDist
            currentPositions[landmark][1] += yDist
            currentPositions[landmark][2] += zDist

        print_landmarks(currentPositions, blankScreen)