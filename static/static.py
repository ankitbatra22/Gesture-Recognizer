import cv2
import mediapipe as mp
from threading import Thread
import numpy as np
from numpy.core.fromnumeric import shape
from numpy.core.records import record
import math
import json
import os
from sklearn.neighbors import KNeighborsClassifier 

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
TOTAL_LANDMARKS = 21
TOTAL_DISTANCES = 5
LABELS_FILE = "labels.json"
MODEL_FILE = "model.npy"
INIT_POS_FILE = "init.npy"
pointsHistory = np.empty(shape=(0, TOTAL_LANDMARKS, 3))

# Booleans to hold request states for starting record of datapoints & starting analysis mode
recordRequest = False
analysisMode = False

def video_runner(cap: np.ndarray, frameWidth: int, frameHeight: int, gestures: np.ndarray, labels: list, annotations: list):
    global mp_hands
    global mp_drawing
    global TOTAL_LANDMARKS
    global recordRequest
    global pointsHistory
    global analysisMode
    
    with mp_hands.Hands(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:


        while cap.isOpened():
            success, image = cap.read()

            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue
            
            # Flip the image horizontally for a later selfie-view display, and convert
            # the BGR image to RGB.
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            results = hands.process(image)

            # Draw the hand annotations on the image.
            image.flags.writeable = True
            pointsList = np.empty(shape=(TOTAL_LANDMARKS, 3))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            if results.multi_hand_landmarks and len(results.multi_hand_landmarks[0].landmark) == TOTAL_LANDMARKS:
                if analysisMode and not recordRequest:
                    recordRequest = True
                hand = results.multi_hand_landmarks[0]

                for point in range(0, len(hand.landmark)):
                    pixelCoordinates = mp_drawing._normalized_to_pixel_coordinates(hand.landmark[point].x, hand.landmark[point].y, frameWidth, frameHeight)
                    cv2.circle(image, pixelCoordinates, 5, (0, 255, 0), -1)
                    image = cv2.putText(image, str(point) + ": " + str(round(hand.landmark[point].z, 2)), pixelCoordinates, cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 255, 0), 1)
                    pointsList[point] = (hand.landmark[point].x, hand.landmark[point].y, hand.landmark[point].z)
                
                if recordRequest:
                    pointsHistory = np.append(pointsHistory, [pointsList], 0)
                    print(pointsHistory.shape)
            elif analysisMode and recordRequest:
                recordRequest = False
                analyze_gesture_with_5d_knn(pointsHistory, gestures, labels, annotations)
                pointsHistory = np.empty(shape=(0, TOTAL_LANDMARKS, 3))

            cv2.imshow('MediaPipe Hands', image)
        
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") :
                break
    
    return

def develop_3d_knns(gestures, labels):
    landmarkModels = []
    for landmark in range(0, TOTAL_LANDMARKS):
        frameModels = []
        for frame in range(0, TOTAL_DISTANCES):
            xTrain = []
            yTrain = []
            for gesture in range(0, len(gestures)):
                point = gestures[gesture][frame][landmark]
                label = labels[gesture]
                xTrain.append(point)
                yTrain.append(label)
            print(np.array(xTrain).shape)
            print(len(yTrain))
            knn = KNeighborsClassifier(n_neighbors=3, metric="euclidean")
            knn.fit(xTrain, yTrain)
            frameModels.append(knn)
        landmarkModels.append(frameModels)
    
    return landmarkModels
    

def analyze_gesture_with_3d_knn(pointsHistory: np.ndarray, gestures: np.ndarray, labels: list, annotations: list):
    if len(pointsHistory) < 5:
        print("Could not get enough points")
        return None
    
    handMovement = perform_distance_calc(pointsHistory)

    predictions = []
    models = develop_3d_knns(handMovement, gestures, labels, annotations)

    for frame in range(0, TOTAL_DISTANCES):
        for landmark in range(0, TOTAL_LANDMARKS):
            prediction = models[landmark][frame].predict([handMovement[frame][landmark]])
            predictions.append(max(set(prediction), key=prediction.tolist().count))
    print(predictions)
    gesturePred = max(set(predictions), key=predictions.count)
    print(annotations[gesturePred])
    print("Accuracy: " + str(predictions.count(gesturePred)/len(predictions)))


def analyze_gesture_with_5d_knn(pointsHistory: np.ndarray, gestures: np.ndarray, labels: list, annotations: list):
    if len(pointsHistory) < 5:
        print("Could not get enough points")
        return None
    
    handMovement = perform_distance_calc(pointsHistory)
    inputPoints, ignore = produce_5d_points([handMovement])

    modelPoints, modelY = produce_5d_points(gestures, labels)
    print(modelPoints.shape)

    knn = KNeighborsClassifier(n_neighbors=3, metric="euclidean")
    knn.fit(modelPoints, modelY)
    prediction: list = knn.predict(inputPoints)
    print(prediction)
    gesturePred = max(set(prediction), key=prediction.tolist().count)
    print(annotations[gesturePred])
    print("Accuracy: " + str(prediction.tolist().count(gesturePred)/len(prediction)))

def produce_5d_points(model: np.ndarray, labels=None):
    points = []
    updatedLabels = []
    for gesture in range(0, len(model)):
        for frame in range(0, len(model[gesture])):
            for landmark in range(0, len(model[gesture][frame])):
                point = model[gesture][frame][landmark]
                x = point[0]
                y = point[1]
                z = point[2]
                points.append((frame*5, landmark*5, x, y, z))
                if labels:
                    updatedLabels.append(labels[gesture])
    
    return np.array(points), updatedLabels



def recording_worker(cap: np.ndarray, gestures: np.ndarray, labels: list, annotations: list, initPos: np.ndarray):
    global recordRequest
    global pointsHistory
    global analysisMode

    prevName = ""

    while True:
        startRecord = input("Begin recording a gesture? (Y/N) ")
        if startRecord == "N":
            break
        else:
            recordRequest = True
        
        input("Press enter to stop recording...")
        recordRequest = False
        
        if len(pointsHistory) < 5:
            print("Could not get enough points!")
            pointsHistory = np.empty(shape=(0, TOTAL_LANDMARKS, 3))
            continue
        
        initPos = np.append(initPos, [pointsHistory[0]], 0)

        newGesture = perform_distance_calc(pointsHistory)
        gestures = np.append(gestures, [newGesture], 0)
        print(gestures.shape)

        name = input("Enter name of gesture or press enter for previous: ")

        if name == "":
            if not prevName:
                if len(labels) == 0:
                    raise ValueError("You did not enter a valid name")
                prevName = annotations[labels[-1]]
            name = prevName
        
        if name in annotations:
            labels.append(annotations.index(name))
        else:
            annotations.append(name)
            labels.append(len(annotations)-1)

        save_model(gestures, initPos, labels, annotations)
        prevName = name

        pointsHistory = np.empty(shape=(0, TOTAL_LANDMARKS, 3))
    
    print("Initiating analysis mode...")
    analysisMode = True
    input("Press enter when done...")
    analysisMode = False

    cap.release()
    cv2.destroyAllWindows()

    return

def perform_distance_calc(pointsHistory: np.ndarray):
    distances = np.empty(shape=(TOTAL_DISTANCES, TOTAL_LANDMARKS, 3))
    increment = math.floor(len(pointsHistory)/TOTAL_DISTANCES)
    
    fill = 0

    for i in range(increment, len(pointsHistory), increment):
        if fill == TOTAL_DISTANCES-1:
            break
        distancePoints = np.empty(shape=(TOTAL_LANDMARKS, 3))
        currPoints = pointsHistory[i]
        prevPoints = pointsHistory[i-increment]
        
        for j in range(0, len(currPoints)):
            xDist = currPoints[j][0] - prevPoints[j][0]
            yDist = currPoints[j][1] - prevPoints[j][1]
            zDist = currPoints[j][2] - prevPoints[j][2]
            
            distancePoints[j] = (xDist, yDist, zDist)

        distances[fill] = distancePoints
        fill += 1

    return distances

def load_model():
    global LABELS_FILE
    global MODEL_FILE
    global INIT_POS_FILE

    labels = []
    annotations = []
    model = np.empty(shape=(0, TOTAL_DISTANCES, TOTAL_LANDMARKS, 3))
    initPositions = np.empty(shape=(0, TOTAL_LANDMARKS, 3))

    if (os.path.exists(LABELS_FILE)):
        with open(LABELS_FILE) as f:
            data = json.load(f)
            if "labels" in data:
                labels = data["labels"]
            if "annotations" in data:
                annotations = data["annotations"]
    else:
        f = open(LABELS_FILE, "w")
        json.dump({}, f)
        f.close()
    
    if (os.path.exists(MODEL_FILE)):
        with open(MODEL_FILE) as f:
            model = np.load(MODEL_FILE)
    
    if (os.path.exists(INIT_POS_FILE)):
        with open(INIT_POS_FILE) as f:
            initPositions = np.load(INIT_POS_FILE)
    
    print("Loaded a model of shape: " + str(model.shape))
    print("Model labels loaded: " + str(labels))
    print("Init positions loaded: " + str(initPositions.shape))
    
    return model, initPositions, labels, annotations

def save_model(model, initPos, labels, annotations):
    global LABELS_FILE
    global MODEL_FILE
    global INIT_POS_FILE

    np.save(MODEL_FILE, model)

    np.save(INIT_POS_FILE, initPos)

    labelData = {"labels": labels, "annotations": annotations}
    with open(LABELS_FILE, "w") as f:
        json.dump(labelData, f)
    


def main():
    # Start the video capture, figure out the frame dimensions
    cap = cv2.VideoCapture(0)
    frameWidth = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frameHeight = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(frameHeight, frameWidth)
    
    model, initPos, labels, annotations = load_model()
    
    # Thread setup
    threads = [Thread(target=video_runner, args=(cap, frameWidth, frameHeight, model, labels, annotations, )),
               Thread(target=recording_worker, args=(cap, model, labels, annotations, initPos, ))]
    
    for thread in threads:
        thread.daemon = True
        thread.start()
    for thread in threads:
        thread.join()


def distance_calc_test():
    fakePointHistory = []
    
    for i in range(0, 33):
        pointsList = []
        for j in range(0, TOTAL_LANDMARKS):
            pointsList.append((i, i, i))
        fakePointHistory.append(pointsList)
    
    fakePointHistory = np.array(fakePointHistory)

    res = perform_distance_calc(fakePointHistory)

    print(res.shape)
    print(res)

if __name__ == "__main__":
    main()