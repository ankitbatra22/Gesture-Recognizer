from collections import OrderedDict
import cv2
import numpy as np
import pickle
from model import Net
from PIL import Image
from time import time
import torch
from torch.autograd import Variable
from torchvision.transforms import *
from model import Net

seq_len = 18
imgs = []
pred = 0

# Capture video from computer camera
camera = cv2.VideoCapture(0)
# Define gesture names
ges = dict()
ges[0] = 'Swiping Left'
ges[1] = 'Swiping Right'
ges[2] = 'Swiping Down'
ges[3] = 'Swiping Up'
ges[4] = 'Stop Sign'
ges[5] = 'Doing other things'
ges[6] = 'No gesture'

print('loading model ...')

loaded_model = Net().cuda()
loaded_model.load_state_dict(torch.load(
    "models/aug9.pt", map_location='cuda'))

print(type(loaded_model))

transform = Compose([
    CenterCrop(100),
    # Resize(size=(300,300)),
    ToTensor()
    # Normalize(mean=[0.485, 0.456, 0.406],
    # std=[0.229, 0.224, 0.225])
])

print("predicting...")

tracker = 0

while True:
    tracker += 1
    ret, camera_frame = camera.read()
    # print(np.shape(frame))

    frame = cv2.resize(camera_frame, (160, 120))  # why

    pre_img = Image.fromarray(frame.astype('uint8'), 'RGB')

    img = transform(pre_img).cuda()

    img_numpy = img.permute(1, 2, 0).cpu().detach().numpy()
    # if keyboard.is_pressed("w"):
    # 	print("reset")
    # 	imgs= []
    # if keyboard.is_pressed("space"):
    # 	print(len(imgs))

    imgs.append(torch.unsqueeze(img, 0))

    if len(imgs) == 18:
        data = torch.cat(imgs)
        # print(data)
        data = data.permute(1, 0, 2, 3)
        output = loaded_model(Variable(data).unsqueeze(0))
        out = (output.data).cpu().detach().numpy()[0]
        #print('Model output:', out)
        indices = np.argmax(out)
        tracker = 0
        print('class:', ges[indices])
        pred = indices

    if len(imgs) == 18:
        imgs = imgs[1:]

    cv2.putText(camera_frame, ges[pred], (40, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.imshow('why is it bad', img_numpy)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
