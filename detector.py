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
ges[0] = 'Sliding Two Fingers Left'
ges[1] = 'Sliding Two Fingers Right'
ges[2] = 'Sliding Two Fingers Down'
ges[3] = 'Sliding Two Fingers Up'
ges[4] = 'Pushing Two Fingers Away'
ges[5] = 'Pulling Two Fingers In'
ges[6] = 'Zooming In With Two Fingers'
ges[7] = 'Zooming Out With Two Fingers'
ges[8] = 'Stop Sign'
ges[9] = 'No gesture'
ges[10] = 'Doing other things'

print('loading model ...')
#state_dict = torch.load('model_best.pth.tar', map_location='cpu')['state_dict']

loaded_model = Net()
loaded_model.load_state_dict(torch.load("july29.pt", map_location='cpu'))

print(type(loaded_model))

transform = Compose([
        CenterCrop(100),
        #Resize(size=(300,300)),
        ToTensor()
        #Normalize(mean=[0.485, 0.456, 0.406],
                  #std=[0.229, 0.224, 0.225])
    ])

print("predicting...")

while True:
	ret, frame = camera.read()
	#print(np.shape(frame))

	#resized_frame = cv2.resize(frame, (149, 84)) #why

	pre_img = Image.fromarray(frame.astype('uint8'), 'RGB')

	img = transform(pre_img)

	imgs.append(torch.unsqueeze(img, 0))
	print(imgs[0])

	if len(imgs) == 18:
		data = torch.cat(imgs)
		#print(data)
		data = data.permute(1, 0, 2, 3)
		output = loaded_model(Variable(data).unsqueeze(0))
		out = (output.data).cpu().numpy()[0]
		print('Model output:', out)
		indices = np.argmax(out)
		print('Max index:', indices)
		pred = indices
		imgs = []
	
	cv2.putText(frame, ges[pred],(40,40), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0),2)
	cv2.imshow('why is it bad',frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

camera.release()
cv2.destroyAllWindows()

