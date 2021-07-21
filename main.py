import torch
import torch.nn as nn
from preprocessing import VideoFolder
import json
from torchvision.transforms import *
import cv2
from PIL import Image as im
from matplotlib import pyplot as plt
from model import Net
import torch.optim as optim
import os
import tqdm

with open("./configs/config.json") as dataFile:
  config = json.load(dataFile)

save_dir = os.path.join(config["output_dir"], config["model_name"])

device = torch.device("cpu")

transform = Compose([
        CenterCrop(84),
        ToTensor()
        #Normalize(mean=[0.485, 0.456, 0.406],
                  #std=[0.229, 0.224, 0.225])
    ])

train_data = VideoFolder(root=config['train_data_folder'],
                             csv_file_input=config['train_data_csv'],
                             csv_file_labels=config['labels_csv'],
                             clip_size=config['clip_size'],
                             nclips=1,
                             step_size=config['step_size'],
                             is_val=False,
                             transform=transform,
                             )

#print(train_data.classes)
#print(train_data)

# Train Data is in the shape of [[images], [class]]
# [images] in the shape of : torch.Size([132, 18, 3, 100]) or WIDTH, DEPTH, CHANNELS, HEIGHT ( FIX THIS )
"""print(train_data[1000][0].shape)
print(len(train_data[1000][0]))
print("NUMBER OF CLASSES", len(train_data.classes))
print((len(train_data[200][0])))
print(train_data[200][0][0].shape)
print(train_data[200][0].shape)
print(len(train_data[200][0]))
print(len(train_data[0][0][0]))
print(train_data[2][0][0].shape)"""
#cv2.imshow("image", train_data[2][0][0]) # show numpy array
#cv2.waitKey(0) # wait for ay key to exit window
#cv2.destroyAllWindows()

"""data_item, target_idx = train_data[0]
data_item.unsqueeze(0)"""

train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=config['batch_size'], shuffle=True,
        num_workers=config['num_workers'], pin_memory=True,
        drop_last=True)

print(len(train_loader))

#print(type(train_loader))
    
  
val_data = VideoFolder(root=config['val_data_folder'],
                          csv_file_input=config['val_data_csv'],
                          csv_file_labels=config['labels_csv'],
                          clip_size=config['clip_size'],
                          nclips=1,
                          step_size=config['step_size'],
                          is_val=True,
                          transform=transform
                          )


val_loader = torch.utils.data.DataLoader(
    val_data,
    batch_size=config['batch_size'], shuffle=False,
    num_workers=config['num_workers'], pin_memory=True,
    drop_last=False)


"""if __name__ == "__main__":
  for local_batch, local_labels in train_loader:
      local_batch, local_labels = local_batch.to(device), local_labels.to(device)"""


if __name__ == '__main__':
  #next(iter(data_loader))
  net = Net()
  optimizer = torch.optim.SGD(net.parameters(), config['lr'],
                                  momentum=config['momentum'],
                                  weight_decay=config['weight_decay'])
  

  x = next(iter(train_loader))
  img = (x[0][0].permute(1,2,3,0))
  print(img.shape)
  print(img[0].shape)
  cv2.imshow("IMAGE", img[0].detach().numpy())
  cv2.waitKey(0)
  
# closing all open windows
  cv2.destroyAllWindows()
  

  criterion = nn.CrossEntropyLoss()

  EPOCHS = 10

  #net.train()

"""  for epoch in range(EPOCHS):
    print("EPOCH: ", epoch)
    for i, (input, target) in enumerate(train_loader):
      print("got here")
      net.zero_grad()
      output = net(input)
      loss = criterion(output, target)
      print("here too")
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      print(loss)
      

    print("LOSS", loss)
"""