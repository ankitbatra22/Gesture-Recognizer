import torch
import torch.nn as nn
from preprocessing import VideoFolder
import json
from torchvision.transforms import *

with open("configs/config.json") as dataFile:
  config = json.load(dataFile)

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
#print(train_data.dataset_object)
#print(train_data.csv_data)
#x = (train_data.dataset_object)
#print(x.classes_dict)
print(train_data[2][0].shape)
"""data_item, target_idx = train_data[0]
data_item.unsqueeze(0)"""

train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=config['batch_size'], shuffle=True,
        num_workers=config['num_workers'], pin_memory=True,
        drop_last=True)
  
print(type(train_loader))

"""def main():
  for input in enumerate(train_loader):
      #input, target = input.to(device), target.to(device)
      print(type(input))"""

#print((train_loader.batch_size))
"""":
  for data,target in train_loader:
    print(data.shape)"""

"""if __name__ == "__main__":
  for local_batch, local_labels in train_loader:
      local_batch, local_labels = local_batch.to(device), local_labels.to(device)"""

"""if __name__ == "__main__":
  print("got here")
  dataiter = iter(train_loader)
  print('got herreee')
  images, labels = dataiter.next()
  images = images.numpy()"""
