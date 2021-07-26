import torch
import torch.nn as nn

activation = nn.ELU()

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    # LAYER 1 sees (32, 3, 18, 100, 100)
    self.conv1 = self.conv_layer(3,64,(1,2,2),(1,2,2))
    # LAYER 2 sees (32, 64, 18, 50, 50)
    self.conv2 = self.conv_layer(64,128,(1,2,2),(1,2,2))
    # LAYER 3 sees (32, 128, 18, 25, 25)
    self.conv3 = self.conv_layer(128,256,(2,2,2),(2,2,2))
    # LAYER 4 sees (32, 256, 9, 12, 12)
    self.conv4 = self.conv_layer(256,256,(2,2,2),(2,2,2))
    # Convoluted feature map shape (latent space): (32, 256, 4, 6, 6) so after flatten: 256*4*6*6 = 36864
    self.fc1 = nn.Linear(36864,512)
    self.fc2 = nn.Linear(512,512)
    self.fc3 = nn.Linear(512,256)
    self.fc4 = nn.Linear(256,11)

  def conv_layer(self, inChannel, outChannel, poolSize, stride):
    layer = nn.Sequential(
      nn.Conv3d(inChannel, outChannel, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm3d(outChannel),
      nn.ELU(),
      nn.MaxPool3d(poolSize, stride=stride, padding=0)
    )
    return layer
  
  def forward(self,x):
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.conv4(x)
    x = x.view(x.size(0), -1)
    x = activation(self.fc1(x))
    x = activation(self.fc2(x))
    x = activation(self.fc3(x))
    x = self.fc4(x)
    return x


