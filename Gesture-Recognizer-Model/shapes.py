import numpy as np
import torch
import torch.nn as nn

def conv3d_output(shape, Kernel, Padding=(0, 0, 0), Stride=(1, 1, 1)):
    #Z : depth
    #Y : height
    #X : width
    #P : padding
    #K : kernel

    Z, Y, X = shape
    
    Z_out = ((Z + 2 * Padding[0] - (Kernel[0] - 1) - 1) / Stride[0]) + 1
    Y_out = ((Y + 2 * Padding[1] - (Kernel[1] - 1) - 1) / Stride[1]) + 1
    X_out = ((X + 2 * Padding[2] - (Kernel[2] - 1) - 1) / Stride[2]) + 1
    
    return (Z_out, Y_out, X_out)

m = nn.Dropout(p=0.2)
inputs = torch.randn(20, 16)
output = m(inputs)

a = np.random.rand(1,32,32,3)
b = np.random.rand(1,32,32,3)
c = np.random.rand(1,32,32,3)
d = np.random.rand(1,32,32,3)

x = np.random.rand(21,3)

#print(np.concatenate((a,b,c,d)).shape)
a = torch.Tensor(a)
#print(type(a))
#print(x)
# [[(3, 32, 32, 3)], [0,0,0,1,0,0,0]]

"""
  for image, target in dataset:
    model.train()
"""
def conv_layer(inChannel, outChannel, poolSize, stride):
    layer = nn.Sequential(
      nn.Conv3d(inChannel, outChannel, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm3d(outChannel),
      nn.ELU(),
      nn.MaxPool3d(poolSize, stride=stride, padding=0)
    )
    return layer

x = np.random.rand(32,3,18,84,84)
x = torch.Tensor(x)
#print(type(x))
#print(x.dtype)
#print(conv1(x).shape)

conv1 = conv_layer(3,64,(1,2,2),(1,2,2))
conv2 = conv_layer(64,128,(1,2,2),(1,2,2))
conv3 = conv_layer(128,256,(2,2,2),(2,2,2))
conv4 = conv_layer(256,256,(2,2,2),(2,2,2))

print(x.shape)
x = conv1(x)
print(x.shape)
x = conv2(x)
print(x.shape)
x = conv3(x)
print(x.shape)
x = conv4(x)
print(x.shape)

