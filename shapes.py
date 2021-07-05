import numpy as np
import torch


a = np.random.rand(1,32,32,3)
b = np.random.rand(1,32,32,3)
c = np.random.rand(1,32,32,3)
d = np.random.rand(1,32,32,3)

x = np.random.rand(21,3)

print(np.concatenate((a,b,c,d)).shape)
a = torch.Tensor(a)
print(type(a))
print(x)
# [[(3, 32, 32, 3)], [0,0,0,1,0,0,0]]

"""
  for image, target in dataset:
    model.train()
"""