import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms

x=torch.randn(10,3)
y=torch.randn(10,2)

linear=nn.Linear(3,2)
print(linear.weight.shape)
print(linear.bias.shape)
print(linear.parameters)
output=linear(x)
#print(output.shape)
