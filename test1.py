import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms

import utils
from models.net1 import Net1

seed = 42
labels = ['t_shirt_top', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle_boots']


transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5), (0.5))])
#Load the training data
dataset = torchvision.datasets.FashionMNIST(root= './data', train = True, transform=transform, download = True)
testset = torchvision.datasets.FashionMNIST(root= './data', train = False, transform=transform, download = True)


trainset_size = int(0.8 *len(dataset))
validset_size = len(dataset) - trainset_size

trainset, validset = torch.utils.data.random_split(dataset, [trainset_size, validset_size], generator=torch.Generator().manual_seed(seed))

supersample = trainset[13][0]
batch_size = 64

print(trainset, validset)
print(trainset[0][0].shape)


# Create data loaders
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(testset, batch_size=4, 
                                         shuffle=False)

print(train_loader, test_loader)

device = 'cuda' if torch.cuda.is_available() else "cpu"


model = Net1()
model.load_state_dict(torch.load("model2.pth"))

loss_fn = nn.CrossEntropyLoss()

print(utils.test(test_loader, model, loss_fn, device))

utils.save_sample(supersample)
print(supersample.shape)
print(utils.predict(supersample, model, device))



