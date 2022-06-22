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

from models.net1 import Net1
import utils

seed = 42
split = 0.8
labels = ['t_shirt_top', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle_boots']


transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5), (0.5))])
#Load the training data
dataset = torchvision.datasets.FashionMNIST(root= './data', train = True, transform=transform, download = True)


trainset_size = int(split *len(dataset))
validset_size = len(dataset) - trainset_size

trainset, validset = torch.utils.data.random_split(dataset, [trainset_size, validset_size], generator=torch.Generator().manual_seed(seed))

supersample = trainset[0][0]
batch_size = 64

print(trainset, validset)
print(trainset[0][0].shape)


# Create data loaders
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(validset, batch_size=4, 
                                         shuffle=False)

print(train_loader, test_loader)

device = 'cuda' if torch.cuda.is_available() else "cpu"




epochs = 100

model = Net1()


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

val_loss = 999999
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    utils.train(train_loader, model, loss_fn, optimizer, device)
    
    current_val_loss = utils.test(test_loader, model, loss_fn, device)
    if current_val_loss < val_loss:
        val_loss = current_val_loss
        torch.save(model.state_dict(), "model1.pth")
        print("Model1 saved")
print("Done!", val_loss)




