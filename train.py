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

#from models.net1 import Net1

from models.cnn import CNN

import utils




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=10, help="number of epochs of training")
    parser.add_argument("--ratio", type=float, default=0.8, help="portion of samples that is used for training (remaining part used for validation during training)")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    args = parser.parse_args()


    seed = 42
    split = args.ratio
    labels = ['t_shirt_top', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle_boots']
    
    
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5), (0.5))])
    #Load the training data
    dataset = torchvision.datasets.FashionMNIST(root= './data', train = True, transform=transform, download = True)
    
    
    trainset_size = int(split *len(dataset))
    validset_size = len(dataset) - trainset_size
    
    trainset, validset = torch.utils.data.random_split(dataset, [trainset_size, validset_size], generator=torch.Generator().manual_seed(seed))
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size)
    test_loader = torch.utils.data.DataLoader(validset, batch_size=4, 
                                             shuffle=False)
    
    
    device = 'cuda' if torch.cuda.is_available() else "cpu"
    
    model = CNN()
    
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    
    val_loss = 999999
    for t in range(args.n_epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        utils.train(train_loader, model, loss_fn, optimizer, device)
        
        current_val_loss = utils.test(test_loader, model, loss_fn, device)
        if current_val_loss < val_loss:
            val_loss = current_val_loss
            torch.save(model.state_dict(), "ttt1.pth")
            print("Model1 saved")
    print("Done!", val_loss)




