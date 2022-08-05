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
from models import model_factory

#python test.py -m Net1 -f weights/Net1.pth
#python test.py -m Net1 cnn linear -f weights/Net1.pth weights/cnn.pth weights/linear.pth


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name", nargs="*", default = ["cnn"], help="models to be tested, type 'all' to test all models")
    parser.add_argument("-f", "--file_name", nargs="*", default=['weights/cnn.pth'], help="names of saved models to be tested")
    args = parser.parse_args()

    #checking if there are as many models as models' names
    if len(args.model_name) != len(args.file_name):
        raise Exception("number of parameters passed to 'model_name' and 'save_as' have to be equal and in correct order")

    seed = 42
    labels = ['t_shirt_top', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle_boots']
    device = 'cuda' if torch.cuda.is_available() else "cpu"    
    
    #preprocessing
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5), (0.5))])
    #Load the testing data
    dataset = torchvision.datasets.FashionMNIST(root= './data', train = True, transform=transform, download = True)
    testset = torchvision.datasets.FashionMNIST(root= './data', train = False, transform=transform, download = True)
    
    #spliting training set into training and validation sets
    trainset_size = int(0.8 *len(dataset))
    validset_size = len(dataset) - trainset_size
    trainset, validset = torch.utils.data.random_split(dataset, [trainset_size, validset_size], generator=torch.Generator().manual_seed(seed))
        
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=4)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=4, 
                                             shuffle=False)
    
    #load a list of models
    models_to_test = model_factory.get_model(args.model_name) 
    
    #testing each model
    for i, model in enumerate(models_to_test):
        #load saved weights to model
        model.load_state_dict(torch.load(args.file_name[i]))
        print(f"Testing model: '{args.model_name[i]}' \nWeights loaded from: '{args.file_name[i]}'")
        
        loss_fn = nn.CrossEntropyLoss()
        train_acc, train_loss = utils.test(train_loader, model, loss_fn, device)
        test_acc, test_loss = utils.test(test_loader, model, loss_fn, device)
        print(f"On Training dataset: \n Accuracy: {(100*train_acc):>0.1f}%, Avg loss: {train_loss:>8f} \n")
        print(f"On Testing dataset: \n Accuracy: {(100*test_acc):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        print('-'*10)
    


