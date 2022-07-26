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



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-bs", "--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("-mn", "--model_name", nargs="*", default = ["Net1"], help="models to be tested, type 'all' to test all models")
    parser.add_argument("-fn", "--file_name", nargs="*", default=['weights/model2-linear.pth'], help="names of saved models to be tested")
    parser.add_argument("-c", "--console_logging", type=bool, default=False, help="Log progress to console?")
    args = parser.parse_args()

    #checking if there are as many models as models' names
    if len(args.model_name) != len(args.file_name):
        raise Exception("number of parameters passed to 'model_name' and 'save_as' have to be equal and in correct order")



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
    

    
    models_to_test = model_factory.get_model(args.model_name) 
    
    
    for i, model in enumerate(models_to_test):
        model.load_state_dict(torch.load(args.file_name[i]))
        
        loss_fn = nn.CrossEntropyLoss()
        
        print(utils.test(test_loader, model, loss_fn, device))
        
        
        utils.save_sample(supersample)
        print(supersample.shape)
        print(utils.predict(supersample, model, device))
    


