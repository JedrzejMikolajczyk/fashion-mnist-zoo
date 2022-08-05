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

from torch.utils.tensorboard import SummaryWriter
#from models.net1 import Net1

from models import model_factory

import utils




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n_epochs", type=int, default=10, help="number of epochs of training")
    parser.add_argument("-r", "--ratio", type=float, default=0.8, help="portion of samples that is used for training (remaining part used for validation during training)")
    parser.add_argument("-b", "--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("-m", "--model_name", nargs="*", default = ["cnn"], help="models to be trained, type 'all' to train all models")
    parser.add_argument("-f", "--file_name", nargs="*", default=['weights/cnn111.pth'], help="name a trained model is to be saved as")
    parser.add_argument("-c", "--console_logging", type=bool, default=False, help="Log progress to console?")
    parser.add_argument("-t", "--tensorboard_logging", type=bool, default=True, help="Log accuracy and loss to tensorboard file?")
    
    args = parser.parse_args()

    #checking if there are as many models as models' names
    if len(args.model_name) != len(args.file_name):
        raise Exception("number of parameters passed to 'model_name' and 'save_as' have to be equal as 1 model is to be saved under 1 name")


    seed = 42
    labels = ['t_shirt_top', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle_boots']
    device = 'cuda' if torch.cuda.is_available() else "cpu"
    
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5), (0.5))])
    #Load the training data
    dataset = torchvision.datasets.FashionMNIST(root= './data', train = True, transform=transform, download = True)
    
    #spliting training set into training and validation sets
    trainset_size = int(args.ratio *len(dataset))
    validset_size = len(dataset) - trainset_size
    trainset, validset = torch.utils.data.random_split(dataset, [trainset_size, validset_size], generator=torch.Generator().manual_seed(seed))
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size)
    val_loader = torch.utils.data.DataLoader(validset, batch_size=4, 
                                             shuffle=False)

    #load a list of models
    models_to_train = model_factory.get_model(args.model_name)

    #perform training for each model
    for i, model in enumerate(models_to_train):
        if args.tensorboard_logging:
            tensorboard_writer = SummaryWriter(args.file_name[i])
               
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
        
        val_loss = 999999
        for t in range(args.n_epochs):
            #train a model and measure loss over validation dataset
            current_train_loss = utils.train(train_loader, model, loss_fn, optimizer, device)
            current_accuracy, current_val_loss = utils.test(val_loader, model, loss_fn, device)
            if args.tensorboard_logging:    
                tensorboard_writer.add_scalar("Training Loss", current_train_loss)
                tensorboard_writer.add_scalar("Validation Loss", current_val_loss)
                tensorboard_writer.add_scalar("Accuracy", current_accuracy)
            if args.console_logging:
                print(f"Epoch {t+1}\n-------------------------------")
                print(f"Test Error: \n Accuracy: {(100*current_accuracy):>0.1f}%, Avg loss: {current_val_loss:>8f} \n")
            if current_val_loss < val_loss:
                val_loss = current_val_loss
                #saving model
                torch.save(model.state_dict(), args.file_name[i])
                if args.console_logging:
                    print("Model saved as " + args.file_name[i])
        print(args.file_name[i] + " finished!")
    print("All finished!")




