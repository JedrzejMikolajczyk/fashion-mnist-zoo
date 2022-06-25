import torch
import matplotlib.pyplot as plt

def save_sample(sample):
    sample = sample.to('cpu')
    plt.imshow(sample.squeeze())
    #add saving img

def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss


def predict(images, model, device, only_label = True):
    model.eval()
    predictions = []
    with torch.no_grad():
        for X in images:
            X = X.to(device)
            print(X.shape)
            X = torch.unsqueeze(X, dim=0)
            X = X[None, :,:,:]
            print(X.shape)
            pred = model(X)
            print(pred.shape)
            pred = torch.nn.functional.softmax(pred, dim=1)
            if only_label:
                predictions.append(torch.argmax(pred))
            else:
                predictions.append(pred)
    return predictions