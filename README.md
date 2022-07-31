# fashion mnist zoo
 This project contains implementations of some PyTorch models and code for training and evaluating them.

## Dataset
[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) is a dataset of Zalando's article images—consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes.

Each training and test example is assigned to one of the following labels:  

0. T-shirt/top
1. Trouser
2. Pullover
3. Dress
4. Coat
5. Sandal
6. Shirt
7. Sneaker
8. Bag
9. Ankle boot

## Results
TODO: add this section

## Usage
### Training
example:  
Train a cnn model and a linear model for 20 epochs and save learned weights to a weights folder as cnn.pth and linear.pth respectively
```bash 
python train.py -n 2 -m cnn linear -f weights/cnn.pth weights/linear.pth
```

Optional arguments: 

| Parameter             | Default         | Description   |	
| :---------------------|:---------------:| :-------------|
| -n, --n_epochs	       |	10              | number of epochs of training
| -r, --ratio           | 0.8             | percent of samples that is used for training (remaining part used for validation during training)
| -b, --batch_size 	    |	64	             | size of the batches
| --lr	                 | 0.0002          | adam: learning rate 
| -m, --model_name	     | ["cnn"]         | models to be trained, type 'all' to train all models 
| -f, --file_name       | ["cnn.pth"]     | name a trained model is to be saved as
| -c, --console_logging | False           | Log progress to console?

### Testing
example:
 ```bash 
python test.py  -m cnn net1 -f weights/cnn.pth weights/net1.pth
```

Optional arguments: 

| Parameter             | Default              | Description   |	
| :---------------------|:--------------------:| :-------------|
| -m, --model_name	     | ["cnn"]              | models to be tested, type 'all' to test all models
| -f, --file_name       | ["weights/cnn.pth"]  | names of saved models to be tested

## Models to be implemented:
- [x] CNN
- [x] feed-forward network
- [x] multi-class perceptron
- [ ] GAN
- [ ] Encoder + some model
- [ ] Many more


