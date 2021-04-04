from Model import Model
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import gc

''' https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html '''

def main():
    gc.collect()
    net = Model().to(device).double() # attempt to run using CUDA

    training_data = np.load("data/training_data.npy", allow_pickle=True)

    SAVE_DATA = False # constant

    # determine if data is saved as tensor file
    if SAVE_DATA:
        features = [x[0]/255.0 for x in training_data]
        labels = [x[1] for x in training_data]

        X = torch.tensor(features)
        y = torch.tensor(labels)

        torch.save(X, 'features.pt')
        torch.save(y, 'labels.pt')
    else: # load tensor from files
        X = torch.load('features.pt')
        y = torch.load('labels.pt')

    loss_function = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01)
    BATCH_SIZE = 100 # constant
    for epoch in range(1): # iterate through dataset range(x) times
        for val in tqdm(range(0, len(training_data), BATCH_SIZE)): # iterate through dataset, jumping BATCH_SIZE at a time
            batch_X = X[val:val + BATCH_SIZE].view(-1, 1, 299, 299) # assign features to current batch
            batch_y = y[val:val + BATCH_SIZE].double() # assign labels to current batch

            batch_X, batch_y = batch_X.to(device), batch_y.to(device) # use GPU if possible

            net.zero_grad() # set gradients to zero

            # pass features through network and calculate loss
            output = net.forward(batch_X.double())
            loss = loss_function(output, batch_y).double()

            print(loss)

            loss.backward()



if __name__ == "__main__":
    print("CUDA enabled -",torch.cuda.is_available())

    if torch.cuda.is_available():
        device = torch.device("cuda:0") # type device, index 0
    else:
        device = torch.device("cpu") # type device

    main()
