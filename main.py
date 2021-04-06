from Model import Model
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn

''' https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html '''

def main():
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


    IMG_SIZE = X[0].size()[0] # get image size from first feature
    BATCH_SIZE = 20 # constant

    print("IMG_SIZE: ", (IMG_SIZE, IMG_SIZE))
    print("BATCH_SIZE: ", BATCH_SIZE)

    net = Model(IMG_SIZE, BATCH_SIZE).to(device).double() # attempt to run using CUDA

    loss_function = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    for epoch in range(20): # iterate through dataset range(x) times
        print(f"Epoch: {epoch}")
        for val in tqdm(range(0, len(training_data), BATCH_SIZE)): # iterate through dataset, jumping BATCH_SIZE at a time
            batch_X = X[val:val + BATCH_SIZE].view(-1, 1, IMG_SIZE, IMG_SIZE).to(device) # assign features to current batch
            batch_y = y[val:val + BATCH_SIZE].double().to(device) # assign labels to current batch

            net.zero_grad() # set gradients to zero

            # pass features through network and calculate loss
            output = net.forward(batch_X.double())
            loss = loss_function(output, batch_y).double()
            print(loss)

            loss.backward()
            optimizer.step() # updates model
        print(f"Epoch: {epoch} - Loss: {loss}")

if __name__ == "__main__":
    print("CUDA enabled -",torch.cuda.is_available())

    if torch.cuda.is_available():
        device = torch.device("cuda") # type device, index 0
    else:
        device = torch.device("cpu") # type device

    main()
