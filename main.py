from Model import Model
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn

''' https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html '''

def main():
    net = Model().double()

    training_data = np.load("data/training_data.npy", allow_pickle=True)

    SAVE_DATA = False

    if SAVE_DATA == True:
        features = [x[0]/255.0 for x in training_data]
        labels = [x[1] for x in training_data]

        X = torch.tensor(features)
        y = torch.tensor(labels)

        torch.save(X, 'features.pt')
        torch.save(y, 'labels.pt')
    else:
        X = torch.load('features.pt')
        y = torch.load('labels.pt')

    loss_function = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01)
    BATCH_SIZE = 100
    for epoch in range(1):
        for val in tqdm(range(0, len(training_data), BATCH_SIZE)):
            batch_X = X[val:val + BATCH_SIZE].view(-1, 1, 299, 299)
            batch_y = y[val:val + BATCH_SIZE]

            net.zero_grad() # set gradients to zero

            # pass features through network and calculate loss
            output = net.forward(batch_X.double())
            loss = loss_function(output, batch_y)

            loss.backward()



if __name__ == "__main__":
    main()
