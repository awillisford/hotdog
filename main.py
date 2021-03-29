from Model import Model
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import torch

''' https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html '''

def main():
    net = Model()

    training_data = np.load("data/training_data.npy", allow_pickle=True)

    SAVE_DATA = False

    if SAVE_DATA == True:
        features = [x[0] for x in training_data]
        labels = [x[1] for x in training_data]

        X = torch.tensor(features)
        y = torch.tensor(labels)

        torch.save(X, 'features.pt')
        torch.save(y, 'labels.pt')
    else:
        X = torch.load('features.pt')
        y = torch.load('labels.pt')

    # BATCH_SIZE = 100
    # for epoch in range(1):
    #     for batch in tqdm(range(0, len(training_data), BATCH_SIZE)):
    #         pass

if __name__ == "__main__":
    main()
