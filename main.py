from Model import Model
import torch.optim as optim
import tqdm

''' https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html '''

def main():
    net = Model()

    BATCH_SIZE = 100
    for epoch in range(1):
        for batch in tqdm(range(0, len(training_data), BATCH_SIZE)):

if __name__ == "__main__":
    main()
