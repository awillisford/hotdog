import torch
import torch.nn as nn
import torch.nn.functional as F

''' https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html '''

class Model(nn.Module):
    def __init__(self):
        super().__init__() # runs __init__ from nn.Module
        self.conv1 = nn.Conv2d(1, 6, 5) # input channels, output channels, kernal size
        self.conv2 = nn.Conv2d(6, 12, 5)

        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(6049200, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 2) # 2, hotdog or not hotdog

    def forward(self, x):
        # max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        print("conv1:", x.size())

        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        print("conv2:", x.size())

        x = torch.flatten(x) # flatten
        print("flattened:", x.size())

        x = F.relu(self.fc1(x))
        print("lin1:", x.size())

        x = F.relu(self.fc2(x))
        print("lin2:", x.size())

        x = self.fc3(x)
        print("lin3:", x.size())

        return x

if __name__ == "__main__":
    net = Model()
    print(net, '\n')
