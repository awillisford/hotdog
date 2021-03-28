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
        self.fc1 = nn.Linear(60492, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 2) # 2, hotdog or not hotdog

    def forward(self, x):
        # max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)

        x = torch.flatten(x) # flatten

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == "__main__":
    net = Model()
    print(net, '\n')

    input = torch.randn(1, 1, 299, 299)
    print(input, '\n')

    out = net.forward(input)
    print(out)
