import numpy as np
import torch
import matplotlib.pyplot as plt

training_data = np.load("training_data.npy", allow_pickle=True)
print("Training data size:", len(training_data))

plt.imshow(training_data[0][0], cmap="gray") # feature
print(training_data[0][1]) # print label, 0 is hotdog, 1 is not hotdog
plt.show()
