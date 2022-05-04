import torch
from torch.utils.data import Dataset, DataLoader 
import numpy as np
import math

class WineDataset(Dataset):

    def __init__(self):
        # data loading
        xy = np.loadtxt('data/wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(xy[:, [0]])  # n_samples, 1
        self.n_samples = xy.shape[0]


    def __getitem__(self, index):
        return self.x[index], self.y[index]  # returns tuple

    def __len__(self):
        return self.n_samples

dataset = WineDataset()
first_data = dataset[0]
features, labels = first_data

print('\nFeatures:', features, '\nLabels:', labels)

dataset = WineDataset()
dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=2)  # now uses subprocesses

# Training loop
num_epochs = 2
total_samples = len(dataset)
n_iterations = math.ceil(total_samples / 4)

print(total_samples, n_iterations)

for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(dataloader):
        if i % 5 == 0:
            print(f'Epoch {epoch}/{num_epochs}: step {i}/{n_iterations}, inputs {inputs.shape}')