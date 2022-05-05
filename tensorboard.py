import enum
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import sys
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
#from tensorboardX import SummaryWriter
# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('runs/mnist1')

# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # so run on GPU if supported

# hyper parameters
input_size = 784  # 28*28 images will be flatten so that's 784
hidden_size = 100
num_classes = 10
num_epochs = 2
batch_size = 100
learning_rate = .001

# MNIST
train_dataset = torchvision.datasets.MNIST(root='./data/mnist/', 
                                           train=True, 
                                           transform=transforms.ToTensor(), 
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='./data/mnist/', 
                                           train=False, 
                                           transform=transforms.ToTensor(), 
                                           download=False)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=batch_size,
                                           shuffle=False)

examples = iter(train_loader)
samples, labels = examples.next()
print(samples.shape, labels.shape)

for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(samples[i][0], cmap='gray')
# plt.show()
#img_grid = torchvision.utils.make_grid(samples)
#writer.add_image('MNIST_images', img_grid)


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out

# Model, Loss, Optimizer
model = NeuralNet(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

writer.add_graph(model, samples.reshape(-1, 28*28))
writer.close()

# Training Loop
n_total_steps = len(train_loader)
for epoch in range(num_epochs):  # loop over all epochs
    for i, (images, labels) in enumerate(train_loader):  # loop over all baches
        # Reshape images first, because currently: 100, 1, 28, 28 but input size is 100, 784
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        # Loss
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(f'epoch {epoch}/{num_epochs}: step {i}/{n_total_steps}, loss = {loss.item():.4f}')
        
# Test & Evaluation
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        # Reshape images first, because currently: 100, 1, 28, 28 but input size is 100, 784
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

        outputs = model(images)

        # returns value, index
        _, predictions = torch.max(outputs, 1)
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()

    acc = 100.0 * n_correct / n_samples

    print(f'Accuracy = {acc} %')