import torch
import torch.nn as nn
import numpy as np

### SOFTMAX

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

x = np.array([2.0, 1.0, 0.1])
outputs = softmax(x)
print('Softmax numpy:', outputs)

x = torch.tensor([2.0, 1.0, 0.1])
outputs = torch.softmax(x, dim=0)
print(outputs)

### CROSS-ENTROPY: with numpy

def cross_entropy(actual, predicted):
    loss = -np.sum(actual * np.log(predicted))
    return loss 

# Y mustr be one-hot-encoded
# if class 0: [1, 0, 0]
# if class 1: [0, 1, 0]
# if class 2: [0, 0, 1]
Y = np.array([1,0,0])

Y_pred_good = np.array([0.7, 0.2, 0.1])
Y_pred_bad = np.array([0.1, 0.3, 0.6])
l1 = cross_entropy(Y, Y_pred_good)
l2 = cross_entropy(Y, Y_pred_bad)
print(f'Loss1 - good - numpy: {l1:.4f}')
print(f'Loss2 - bad - numpy: {l2:.4f}')

### CROSS-ENTROPY: with PyTorch
print('\nPrediction with PyTorch:')
loss = nn.CrossEntropyLoss()

Y = torch.tensor([0])  # only correct class label
# size = nsamples x nclasses = 1x3
Y_pred_good = torch.tensor([[2.0, 1.0, 0.1]])
Y_pred_bad = torch.tensor([[0.5, 2.0, 0.3]])

l1 = loss(Y_pred_good, Y)
l2 = loss(Y_pred_bad, Y)

print(f'Loss1 - good - numpy: {l1.item():.4f}')
print(f'Loss2 - bad - numpy: {l2.item():.4f}')

# Get predictions
_, predictions1 = torch.max(Y_pred_good, 1)
_, predictions2 = torch.max(Y_pred_bad, 1)

print(f'Pred 1: {predictions1}')
print(f'Pred 2: {predictions2}')

# With multiple classes
print('\nMULTI CLASS Prediction')
Y = torch.tensor([2, 0, 1])
Y_pred_good = torch.tensor([[0.1, 2.0, 2.1],
                            [2.0, 1.0, 0.1],
                            [0.1, 3.0, 0.1]])

loss = loss(Y_pred_good, Y)
print(f'Loss multiple classes: {loss.item():.4f}')
_, prediction = torch.max(Y_pred_good, 1)
print(f'Pred: {prediction}')


####################################

class NeuralNet1(nn.Module):  # BINARY Class
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet1, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, 1)  # always one with binary

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        # sigmoid at the end
        y_pred = torch.sigmoid(out)
        return out

    
model = NeuralNet1(input_size=28*28, hidden_size=5, num_classes=3)
criterion = nn.BCELoss()

####################################

class NeuralNet2(nn.Module):  # MULTI Class
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet2, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        # no softmax at the end
        return out

    
model = NeuralNet2(input_size=28*28, hidden_size=5, num_classes=3)
criterion = nn.CrossEntropyLoss()