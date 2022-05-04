"""
1) Design the model (input, output, forward pass)
2) Construct loss and optimizer
3) Training loop:
    - forward pass: compute prediction and loss
    - backward pass: gradients
    - update weights
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
print('OK')

### 0) PREPARE DATA
X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)

# convert numpy to tensor
X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))

# reshape - want to make it a column vector
y = y.view(y.shape[0], 1)
print(y)

n_samples, n_features = X.shape

### 1) MODEL
input_size = n_features
output_size = 1

model = nn.Linear(input_size, output_size)

### 2) LOSS and OPTIMIZER
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

### 3) TRAINING LOOP
num_epochs = 1000
for epoch in range(num_epochs):
    # forward pass and loss
    y_pred = model(X)
    loss = criterion(y_pred, y)

    # backward pass
    loss.backward()

    # update
    optimizer.step()

    # empty gradients
    optimizer.zero_grad()

    if epoch % 10 == 0:
        print(f'epoch {epoch}: loss = {loss.item():.4f}')

# Plot
predicted = model(X).detach().numpy()
plt.plot(X_numpy, y_numpy, 'ro')
plt.plot(X_numpy, predicted, 'b')
plt.show()
