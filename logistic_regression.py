import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

###### LOAD & PREPARE DATA ######
# Load data and have a first look at size and features
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target
n_samples, n_features = X.shape
print('\nNumber samples:', n_samples, '\nNumber features per sample:', n_features, '\n')

# Split the data to train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=1234)

# Scale the features
sc = StandardScaler()  # always recommended when Logistic regression
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Convert to torch tensors
X_train = torch.from_numpy(X_train.astype(np.float32))  # because currently it is a double which produces errors later on
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

# Reshape y data
y_train = y_train.view(y_train.shape[0], 1)  # from 1 row to 1 column
y_test = y_test.view(y_test.shape[0], 1)


###### MODEL ######
class LogisticRegression(nn.Module):
    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_predicted = torch.sigmoid(self.linear(x))
        return y_predicted

model = LogisticRegression(n_input_features=n_features)


###### LOSS & OPTIMIZER ######
criterion = nn.BCELoss()  # Binary Cross Entropy
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


###### TRAINING LOOP ######
n_epochs = 100000

for epoch in range(n_epochs):
    # forward pass & loss calculation
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)

    # backward pass
    loss.backward()

    # update weights
    optimizer.step()

    # zero gradients
    optimizer.zero_grad()

    if epoch % 10000 == 0:
        print(f'Epoch {epoch}/{n_epochs}: loss = {loss.item():.4f}')

    with torch.no_grad():
        y_pred = model(X_test)
        y_pred_cls = y_pred.round()
        acc = y_pred_cls.eq(y_test).sum() / float(y_test.shape[0])
print(f'\nAccuracy: {acc:.4f}')