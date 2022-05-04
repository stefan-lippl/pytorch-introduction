import numpy as np

# f = w * x
# f = 2 * x

X = np.array([1, 2, 3, 4], dtype=np.float32)
Y = np.array([2, 4, 6, 8], dtype=np.float32)

w = 0.0

# Model prediction
def forward(x):
    return w * x

# Loss = MSE
def loss(y, y_predicted):
    return ((y_predicted-y)**2).mean()

# Gradient
# MSE = 1/N * (w*x - y)**2
# dJ/dw = 1/N 2x * (w*x - y)  - derivative
def gradient(x, y, y_predicted):
    return np.dot(2*x, y_predicted-y).mean()

print(f'Predictin before training: f(5) = {forward(5):.3f}')

# Training
learning_rate = 0.01
n_iters = 10

for epoch in range(n_iters):
    # prediction = forward pass
    y_pred = forward(X)

    # loss
    l = loss(Y, y_pred)

    # gradients
    dw = gradient(X, Y, y_pred)

    # update weights
    w -= learning_rate * dw

    print(f'epoch {epoch+1}: w = {w:.3f}, loss= {l:.8f}')

print(f'Prediction after {n_iters} training: f(5) = {forward(5):.3f}')