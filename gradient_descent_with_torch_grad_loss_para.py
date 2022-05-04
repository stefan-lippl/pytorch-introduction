import torch
import torch.nn as nn  # neural network module

# f = w * x
# f = 2 * x

X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)

w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

# Model prediction
def forward(x):
    return w * x

print(f'Predictin before training: f(5) = {forward(5):.3f}')

# Training
learning_rate = 0.01
n_iters = 50

loss = nn.MSELoss()  # the var loss gets a callable function which can be used in the training loop
optimizer = torch.optim.SGD([w], lr=learning_rate)

for epoch in range(n_iters):
    # prediction = forward pass
    y_pred = forward(X)

    # loss
    l = loss(Y, y_pred)

    # gradients = backward pass
    l.backward()  # Calc dl/dw

    # update weights
    optimizer.step()

    # Empty/Zero gradients
    optimizer.zero_grad()
    
    if epoch % 5 == 0:
        print(f'epoch {epoch}: w = {w:.3f}, loss= {l:.8f}')

print(f'Prediction after {n_iters} trainings: f(5) = {forward(5):.3f}')