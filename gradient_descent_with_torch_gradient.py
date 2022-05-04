import torch

# f = w * x
# f = 2 * x

X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)

w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

# Model prediction
def forward(x):
    return w * x

# Loss = MSE
def loss(y, y_predicted):
    return ((y_predicted-y)**2).mean()

print(f'Predictin before training: f(5) = {forward(5):.3f}')

# Training
learning_rate = 0.01
n_iters = 10

for epoch in range(n_iters):
    # prediction = forward pass
    y_pred = forward(X)

    # loss
    l = loss(Y, y_pred)

    # gradients = backward pass
    l.backward()  # Calc dl/dw

    # update weights
    with torch.no_grad():
        w -= learning_rate * w.grad

    # Empty/Zero gradients
    w.grad.zero_()
    
    print(f'epoch {epoch+1}: w = {w:.3f}, loss= {l:.8f}')

print(f'Prediction after {n_iters} trainings: f(5) = {forward(5):.3f}')