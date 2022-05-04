import torch
import torch.nn as nn  # neural network module

# f = w * x
# f = 2 * x

X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)

X_test = torch.tensor([5], dtype=torch.float32)

n_samples, n_features = X.shape
print(n_samples, n_features)

input_size = n_features
output_size = n_features

class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        # define layers
        self.lin = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.lin(x)

model = LinearRegression(input_size, output_size)

print(f'Predictin before training: f(5) = {model(X_test).item():.3f}')

# Training
learning_rate = 0.01
n_iters = 1000

loss = nn.MSELoss()  # the var loss gets a callable function which can be used in the training loop
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(n_iters):
    # prediction = forward pass
    y_pred = model(X)

    # loss
    l = loss(Y, y_pred)

    # gradients = backward pass
    l.backward()  # Calc dl/dw

    # update weights
    optimizer.step()

    # Empty/Zero gradients
    optimizer.zero_grad()
    
    if epoch % 5 == 0:
        [w, b] = model.parameters()
        print(f'epoch {epoch}: w = {w[0][0].item():.3f}, loss= {l:.8f}')

print(f'Prediction after {n_iters} trainings: f(5) = {model(X_test).item():.3f}')