import torch

# Create tensors
x = torch.tensor(1.0)  # .0 to make it a float
y = torch.tensor(2.0)

# Initial weights
w = torch.tensor(1.0, requires_grad=True)

# Forward pass and compute the loss
y_hat = w * x 
loss = (y_hat - y)**2
print('Loss:', loss)

# Backward pass
loss.backward()  # This is the whole gradient computation :)
print('Weights grad:', w.grad)

# Next steps: Update weights & next forward and backward pass