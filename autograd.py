"""
Gradients are essential for model optimization.
"""

from requests import request
from setuptools import Require
import torch

x = torch.randn(3, requires_grad=True)
print('Original: ', x)

y = x + 2
print(y)

z = y * y * 2
z = z.mean()
print(z)

z.backward()
print(x.grad)


# Prevent PyTorch tracking the history and calc grad_fn attr
x = torch.randn(3, requires_grad=True)
print('Original: ', x)

# 1) x.requires_grad_(False)
# 2) x.detach()
# 3) with torch.no_grad(): ...

x.requires_grad_(False)  # _ remember: modifies variable inplace
print('1) ', x)

y = x.detach()  # will create a new tensor with the same values but doesn't require gradient
print('2) ', y)

with torch.no_grad():
    y = x + 2
    print('3) ', y)

# Correct gradients
weights = torch.ones(4, requires_grad=True)
for epoch in range(2):
    model_output = (weights*3).sum()  # dummy operation, simulates model output
    model_output.backward()  # calculate the gradients
    print(epoch, weights.grad)

    weights.grad.zero_()  # correct gradients

# Optimizer
weights = torch.ones(4, requires_grad=True)

optimizer = torch.optim.SGD(weights, lr=0.01)
optimizer.step()
optimizer.zero_grad()