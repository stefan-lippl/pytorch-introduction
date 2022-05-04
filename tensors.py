"""
This is the script to get a brief overview over simple PyTorch 
declarations and operations.
"""
import torch

###### Basics ######

# Create Tensor: 1D Vector with 3 elements
x1d = torch.empty(3)
print('1D: ', x1d)

# Create Tensor: 2D Matrix with 3 elements
x2d = torch.empty(2,3)
print('2D: ', x2d)

# Create Tensor: 3D Matrix with 3 elements
x3d = torch.empty(3,3)
print('3D: ', x3d)

# Create Tensor: With random values
ran = torch.rand(2,2)
print('Random: ', ran)

# Create Tensor: Zero tensors
zer = torch.zeros(2,2)
print('Zeros: ', zer)

# Create Tensor: One tensors
one = torch.ones(2,2)
print('Ones: ', one)

# Construct a tensor from data (e.g. Python-List)
con = torch.tensor([2.5, 1.7])
print('Construct from List: ', con)

# Operation: Check data type
print('Data Type: ', one.dtype)

# Operation: Change data type by parameter
dt = torch.ones(2,2, dtype=torch.int)
print('Data Type int: ', dt.dtype)

# Operation: Check the size of a tensor
print('Size: ', dt.size())


###### More Operations ######


# Addition
x = torch.rand(2,2)
y = torch.rand(2,2)

z1 = x + y
print('Addition 1: ', z1)
# or
z2 = torch.add(x, y)
print('Addition 2: ', z2)
# Inplace operation
# Note: every function in pt with a training _ will do an inplace operation
# Add
y.add_(x) 
print('Inplace operation add: ', y)

# Subtract
x = torch.rand(2,2)
y = torch.rand(2,2)

z1 = x - y
# or
z2 = torch.subtract(x,y)
print('Subtraction 1: ', z1)
print('Subtraction 2: ', z2)
# Inplace Subtraction
x.subtract_(y)
print('Inplace Subtraction: ', x)

# Multiply
x = torch.rand(2,2)
y = torch.rand(2,2)

z1 = x * y
# or
z2 = torch.mul(x,y)
print('Multi 1: ', z1)
print('Multi 2: ', z2)
# Inplace Multiply
x.mul_(y)
print('Inplace Multi: ', x)

# Division
x = torch.rand(2,2)
y = torch.rand(2,2)

z1 = x / y
# or
z2 = torch.div(x,y)
print('Div 1: ', z1)
print('Div 2: ', z2)
# Inplace Devision
x.div_(y)
print('Inplace Div: ', x)


# Slicing
x = torch.rand(5, 3)
print('Original: ', x)
print('Sliced first column: ', x[:, 0]) # get first column (like Pandas)
print('Sliced first row', x[0, :])


# Reshaping
x = torch.rand(4,4) # total 16 values
print('Original: ', x)
y1 = x.view(16) # total must match with original tensor
print('After reshape to 1D vector: ', y1)
# or
y2 = x.view(-1, 8) # pt will determine right size -> 2x8 tensor
print('8 elements per row: ', y2)
print('Size: ', y2.size())


# Converting from np to torch and vise verca
import numpy as np
a = torch.ones(5)
print(a, type(a))
b = a.numpy()  # point to same memory location, change in a is change in b
print(b, type(b))

a = np.ones(5)
print(a, type(a))
b = torch.from_numpy(a)
print(b, type(b))


# Specify Cuda device
if torch.cuda.is_available():
    device = torch.device("cuda")
    x = torch.ones(5, device=device)  # puts the tensor on the GPU
    # or
    y = torch.ones(5)
    y = y.to(device)
    z = x + y  # some kind of operation will be performed on the GPU, much faster
    # after processing you can do
    z = y.to('cpu')
else:
    print('No cuda device available')


# Requires Grad
# Tells PyTorch that it will need to calculate the gradients for this
# tensor later in your optimization steps -> rg=True required
x = torch.ones(5, requires_grad=True)
print(x)
