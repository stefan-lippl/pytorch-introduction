# PyTorch Basics

This is a brief summary of the 5h [course](https://www.youtube.com/watch?v=c36lUUr864M) from Patrick (YT: [Python Engineer](https://www.youtube.com/channel/UCbXgNpp0jedKWcQiULLbDTA)) about how to use **PyTorch**.

<br>

### Table of Content

| Chapter | Description |
| ------- | ----------- |
| [PyTorch Installation](#pytorch-installation) | How to setup PyTorch locally (the right way incl. vir env) |
| [Tensors Basics](#tensors-basics) | Create tensors and basic operations (Add, Sum etc.) |
| [Autograd](#gradient-optimization-with-autograd) |  |
| [Backpropagation](#backpropagation) |  |
| [Gradient Descent](#gradient-descent) |  |

<br>

***

<br>
<br>
<br>

# PyTorch Installation

1) Go to [pytorch.org](www.pytorch.org) and click on **Install**
2) **Configure** your settings, in my case:

    | Settings | Value |
    | - | - |
    | PyTorch Build | Stable (1.10.0) |
    | Your OS | Mac |
    | Package | Conda |
    | Language | Python |
    | Compute Platform | CPU |

    Unfortunally I have no GPU in my Mac, else you can use a specific *CUDA* version. Therefor you have to first download [*CUDA* Toolkit](www.google.de). *Cuda* Toolkit is a dev environment for creating high performance GPU accelerated applications. If you have a NVIDIA GPU in your machine i highly recomment to download the toolkit.

3) Copy the **Run this Command**, in my case `conda install pytorch torchvision torchaudio -c pytorch`
4) Open up the terminal
5) Navigate to a folder of your choice where the project should be located (navigate with `cd` followed by the folder path)
6) Create a virtual environment with *conda*

    ```bash
    conda create --name pytorch-basics python=3.7
    ```

    > Note: ATM **PyTorch** can't work with python 3.8+ so use Python 3.7 instead

   ```bash
    conda activate pytorch-basics
    ```
7) Now paste the installation command from 3) in the terminal, this will install **PyTorch** inside the virtual environment
    ```bash
    conda install pytorch torchvision torchaudio -c pytorch
    ```
8) Verify the **PyTorch** installation by starting Python inside this environment. Therefor start python
    ```bash
    python
    ```
    followed by 
    ```bash
    import torch
    ```
    
    <br>

    > If the installation was *NOT* correct, you would get now a **`ModuleNotFoundError`**. In this case try again to install PyTorch with the command, created in 3). You can also see [this troubleshoot article](https://pytorch.org/get-started/locally/#mac-prerequisites) for further informations. 
    
    **Else you successfull installed **PyTorch****

9) (*optional*) You can also verify if *Cuda* is available 
    ```bash
    import torch
    torch.cuda.is_available()
    ```
    In my case (MacOs) I get False returned.

<br>
<br>
<br>

***

<br>
<br>
<br>

# Tensors Basics
You can find basic operations with tensors in the script `tensors.py`.

### Topics:
- Create:
    - 1D Vectors
    - 2D Matrix
    - 3D Tensor
    - Custom Tensor (variable size + values)
- dtype
- size
- Operations:
    - Addition of tensors
    - Subtraction of tensors
    - Multiplication of tensors
    - Division of tensors
- Inplace operations
    - Add
    - Subtract
    - Multiply
    - Division
- Slicing
- Reshaping
- Converting numpy <-> tensor
- Specify Cuda device
- requires_grad

<br>
<br>
<br>

***

<br>
<br>
<br>

# Gradient optimization with `autograd`
You can find gradient optimization examples in the script `autograd.py`.

Important to know is, whenever you want to calculate the gradients, you must specify `requires_grad=True` as attribute inside your tensor.

### Topics:
- Vector Jacobian Product (https://arxiv.org/pdf/2104.00219.pdf)
- Stop PyTorch tracking history

<br>
<br>
<br>

***

<br>
<br>
<br>

# Backpropagation
You can find gradient optimization examples in the script `backpropagation.py`.

Whole concept:
1) Forward pass: Compute Loss
2) Compute local gradients
3) Backward pass: Compute dLoss / dWeights using the Chain Rule

![Backpropagation](media/backprop.png)


### Topics:
- Chain rule: dz/dx = dz/dy * dy/dx for x -> a(x) -> y -> b(y) -> z
- Computational Graph
- Calculate loss (with y_hat)

<br>
<br>
<br>

***

<br>
<br>
<br>

# Gradient Descent
You can find gradient optimization examples in the script `backpropagation.py`.