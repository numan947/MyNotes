# Basic Tensor

`Tensor` is the base datastructure in `PyTorch`. Every `array like` datastructure is a tensor in this framework.

## torch module

---

`import torch`

`torch.manual_seed(seed)`&rarr; sets the seed for the environment globally.

`torch.zeros(shape,required_grad=True)`&rarr; returns a Tensor of zeros of the shape with gradient tracking enabled.

`torch.randn(shape,required_grad=True)`&rarr; returns a Tensor of the given shape randomly from the normal distribution.

`torch.rand_like(Tensor,required_grad=True)`&rarr; returns a Tensor of the same shape of the given tensor randomly from the normal distribution

`torch.sum(Tensor,dim=0|1|2...)`&rarr; sums up and returns a value/Tensor after summing up along the provided axis

`torch.from_numpy(arr)`&rarr; creates a Tensor from a numpy array

`torch.mm(matA,matB)|torch.matmul(matA,matB)`&rarr; does matrix multiplication and returns where matA, matB are tensors with matching dimensions

```python
with torch.no_grad():
    calculation are done here that requires no gradient  
    tracking/update, e.g validation/test work on the
    current trained model
```

`torch.set_grad_enabled(True|False)`&rarr; globally turn off/on gradient tracking

`torch.save(StateDictOfModel,PathToSaveFile)`&rarr; saves the statedict of a model in the specified file

`torch.load(PathToSaveFile)`&rarr; loads the statedict of a model and creates a PyTorch model from that.

## Tensor

---

`_`&rarr; at the end  means that it's an inplace operation 

`tensor.mul_(constant)`&rarr; inplace multiplication of the tensor with the constant

`tenor.sum(dim=0|1|..)`&rarr; similar to `torch.sum()`

`tensor.reshape(a,b)`&rarr; returns the tensor in (a,b) shape, may be cloned or not, safety not guarenteed

`tensor.resize_(a,b)`&rarr; inplace operation of the reshape operation, can handle overflow and underflow

`tensor.view(a,b)`&rarr; returns a tensor of the shape `axb` created from it's original shape

`tensor.view(a,-1)`&rarr; returns a tensor with of shape `a` in first axis, the second one is resolved automatically

`tensor.numpy()`&rarr; returns a numpy version of the tensor, but doesn't clone the object, so any change in numpy/tensor is reflected on the whole thing

```text
MATRIX MULTIPLICATION FOR SUMMING UP THE WEIGHTS AND INPUTS
h = [x1,x2,....,xn][
                    [w1],
                    [w2],
                    .
                    .
                    .
                    .
                    .
                    ,
                    [wn]
                    ]
```

## NN Module

---

`from torch import nn`

### How to use

1. create a new class and subclass `nn.Module`
2. inside `__init__(self):` of the new class, call `super().__init__()` at the very first line
3. create the architecture
4. override/create  `def forward(self,x)` method

### Module functionalities

`nn.Linear(inputNumber,outputNumber)`&rarr; creates a Linear layer that takes `inputNumber` inputs and outputs/forwards `outputNumber` outputs

`nn.Sigmoid(dim=0|1|..)`&rarr; a sigmoid activation function

`nn.Softmax(dim=0|1|..)`&rarr; a softmax activation function

`nn.CrossEntropyLoss()`&rarr; a loss function that combines `nn.LogSoftmax()` and `nn.NLLLoss()`

`model = nn.Module instance() , preferably a class subclassing from nn.Module`

`model.layerName.weights | model.layerName.bias`&rarr; returns `AutoGrad` variables corresponding to the weights and bias of the layer in consideration

