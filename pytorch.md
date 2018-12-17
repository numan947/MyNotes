# Basic Tensor

`Tensor` is the base datastructure in `PyTorch`. Every `array like` datastructure is a tensor in this framework.

*Gradient points in the direction of fastest change, so taking -ve and adding to the weigths minimizes loss*


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

`tensor.fill_(val)`&rarr; inplace filling of a tensor

`tensor.normal_(mean,sigma)`&rarr; similar to  &uarr;

`tensor.mul_(constant)`&rarr; inplace multiplication of the tensor with the constant

`tenor.sum(dim=0|1|..)`&rarr; similar to `torch.sum()`

`tensor.reshape(a,b)`&rarr; returns the tensor in (a,b) shape, may be cloned or not, safety not guarenteed

`tensor.resize_(a,b)`&rarr; inplace operation of the reshape operation, can handle overflow and underflow

`tensor.view(a,b)`&rarr; returns a tensor of the shape `axb` created from it's original shape

`tensor.view(a,-1)`&rarr; returns a tensor with of shape `a` in first axis, the second one is resolved automatically

`tensor.numpy()`&rarr; returns a numpy version of the tensor, but doesn't clone the object, so any change in numpy/tensor is reflected on the whole thing

`tensor.backward()`&rarr; calculates and sets the gradients for all previously tracked Tensors upto current one

`tensor.item()`&rarr; returns the value of a single item `Tensor`

`tensor.type(Tensor.SomeOtherType)`&rarr; changing type of `Tensor`

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

`tensor.topk(n,dim=0|1|2...)`&rarr; returns n highest values and index along the dimension

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

`nn.Dropout(p=probability)`&rarr; drop out probability of the current layer's nodes

`model = nn.Module instance() , preferably a class subclassing from nn.Module`

`model.layerName.weights | model.layerName.bias`&rarr; returns `AutoGrad` variables corresponding to the weights and bias of the layer in consideration

`model.eval()`&rarr; model goes into evaluation mode and sets the `dropout` layers off

`model.train()`&rarr; model goes into training mode and sets the `dropout` layers on

`model.parameters()`&rarr; returns all the parameters of the model

`model.someLayer.parameters()`&rarr; returns the parameters of `someLayer` only

`model.state_dict()`&rarr; returns the `stateDictOfModel` that can be used to save the current state of the model

```python
#Another way to build a model without creating class

model = nn.Sequential(
    nn.SomeLayer1(),
    nn.SomeActivationFunction1(),
    nn.SomeLayer2(),
    nn.SomeActivationFunction2(),
    .
    .
    .
)

logits = model(data) #output of the model
loss = criterion(logits,actualLabels) #calculate loss
```

#### `import torch.nn.functional as F`

Special module providing some static access to different functions.

`F.Sigmoid()`

`F.Softmax()`

## AutoGrads

`model.layerName.weights|bias`&rarr; returns a `AutoGrad` variables, which are different from `Tensor` variables.

`autoGradVariable.data`&rarr; returns a pointer to the `Tensor` inside the variable

## DataSets, `torchvision`

---

`from torchvision import datasets, transforms`

`from torch.utils.data import DataLoader, ImageFolder`

`transofrms.Compose()`&rarr; for composing a pipeline of transforms to do on the loaded data

`transforms.Normalize((mean1,mean2,...),(std1,std2,...))`

`transforms.ToTensor()`&rarr; This should be the last transform in the pipeline

`transofrms.SeeMoreInTheDocks()`&rarr; this module also have image augmentation functions

`DataLoader(tensorData,batachSize,transforms,shuffle...)`&rarr; loads and make batches of the tensorData provided

`iter(dataLoaderObject)`&rarr; returns an iterator, so that we can call: iter.next() to get an image, label from the dataLoader

`ImageFolder(pathToImageFolder, transforms)`&rarr; loads data from imagefolder, where the folder is organized for general image classification.

`datasets.DataSetName(rootToSave,and other flags)`&rarr; for using the general datasets

## Optimizers

---
`from torch import optim`

`optim module`&rarr; contains implementations of  common and useful optimizers that can be used to train our model

`optim.SGD(modelParameters,learningRate)`

`optimizer.zero_grad()`&rarr; clears the current accumulated gradients, must be performed before backward pass

`optimizer.step()`&rarr; updates the gradients, i.e. takes a step

## SIMPLE NEURAL NETWORK

### Training
```python
epochs = 5
for e in range(epochs):
    running_loss = 0 #loss in current epoch
    for images, labels in trainloader:

        #forwar propagation
        log_ps = model(images) #last layer has LogSoftMax activation
        loss = criterion(log_ps, labels) #loss is NLLLoss() function

        #backward propagation
        optimizer.zero_grad() #zeroing out the gradietns
        loss.backward() #passing the loss backward
        optimizer.step()
        #accumulate loss for viewing purpose
        running_loss += loss.item()
    else:
        print(f"Training loss: {running_loss/len(trainloader)}")
```

### Testing

```python
# ps contains probabilities of all classes, torch.exp() is used on the logits to get the actual probabilities as the activation of outputlayer is LogSoftMax
img,label = (iter(testloader)).next()
ps = torch.exp(model(img)) 
```

## Transfer Learning

### model building

```python
from torchvision import models

# Use GPU if it's available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.densenet121(pretrained=True)

# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False

#our own classifier is added as the classifier of the densenet
model.classifier = nn.Sequential(nn.Linear(1024, 256),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(256, 2),
                                 nn.LogSoftmax(dim=1))

#our own criterion for loss
criterion = nn.NLLLoss()

# Only train the classifier parameters, feature parameters are frozen
optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)

#move the mode to cpu/gpu based on availability
model.to(device);
```

## Training with vailadtion

```python
epochs = 1
steps = 0
running_loss = 0
print_every = 5

for epoch in range(epochs):
    for inputs, labels in trainloader:
        steps += 1
        # Move input and label tensors to the default device
        inputs, labels = inputs.to(device), labels.to(device)

        #zero out the accumulated gradients
        optimizer.zero_grad()

        #forward pas
        logps = model.forward(inputs)
        loss = criterion(logps, labels)

        #backwardpass
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        #validation step
        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0

            #turn off dropout layers
            model.eval()

            #do not accumulate gradients for the  testset
            with torch.no_grad():
                for inputs, labels in testloader:
                    inputs, labels = inputs.to(device), labels.to(device)

                    #forward pass
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)

                    #loss accumulation
                    test_loss += batch_loss.item()

                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Test loss: {test_loss/len(testloader):.3f}.. "
                  f"Test accuracy: {accuracy/len(testloader):.3f}")
            running_loss = 0

            #turn on dropout layers
            model.train()
```