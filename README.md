# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Regression problems involve predicting a continuous output variable based on input features. Traditional linear regression models often struggle with complex patterns in data. Neural networks, specifically feedforward neural networks, can capture these complex relationships by using multiple layers of neurons and activation functions. In this experiment, a neural network model is introduced with a single linear layer that learns the parameters weight and bias using gradient descent.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: Generate Dataset

Create input values  from 1 to 50 and add random noise to introduce variations in output values .

### STEP 2: Initialize the Neural Network Model

Define a simple linear regression model using torch.nn.Linear() and initialize weights and bias values randomly.

### STEP 3: Define Loss Function and Optimizer

Use Mean Squared Error (MSE) as the loss function and optimize using Stochastic Gradient Descent (SGD) with a learning rate of 0.001.

### STEP 4: Train the Model

Run the training process for 100 epochs, compute loss, update weights and bias using backpropagation.

### STEP 5: Plot the Loss Curve

Track the loss function values across epochs to visualize convergence.

### STEP 6: Visualize the Best-Fit Line

Plot the original dataset along with the learned linear model.

### STEP 7: Make Predictions

Use the trained model to predict  for a new input value .

## PROGRAM

### Name:SANJAI S

### Register Number:212223230185

```
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
X = torch.linspace(1,70,70).reshape(-1,1)
torch.manual_seed(71)
e = torch.randint(-8,9,(70,1),dtype=torch.float)
y = 2*X + 1 + e
print(y.shape)
plt.scatter(X.numpy(), y.numpy(),color='red')  # Scatter plot of data points
plt.xlabel('x')
plt.ylabel('y')
plt.title('Generated Data for Linear Regression')
plt.show()
class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        
    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred
torch.manual_seed(59)
model = nn.Linear(1, 1)
print('Weight:', model.weight.item())
print('Bias:  ', model.bias.item())
loss_function = nn.MSELoss()

optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
epochs = 50
losses = []

for epoch in range(1, epochs + 1):
    optimizer.zero_grad()
    y_pred = model(X)
    loss = loss_function(y_pred, y)
    losses.append(loss.item())

    loss.backward()
    optimizer.step()


    print(f'epoch: {epoch:2}  loss: {loss.item():10.8f}  '
          f'weight: {model.weight.item():10.8f}  '
          f'bias: {model.bias.item():10.8f}')
plt.plot(range(epochs), losses)
plt.ylabel('Loss')
plt.xlabel('epoch');
plt.show()

x1 = torch.tensor([X.min().item(), X.max().item()])


w1, b1 = model.weight.item(), model.bias.item()


y1 = x1 * w1 + b1

print(f'Final Weight: {w1:.8f}, Final Bias: {b1:.8f}')
print(f'X range: {x1.numpy()}')
print(f'Predicted Y values: {y1.numpy()}')

plt.scatter(X.numpy(), y.numpy(), label="Original Data")
plt.plot(x1.numpy(), y1.numpy(), 'r', label="Best-Fit Line")
plt.xlabel('x')
plt.ylabel('y')
plt.title('Trained Model: Best-Fit Line')
plt.legend()
plt.show()
```
### Dataset Information
<img width="798" height="572" alt="image" src="https://github.com/user-attachments/assets/e41266f9-3b87-41f8-8be2-63af30fdaa12" />


### OUTPUT
Training Loss Vs Iteration Plot


<img width="710" height="743" alt="image" src="https://github.com/user-attachments/assets/aca936ec-0a46-4803-bf73-41f8db996d1e" />

Best Fit line plot


<img width="693" height="441" alt="image" src="https://github.com/user-attachments/assets/53fd059a-302b-4c34-a8a1-3f5cc1b29736" />


<img width="671" height="430" alt="image" src="https://github.com/user-attachments/assets/1b71ac3b-56ac-4e0a-806e-8178d720ac63" />



### New Sample Data Prediction

<img width="384" height="66" alt="image" src="https://github.com/user-attachments/assets/c085ef36-f1d0-4ed7-baa8-a05f09574595" />


## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
