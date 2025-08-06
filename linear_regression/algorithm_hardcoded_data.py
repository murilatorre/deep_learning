import torch
import torch.nn as nn

""""
Linear Regression Model
This module defines a simple linear regression model using PyTorch.
"""
class LinearRegressionModel(nn.Module):
    # nn.Module is the base class for all neural network modules in PyTorch.
    
    # Initialization method for the LinearRegressionModel class
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()  # calls the constructor of the parent class (nn.Module)
        self.linear = nn.Linear(input_dim, output_dim) # defines a linear layer with input and output dimensions

    # Defines how the model processes the data
    def forward(self, x):
        y_pred = self.linear(x) # applies the linear transformation to the input data
        return y_pred
    
""""
" Training algorithm for Linear Regression
"""
def train(model, criterion, optimizer, x_data, y_data, num_epochs=100):
    for epoch in range(num_epochs):
        pred = model(x_data)            # Forward pass: compute predicted y by passing x_data through the model
        loss = criterion(pred, y_data)  # Compute the loss using the criterion (mean squared error)
        optimizer.zero_grad()           # Zero the gradients before the backward pass
        loss.backward()                 # Backward pass: compute gradient of the loss with respect to model parameters
        optimizer.step()                # Update model parameters using the optimizer


# Model creation
model = LinearRegressionModel(input_dim=1, output_dim=1)

# Loss function
criterion = nn.MSELoss()

# Optimizer (gradient descent)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Example data (for demonstration purposes)
x_data = torch.tensor([[1.0], [2.0], [3.0]])
y_data = torch.tensor([[2.0], [4.0], [6.0]])

# Training the model
train(model, criterion, optimizer, x_data, y_data, num_epochs=100)

# Example usage
x_test = torch.tensor([[4.0]])
y_test_pred = model(x_test)
print(f"Predicted value for input {x_test.item()}: {y_test_pred.item()}")
