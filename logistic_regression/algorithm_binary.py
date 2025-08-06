import torch
import torch.nn as nn

"""
Binary Logistic Regression Model
This module defines a simple binary logistic regression model using PyTorch.
"""
class BinaryLogisticRegressionModel(nn.Module):
    
    def __init__(self, input_dim):
        super(BinaryLogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)  # Linear layer with output dimension of 1 for binary classification

    def forward(self, x):
        combination = self.linear(x)                # Apply linear transformation
        probabilities = torch.sigmoid(combination)  # Apply sigmoid to get probabilities
        return probabilities
    
def train(model, criterion, optimizer, x_data, y_data, num_epochs=100):
    for epoch in range(num_epochs):
        pred = model(x_data)            # Forward pass: compute predicted y by passing x_data through the model
        loss = criterion(pred, y_data)  # Compute the loss using the criterion (mean squared error)
        optimizer.zero_grad()           # Zero the gradients before the backward pass
        loss.backward()                 # Backward pass: compute gradient of the loss with respect to model parameters
        optimizer.step()   
    
# Model creation
model = BinaryLogisticRegressionModel(input_dim=1)  # Example with 2 input features

# Loss function
criterion = nn.BCELoss()  # Binary Cross Entropy Loss for binary classification

# Optimizer (gradient descent)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Example data (for demonstration purposes)
x_data = torch.tensor([[1.0], [2.0], [3.0]])
y_data = torch.tensor([[0.0], [1.0], [1.0]])  # Binary labels

# Training the model
train(model, criterion, optimizer, x_data, y_data, num_epochs=100)

# Example usage
x_test = torch.tensor([[4.0]])
y_test_pred = model(x_test)
print(f"Predicted probability for input {x_test.item()}: {y_test_pred.item()}")