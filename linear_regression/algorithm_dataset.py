import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset # Dataset: stores the samples and their corresponding labels.
                                                 # DataLoader: provides an iterable over the dataset.


class CustomDataset(Dataset):

    # Initialization method for the CustomDataset class
    def __init__(self, file_path):
        data = pd.read_csv(file_path)
        self.x_data = torch.tensor(data[['CRIM', 'RM']].dropna().values, dtype=torch.float32)
        self.y_data = torch.tensor(data[['MEDV']].dropna().values, dtype=torch.float32)  
        self.length = len(self.x_data)

    # Returns the number of samples in the dataset
    def __len__(self):
        return self.length

    # Returns a sample and its corresponding label at the given index
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]


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
def train(model, criterion, optimizer, dataloader, num_epochs=100):
    for epoch in range(num_epochs):
        for x_data, y_data in dataloader:
            pred = model(x_data)            # Forward pass: compute predicted y by passing x_data through the model
            loss = criterion(pred, y_data)  # Compute the loss using the criterion (mean squared error)
            optimizer.zero_grad()           # Zero the gradients before the backward pass
            loss.backward()                 # Backward pass: compute gradient of the loss with respect to model parameters
            optimizer.step()                # Update model parameters using the optimizer


# Model creation
model = LinearRegressionModel(input_dim=2, output_dim=1)

# Loss function
criterion = nn.MSELoss()

# Optimizer (gradient descent)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Example data from csv
dataset = CustomDataset('../datasets/HousingData.csv')
dataloader = DataLoader(dataset, batch_size=30, shuffle=True)

# Training the model
train(model, criterion, optimizer, dataloader, num_epochs=100)

# Example usage
x_test = torch.tensor([[0.00632, 6.575]], dtype=torch.float32)  # Example input with two features
y_test_pred = model(x_test)
print(f"Predicted value for input: {y_test_pred.item()}")
