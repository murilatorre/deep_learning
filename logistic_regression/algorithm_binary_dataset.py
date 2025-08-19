import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset # Dataset: stores the samples and their corresponding labels.
                                                 # DataLoader: provides an iterable over the dataset.
from sklearn.preprocessing import LabelEncoder   # LabelEncoder: encodes categorical labels as integers                                                 


class CustomDataset(Dataset):

    # Initialization method for the CustomDataset class
    def __init__(self, file_path):
        data = pd.read_csv(file_path)
        label_encoder = LabelEncoder()  
        data["Sex"] = label_encoder.fit_transform(data["Sex"]) # Female=0 and Male=1 (alphabetical order)
        self.x_data = torch.tensor(data[['Sex', 'Age', 'Height', 'Weight']].dropna().values, dtype=torch.float32)
        self.y_data = torch.tensor(data[['Shoe number']].dropna().values, dtype=torch.float32)  
        self.length = len(self.x_data)

    # Returns the number of samples in the dataset
    def __len__(self):
        return self.length

    # Returns a sample and its corresponding label at the given index
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]                      

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
    
def train(model, criterion, optimizer, dataloader, num_epochs=100):
    for epoch in range(num_epochs):
        for x_data, y_data in dataloader:
            pred = model(x_data)            # Forward pass: compute predicted y by passing x_data through the model
            loss = criterion(pred, y_data)  # Compute the loss using the criterion (binary cross entropy)
            optimizer.zero_grad()           # Zero the gradients before the backward pass
            loss.backward()                 # Backward pass: compute gradient of the loss with respect to model parameters
            optimizer.step()                # Update model parameters using the optimizer

# Model creation
model = BinaryLogisticRegressionModel(input_dim=4)  # Example with 4 input features

# Loss function
criterion = nn.BCELoss()  # Binary Cross Entropy Loss for binary classification

# Optimizer (gradient descent)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Example data from csv
dataset = CustomDataset('../datasets/dataMAC0460_5832.csv')
dataloader = DataLoader(dataset, batch_size=30, shuffle=True)

# Training the model
train(model, criterion, optimizer, dataloader, num_epochs=100)

# Example usage
x_test = torch.tensor([[0, 25.0, 160.0, 75.0]], dtype=torch.float32)

