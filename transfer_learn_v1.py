from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

random_seed = 42

# load data
ce_X = pd.read_csv('data/CE_X.csv')
ce_y = pd.read_csv('data/CE_y.csv')

# reserve 20% as final test data
ce_X_train, ce_X_test, ce_y_train, ce_y_test = train_test_split(
    ce_X, ce_y, test_size=0.2, random_state=random_seed
    )

cond_X = pd.read_csv('data/conductivity_X.csv', index_col=0)
cond_y = pd.read_csv('data/conductivity_y.csv', index_col=0)

# have columns in cond_x be in the same order as ce_X
cond_X = cond_X[ce_X.columns]

# Standardize features
scaler = StandardScaler()
cond_X = scaler.fit_transform(cond_X)
ce_X_train = scaler.transform(ce_X_train)
ce_X_test = scaler.transform(ce_X_test)

# Define neural network model with 3 layers
class NeuralNet(nn.Module):
    def __init__(self, input_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Prepare data for PyTorch
def prepare_data(X, y, bs=32):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y.to_numpy(), dtype=torch.float32).view(-1, 1)
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=bs, shuffle=True)

# Prepare datasets
ce_train_loader = prepare_data(ce_X_train, ce_y_train, bs=10)
ce_test_loader = prepare_data(ce_X_test, ce_y_test, bs=10)
cond_train_loader = prepare_data(cond_X, cond_y, bs=32)

# Initialize model, loss function, and optimizer
input_size = ce_X.shape[1]
model = NeuralNet(input_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

# Freeze layers except for the final output layer for transfer learning
def freeze_layers(model):
    for param in model.fc1.parameters():
        param.requires_grad = False
    for param in model.fc2.parameters():
        param.requires_grad = False

def train_model(model, train_loader, criterion, optimizer, epochs=500):
    model.train()
    epoch_losses = []
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        epoch_losses.append(avg_loss)
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}')
    return model, epoch_losses

model, cond_epoch_losses = train_model(model, cond_train_loader, criterion, optimizer)\

freeze_layers(model)

model, ce_epoch_losses = train_model(model, ce_train_loader, criterion, optimizer)

def evaluate_model(model, test_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()
    mse = total_loss / len(test_loader)
    return mse

ce_test_mse = evaluate_model(model, ce_test_loader, criterion)
print(f'CE Test MSE: {ce_test_mse}')

# Train model from scratch on CE data
model_scratch = NeuralNet(input_size)
optimizer_scratch = optim.Adam(model_scratch.parameters(), lr=0.005)

model_scratch, ce_scratch_epoch_losses = train_model(model_scratch, ce_train_loader, criterion, optimizer_scratch)

# Evaluate model trained from scratch and calculate MSE
ce_scratch_test_mse = evaluate_model(model_scratch, ce_test_loader, criterion)
print(f'CE Test MSE (Scratch): {ce_scratch_test_mse}')

# Compare results
print(f'CE Test MSE (Transfer Learning): {ce_test_mse}')
print(f'CE Test MSE (Scratch): {ce_scratch_test_mse}')