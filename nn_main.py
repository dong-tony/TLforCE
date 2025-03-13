import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from nn_helper import net, prepare_data, train_model, freeze_layers, evaluate_model

random_seed = 42

# load data
ce_X = pd.read_csv('data/CE_X.csv')
ce_y = pd.read_csv('data/CE_y.csv')

cond_X = pd.read_csv('data/conductivity_X.csv', index_col=0)
cond_y = pd.read_csv('data/conductivity_y.csv', index_col=0)

cond_X = cond_X[ce_X.columns]

#split data
ce_X_train, ce_X_test, ce_y_train, ce_y_test = train_test_split(
    ce_X, ce_y, test_size=0.2, random_state=random_seed
    )

# Standardize features
scaler = StandardScaler()
cond_X = scaler.fit_transform(cond_X)
ce_X_train = scaler.transform(ce_X_train)
ce_X_test = scaler.transform(ce_X_test)
# standardize conductivity values
scaler_y = StandardScaler()
cond_y = scaler_y.fit_transform(cond_y)

cond_X_train, cond_X_test, cond_y_train, cond_y_test = train_test_split(
    cond_X, cond_y, test_size=0.2, random_state=random_seed
    )

# Prepare dataloaders for pytorch
ce_train_loader = prepare_data(ce_X_train, ce_y_train.to_numpy(), bs=10) # small batch size for smaller CE dataset
ce_test_loader = prepare_data(ce_X_test, ce_y_test.to_numpy(), bs=10)
cond_full_loader = prepare_data(cond_X, cond_y, bs=32)
cond_train_loader = prepare_data(cond_X_train, cond_y_train, bs=32)
cond_test_loader = prepare_data(cond_X_test, cond_y_test, bs=32)

cond_model = net(cond_X.shape[1], 64, 64, 1)
print('Training Conductivity Model')
cond_model, _ = train_model(cond_model, 
                            cond_train_loader, 
                            nn.MSELoss(), 
                            optim.Adam(cond_model.parameters(), lr=0.001)
                            )

cond_test_mse = evaluate_model(cond_model, cond_test_loader, nn.MSELoss())
print(f'Conductivity Test MSE: {cond_test_mse}')

# Calculate R2 score for cond_model
cond_model.eval()
with torch.no_grad():
    y_true = []
    y_pred = []
    for inputs, targets in cond_test_loader:
        outputs = cond_model(inputs)
        y_true.extend(targets.numpy())
        y_pred.extend(outputs.numpy())
cond_r2 = r2_score(y_true, y_pred)
print(f'Conductivity Test R2: {cond_r2}')

print('training CE model without transfer')
ce_model = net(cond_X.shape[1], 64, 64, 1)

ce_model, _ = train_model(ce_model, 
                            ce_train_loader, 
                            nn.MSELoss(), 
                            optim.Adam(ce_model.parameters(), lr=0.001)
                            )
ce_test_mse = evaluate_model(ce_model, ce_test_loader, nn.MSELoss())
print(f'CE Test MSE without transfer: {ce_test_mse}')

print('training ce model with transfer from conductivity')
cond_model_transfer = net(cond_X.shape[1], 64, 64, 1)
cond_model_transfer, _ = train_model(cond_model_transfer, 
                            cond_full_loader, # use full conductivity dataset
                            nn.MSELoss(), 
                            optim.Adam(cond_model_transfer.parameters(), lr=0.001)
                            )
# freeze_layers(cond_model_transfer)
ce_model_transfer, _ = train_model(cond_model_transfer, 
                            ce_train_loader, # use full conductivity dataset
                            nn.MSELoss(), 
                            optim.Adam(cond_model_transfer.parameters(), lr=0.001)
                            )
ce_transfer_test_mse = evaluate_model(ce_model_transfer, ce_test_loader, nn.MSELoss())
print(f'CE Test MSE with transfer: {ce_transfer_test_mse}')



