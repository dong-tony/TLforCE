###################################################
# This script uses linear regression to predict
# the LCE values of Kim et al. (2023) PNAS dataset.
####################################################

# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Define features (X) and target variable (y)
X = pd.read_csv('data/CE_X.csv').to_numpy()
y = pd.read_csv('data/CE_y.csv').to_numpy()

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features (mean=0, std=1) for better regression performance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train linear regression model
linearReg_model = LinearRegression()
linearReg_model.fit(X_train, y_train)

# Make predictions
y_pred = linearReg_model.predict(X_test)

# Evaluate model performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Print model coefficients
print("Model Coefficients:", linearReg_model.coef_)
print("Intercept:", linearReg_model.intercept_)

############## Plot predicted vs. actual values #################
plt.scatter(y_test, y_pred, alpha=0.9, label='Actual vs. Predicted')
plt.xlabel("Actual LCE Values")
plt.ylabel("Predicted LCE Values")
plt.title("Linear Regression: Actual vs. Predicted LCE")
plt.text(1.6, 0.7, f"MSE: {mse:.2f}", size=20, rotation=0.,
         ha="left", va="bottom",
         bbox=dict(boxstyle="round",
                   ec=(1., 0.5, 0.5),
                   fc=(1., 0.8, 0.8),
                   )
         )
plt.text(1.6, 0.95, f"R² Score: {r2:.2f}", size=20, rotation=0.,
         ha="left", va="bottom",
         bbox=dict(boxstyle="round",
                  ec=(0.0, 0.0, 0.5),
                  fc=(0.7, 0.8, 1.0),
                  )
         )
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r', linestyle="--", label='Perfect Fit Line')
plt.legend()
plt.show()

########### Cross Validation of Linear Regression Model ################

from sklearn.model_selection import cross_val_score
# Generate cross-validation scores for linear regression
scores = cross_val_score(linearReg_model, X, y, cv=5, scoring='neg_mean_squared_error')
# Use other scoring metrics if needed, e.g., 'r2' for R-squared

# Convert negative MSE scores to positive
mse_scores = -scores

# Print the results
print("Cross-Validation MSE Scores:", mse_scores)
print("Average MSE:", mse_scores.mean())
print("Standard Deviation of MSE:", mse_scores.std())
