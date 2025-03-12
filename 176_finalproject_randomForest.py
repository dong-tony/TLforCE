################################################
# This script uses a Random Forest Regressor 
# to predict the LCE values of Kim et al. (2023) 
# PNAS dataset.
################################################

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load Excel file
path = 'drive/MyDrive/Colab Notebooks/176 Final Project/'
data = pd.read_excel(path+'pnasDataset.xlsx', engine='openpyxl',index_col=0, skiprows=1).fillna(0)

# Check that we are looking at the right columns
print(data.iloc[:,23])
print(data.iloc[:,9])
print(data.iloc[:,22])

################### Set up model ######################

# Define features (X) and target variable (y)
y = data.iloc[:, 23].to_numpy()
X = data.iloc[:,9:22].to_numpy()

# Split dataset into training (80%) and test (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

# Train Random Forest Regressor
RFmodel = RandomForestRegressor(n_estimators=300, random_state=32)
RFmodel.fit(X_train, y_train)

# Make predictions
y_pred = RFmodel.predict(X_test)

# Evaluate model performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2:.2f}")

######################## Plot predicted vs. actual values #########################
plt.scatter(y_test, y_pred, alpha=0.7, label='Actual vs. Predicted')
plt.xlabel("Actual LCE Values")
plt.ylabel("Predicted LCE Values")
plt.title("Random Forest Regression: Actual vs. Predicted LCE")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r', linestyle="--", label='Perfect Fit Line')  # Perfect prediction line
plt.text(1.5, 0.7, f"MSE: {mse:.2f}", size=20, rotation=0.,
         ha="left", va="bottom",
         bbox=dict(boxstyle="round",
                   ec=(1., 0.5, 0.5),
                   fc=(1., 0.8, 0.8),
                   )
         )
plt.text(1.5, 0.95, f"R² Score: {r2:.2f}", size=20, rotation=0.,
         ha="left", va="bottom",
         bbox=dict(boxstyle="round",
                  ec=(0.0, 0.0, 0.5),
                  fc=(0.7, 0.8, 1.0),
                  )
         )
plt.legend()
plt.show()

#################### Cross Validation of Random Forest Model ###################################

# Generate cross validation
scores = cross_val_score(RFmodel, X, y, cv=5, scoring='neg_mean_squared_error')
# Use other scoring metrics if needed, e.g., 'r2' for R-squared

# Convert negative MSE scores to positive
mse_scores = -scores

# Print the results
print("Cross-Validation MSE Scores: ", mse_scores, " Average MSE:", mse_scores.mean(), " Standard Deviation of MSE:", mse_scores.std())
