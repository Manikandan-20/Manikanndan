# Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import warnings
import os

warnings.filterwarnings("ignore")

# Step 1: Load Dataset (Change this path if your dataset is in a different location)
if not os.path.exists('train.csv'):
    print("Please upload the Ames Housing dataset (train.csv)")

# Step 2: Load Data and Inspect
df = pd.read_csv('train.csv')

# Inspect the data
print(df.head())
print(df.info())

# Step 3: Data Preprocessing

# Separate numerical and categorical columns
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = df.select_dtypes(include=['object']).columns

# Fill missing values for numerical columns with the median
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())

# Fill missing values for categorical columns with the mode (most frequent value)
for col in categorical_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Convert categorical columns into numerical using Label Encoding
label_encoder = LabelEncoder()
for col in categorical_cols:
    df[col] = label_encoder.fit_transform(df[col])

# Step 4: Split Data into Features (X) and Target (y)
X = df.drop('SalePrice', axis=1)
y = df['SalePrice']

# Step 5: Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Feature Scaling (Optional but recommended for some algorithms)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 7: Train Models - Linear Regression and Random Forest Regressor

# Initialize models
lr_model = LinearRegression()
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the models
lr_model.fit(X_train_scaled, y_train)
rf_model.fit(X_train_scaled, y_train)

# Step 8: Make Predictions
lr_predictions = lr_model.predict(X_test_scaled)
rf_predictions = rf_model.predict(X_test_scaled)

# Step 9: Evaluate Models

# Evaluate Linear Regression Model
lr_mse = mean_squared_error(y_test, lr_predictions)
lr_r2 = r2_score(y_test, lr_predictions)

# Evaluate Random Forest Model
rf_mse = mean_squared_error(y_test, rf_predictions)
rf_r2 = r2_score(y_test, rf_predictions)

print(f'Linear Regression MSE: {lr_mse:.2f}, R²: {lr_r2:.2f}')
print(f'Random Forest MSE: {rf_mse:.2f}, R²: {rf_r2:.2f}')

# Step 10: Hyperparameter Tuning for Random Forest (Optional)

# Define the parameter grid for tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# Perform grid search
grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42), param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train_scaled, y_train)

# Print the best parameters found
print(f"Best Parameters for Random Forest: {grid_search.best_params_}")

# Step 11: Evaluate the tuned model
best_rf_model = grid_search.best_estimator_
rf_tuned_predictions = best_rf_model.predict(X_test_scaled)

# Evaluate tuned Random Forest Model
rf_tuned_mse = mean_squared_error(y_test, rf_tuned_predictions)
rf_tuned_r2 = r2_score(y_test, rf_tuned_predictions)

print(f'Tuned Random Forest MSE: {rf_tuned_mse:.2f}, R²: {rf_tuned_r2:.2f}')

# Step 12: Feature Importance from Random Forest Model
# Feature importance visualization using the best tuned model
feature_importances = best_rf_model.feature_importances_

# Plot feature importance
plt.figure(figsize=(12, 6))
sns.barplot(x=feature_importances, y=X.columns)
plt.title('Feature Importance (Random Forest)')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.show()

# Step 13: Predict on New Data (Example)
new_house = {
    'MSSubClass': 60,  # Example feature
    'OverallQual': 7,  # Example feature
    'GrLivArea': 1500,  # Example feature
    'GarageCars': 2,  # Example feature
    'TotRmsAbvGrd': 8,  # Example feature
    'Fireplaces': 1,  # Example feature
    # Add more features as per your dataset
}

# Convert the new data to a DataFrame and scale it
new_house_df = pd.DataFrame([new_house])
new_house_scaled = scaler.transform(new_house_df)

# Predict using the best model
new_house_price = best_rf_model.predict(new_house_scaled)

print(f"Predicted Price for the New House: ${new_house_price[0]:,.2f}")
