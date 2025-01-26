import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import numpy as np

# Load the dataset
data = pd.read_csv("dataset.csv")

# Separate features and target
X = data.drop("target", axis=1)
y = data["target"]

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize the Random Forest model
rf_model = RandomForestClassifier(random_state=42)

# Define the parameter grid for hyperparameter tuning
param_grid_rf = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
}

# Initialize GridSearchCV for hyperparameter tuning of RF
grid_search_rf = GridSearchCV(
    estimator=rf_model, param_grid=param_grid_rf, cv=5, scoring="roc_auc", n_jobs=-1
)

# Fit GridSearchCV to the training data
grid_search_rf.fit(X_train, y_train)

# Print the best parameters found by GridSearchCV for RF
print(f"Best parameters for RF: {grid_search_rf.best_params_}")

# Initialize the Gradient Boosting model
gb_model = GradientBoostingClassifier(random_state=42)

# Define the parameter grid for hyperparameter tuning
param_grid_gb = {
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.01, 0.1, 0.2],
    "max_depth": [3, 5, 7],
}

# Initialize GridSearchCV for hyperparameter tuning of GB
grid_search_gb = GridSearchCV(
    estimator=gb_model, param_grid=param_grid_gb, cv=5, scoring="roc_auc", n_jobs=-1
)

# Fit GridSearchCV to the training data
grid_search_gb.fit(X_train, y_train)

# Print the best parameters found by GridSearchCV for GB
print(f"Best parameters for GB: {grid_search_gb.best_params_}")

# Use the best estimators to predict on the test set
best_rf_model = grid_search_rf.best_estimator_
best_gb_model = grid_search_gb.best_estimator_

# Predict probabilities
y_pred_proba_rf = best_rf_model.predict_proba(X_test)[:, 1]
y_pred_proba_gb = best_gb_model.predict_proba(X_test)[:, 1]

# Average the predictions
y_pred_proba_ensemble = (y_pred_proba_rf + y_pred_proba_gb) / 2

# Calculate the AUC score on the test set
auc_score = roc_auc_score(y_test, y_pred_proba_ensemble)
print(f"AUC score on the test set: {auc_score}")

# Save the test predictions to a submission file
submission = pd.DataFrame({"Id": X_test.index, "Prediction": y_pred_proba_ensemble})
submission.to_csv("results.csv", index=False)
