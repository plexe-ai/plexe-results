import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import roc_auc_score, make_scorer
import lightgbm as lgb

# Load the dataset
data = pd.read_csv("dataset.csv")

# Assume the last column is the target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Define the LightGBM model
model = lgb.LGBMClassifier()

# Define the parameter grid
param_grid = {
    "num_leaves": [31, 50],
    "max_depth": [-1, 10, 20],
    "learning_rate": [0.1, 0.01],
    "n_estimators": [100, 200],
}

# Use AUC as the scoring metric
scorer = make_scorer(roc_auc_score, needs_proba=True)

# Initialize GridSearchCV
grid_search = GridSearchCV(
    estimator=model, param_grid=param_grid, scoring=scorer, cv=3, n_jobs=-1
)

# Fit grid search
grid_search.fit(X, y)

# Get the best model
best_model = grid_search.best_estimator_

# Initialize variables for cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)
auc_scores = []

# Cross-validation with best model
for train_index, val_index in kf.split(X):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    # Train the best model
    best_model.fit(X_train, y_train)

    # Predict and calculate AUC
    val_preds = best_model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_preds)
    auc_scores.append(auc)

# Calculate the mean AUC score
mean_auc = np.mean(auc_scores)
print(f"Mean AUC score from 10-fold CV with tuned parameters: {mean_auc:.3f}")

# Assuming test data is available in the same format
# test_data = pd.read_csv('./input/test.csv')
# test_preds = best_model.predict_proba(test_data)[:, 1]

# Save test predictions
# submission = pd.DataFrame({'Id': test_data.index, 'Prediction': test_preds})
# submission.to_csv('results.csv', index=False)
