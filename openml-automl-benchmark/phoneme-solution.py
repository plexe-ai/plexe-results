import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load the dataset
data = pd.read_csv("dataset.csv")

# Separate features and target
X = data.drop("Class", axis=1)
y = data["Class"]

# Encode the target variable
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Initialize the model
model = ExtraTreesClassifier(random_state=42)

# Define the parameter grid for hyperparameter tuning
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}

# Initialize StratifiedKFold
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Define AUC scorer
auc_scorer = make_scorer(roc_auc_score, needs_proba=True)

# Perform grid search with cross-validation
grid_search = GridSearchCV(
    estimator=model, param_grid=param_grid, scoring=auc_scorer, cv=skf, n_jobs=-1
)
grid_search.fit(X, y_encoded)

# Get the best model
best_model = grid_search.best_estimator_

# Perform 10-fold cross-validation with the best model
auc_scores = []
for train_index, val_index in skf.split(X, y_encoded):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y_encoded[train_index], y_encoded[val_index]

    # Train the best model
    best_model.fit(X_train, y_train)

    # Predict probabilities
    y_pred_proba = best_model.predict_proba(X_val)[:, 1]

    # Calculate AUC score
    auc = roc_auc_score(y_val, y_pred_proba)
    auc_scores.append(auc)

# Print the mean AUC score
print("Mean AUC Score after hyperparameter tuning:", np.mean(auc_scores))

# Train the best model on the full dataset
best_model.fit(X, y_encoded)

# Predict on the full dataset for submission (since no separate test set is provided)
y_pred_proba_full = best_model.predict_proba(X)[:, 1]

# Prepare submission file
submission = pd.DataFrame(
    {"Id": np.arange(len(y_pred_proba_full)), "Prediction": y_pred_proba_full}
)
submission.to_csv("results.csv", index=False)
