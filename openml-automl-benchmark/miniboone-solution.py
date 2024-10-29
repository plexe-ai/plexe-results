import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import roc_auc_score
import xgboost as xgb

# Load the dataset
data = pd.read_csv("dataset.csv")

# Preprocess the target variable
data["signal"] = data["signal"].apply(lambda x: 1 if x == "b'True'" else 0)

# Separate features and target
X = data.drop(columns=["signal"])
y = data["signal"]

# Define the XGBoost classifier
model = xgb.XGBClassifier(
    use_label_encoder=False, eval_metric="logloss", random_state=42
)

# Define the parameter grid for RandomizedSearchCV
param_grid = {
    "n_estimators": [100, 200, 300, 400],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "max_depth": [3, 5, 7, 9],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
}

# Initialize the RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_grid,
    n_iter=20,
    scoring="roc_auc",
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1,
)

# Fit RandomizedSearchCV
random_search.fit(X, y)

# Get the best parameters
best_params = random_search.best_params_
print("Best hyperparameters:", best_params)

# Initialize the cross-validation
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
auc_scores = []

# Perform cross-validation with the best parameters
for train_index, val_index in kf.split(X, y):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    # Define the XGBoost classifier with best parameters
    model = xgb.XGBClassifier(
        use_label_encoder=False, eval_metric="logloss", random_state=42, **best_params
    )

    # Train the model
    model.fit(X_train, y_train)

    # Predict on validation set
    y_pred = model.predict_proba(X_val)[:, 1]

    # Calculate AUC score
    auc = roc_auc_score(y_val, y_pred)
    auc_scores.append(auc)

# Print the mean AUC score
print("Mean AUC score from 10-fold cross-validation:", np.mean(auc_scores))

# Train on the entire dataset with best parameters
model.fit(X, y)

# Make predictions on the test set
y_test_pred = model.predict_proba(X)[:, 1]

# Create a submission file
submission = pd.DataFrame({"Id": data.index, "Prediction": y_test_pred})
submission.to_csv("results.csv", index=False)
