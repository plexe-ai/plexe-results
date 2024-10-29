import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

# Load the dataset
data = pd.read_csv("dataset.csv")

# Encode categorical features
label_encoders = {}
for column in ["Airline", "AirportFrom", "AirportTo", "DayOfWeek", "Delay"]:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column].astype(str))
    label_encoders[column] = le

# Split the data
X = data.drop("Delay", axis=1)
y = data["Delay"]

# Define parameter grid for hyperparameter tuning
param_grid = {
    "learning_rate": [0.01, 0.1, 0.2],
    "max_depth": [3, 5, 7],
    "n_estimators": [100, 200, 300],
}

# Initialize XGBoost model
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")

# Initialize GridSearchCV
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    cv=3,
    scoring="roc_auc",
    verbose=1,
    n_jobs=-1,
)

# Fit GridSearchCV
grid_search.fit(X, y)

# Get the best parameters
best_params = grid_search.best_params_
print(f"Best hyperparameters: {best_params}")

# Initialize KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
auc_scores = []

# Train and evaluate the model using 5-fold cross-validation with the best hyperparameters
for train_index, val_index in kf.split(X):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    model = xgb.XGBClassifier(
        **best_params, use_label_encoder=False, eval_metric="logloss"
    )
    model.fit(X_train, y_train)

    y_pred = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred)
    auc_scores.append(auc)

# Print the average AUC score
print(f"Average AUC score: {np.mean(auc_scores):.3f}")

# Train on the full dataset with the best hyperparameters and make predictions for submission
model = xgb.XGBClassifier(**best_params, use_label_encoder=False, eval_metric="logloss")
model.fit(X, y)
test_predictions = model.predict_proba(X)[:, 1]

# Save predictions to submission.csv
submission = pd.DataFrame(
    {"Id": np.arange(len(test_predictions)), "Delay": test_predictions}
)
submission.to_csv("result.csv", index=False)