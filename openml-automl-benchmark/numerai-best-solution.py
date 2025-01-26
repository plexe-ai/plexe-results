import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# Load the dataset
data = pd.read_csv("dataset.csv")

# Ensure target variable is integer type
data["attribute_21"] = data["attribute_21"].astype(int)

# Separate features and target
X = data.drop(columns=["attribute_21"])
y = data["attribute_21"]

# Initialize cross-validation
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
auc_scores = []

# 5-fold cross-validation
for train_index, val_index in skf.split(X, y):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    # Create LightGBM datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    # Set LightGBM parameters
    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "verbosity": -1,
        "seed": 42,
    }

    # Train the model
    model = lgb.train(
        params,
        train_data,
        num_boost_round=100,
        valid_sets=[val_data],
        verbose_eval=False
    )

    # Predict and evaluate
    y_pred = model.predict(X_val)
    auc = roc_auc_score(y_val, y_pred)
    auc_scores.append(auc)

# Calculate and print mean AUC score
mean_auc = sum(auc_scores) / n_splits
print(f"Mean AUC score: {mean_auc:.4f}")

# Train on full dataset and save predictions for test set
model = lgb.train(params, lgb.Dataset(X, label=y))
test_predictions = model.predict(X)

# Save predictions to results.csv
results = pd.DataFrame({"Id": data.index, "Prediction": test_predictions})
results.to_csv("results.csv", index=False)
