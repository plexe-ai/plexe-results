import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, make_scorer
from catboost import CatBoostClassifier, Pool

# Load data
data = pd.read_csv("dataset.csv")

# Encode target variable
data["class"] = data["class"].apply(lambda x: 1 if x == "b'good'" else 0)

# Separate features and target
X = data.drop("class", axis=1)
y = data["class"]

# Identify categorical features
cat_features = X.select_dtypes(include=["object"]).columns.tolist()

# Initialize model
model = CatBoostClassifier(cat_features=cat_features, verbose=0)

# Define parameter grid
param_grid = {
    "iterations": [500, 1000, 1500],
    "learning_rate": [0.01, 0.05, 0.1],
    "depth": [4, 6, 8],
}

# Custom scorer
scorer = make_scorer(roc_auc_score, needs_proba=True)

# Cross-validation
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Grid Search
grid_search = GridSearchCV(
    estimator=model, param_grid=param_grid, scoring=scorer, cv=kf, n_jobs=-1
)
grid_search.fit(X, y)

# Best model
best_model = grid_search.best_estimator_

# Evaluate best model
auc_scores = []
for train_index, val_index in kf.split(X, y):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    val_pool = Pool(X_val, y_val, cat_features=cat_features)

    best_model.fit(train_pool)
    y_pred = best_model.predict_proba(val_pool)[:, 1]
    auc = roc_auc_score(y_val, y_pred)
    auc_scores.append(auc)

# Print mean AUC score
print(f"Mean AUC score: {np.mean(auc_scores):.4f}")

# Train on full dataset
train_pool = Pool(X, y, cat_features=cat_features)
best_model.fit(train_pool)

# Save predictions
predictions = best_model.predict_proba(train_pool)[:, 1]
submission = pd.DataFrame({"Id": data.index, "Probability": predictions})
submission.to_csv("results.csv", index=False)
