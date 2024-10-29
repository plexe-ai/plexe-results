from sklearn.model_selection import RandomizedSearchCV, KFold
import lightgbm as lgb
import numpy as np

# Define parameter grid for hyperparameter tuning
param_grid = {
    "num_leaves": np.arange(20, 150, 10),
    "max_depth": np.arange(3, 15, 2),
    "learning_rate": np.linspace(0.01, 0.3, 10),
    "feature_fraction": np.linspace(0.5, 1.0, 5),
}

# Initialize LightGBM classifier
lgb_model = lgb.LGBMClassifier()

# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=lgb_model,
    param_distributions=param_grid,
    n_iter=50,
    scoring="roc_auc",
    cv=KFold(n_splits=10, shuffle=True, random_state=42),
    random_state=42,
    n_jobs=-1,
)

# Fit RandomizedSearchCV
random_search.fit(features, target)

# Best model from the search
best_model = random_search.best_estimator_

# 10-fold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)
auc_scores = []

for train_index, val_index in kf.split(features):
    X_train, X_val = features.iloc[train_index], features.iloc[val_index]
    y_train, y_val = target.iloc[train_index], target.iloc[val_index]

    # Train the best model
    best_model.fit(X_train, y_train)

    # Predict and evaluate AUC score
    y_pred = best_model.predict_proba(X_val)[:, 1]
    auc_score = roc_auc_score(y_val, y_pred)
    auc_scores.append(auc_score)

# Calculate average AUC score
average_auc_score = np.mean(auc_scores)
print(f"Average AUC Score: {average_auc_score}")
