import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

# Load data
data = pd.read_csv("dataset.csv")

# Preprocess target variable
data["defects"] = data["defects"].apply(lambda x: 1 if x == "b'true'" else 0)

# Features and target
X = data.drop(columns=["defects"])
y = data["defects"]

# Create polynomial features (interaction features)
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(X)

# Initialize model
model = GradientBoostingClassifier(random_state=42)

# Define the hyperparameter space
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.2],
    "min_samples_split": [2, 5, 10],
}

# Set up the GridSearchCV
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
    scoring="roc_auc",
    n_jobs=-1,
)

# Perform the search
grid_search.fit(X_poly, y)

# Evaluate the best model
best_model = grid_search.best_estimator_
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
auc_scores = cross_val_score(best_model, X_poly, y, cv=cv, scoring="roc_auc")

# Print the average AUC score
print(
    f"Average AUC score from 10-fold CV with Gradient Boosting: {np.mean(auc_scores):.3f}"
)

# Train model on entire dataset
best_model.fit(X_poly, y)

# If test data is provided, make predictions and save to submission.csv
# test_data = pd.read_csv('./input/test.csv')
# test_data_poly = poly.transform(test_data)
# test_predictions = best_model.predict_proba(test_data_poly)[:, 1]
# submission = pd.DataFrame({'id': test_data['id'], 'defects': test_predictions})
# submission.to_csv('results.csv', index=False)
