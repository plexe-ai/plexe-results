import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score
from skopt import BayesSearchCV
from catboost import CatBoostClassifier
from sklearn.feature_selection import RFE

# Load data
data = pd.read_csv("dataset.csv")

# Encode the target variable
label_encoder = LabelEncoder()
data["class"] = label_encoder.fit_transform(data["class"])

# Split features and target
X = data.drop("class", axis=1)
y = data["class"]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize the CatBoost Classifier
catboost_classifier = CatBoostClassifier(random_state=42, silent=True)

# Perform Recursive Feature Elimination (RFE)
rfe = RFE(estimator=catboost_classifier, n_features_to_select=10)
X_rfe = rfe.fit_transform(X_scaled, y)

# Define the parameter search space for Bayesian Optimization
param_space = {
    "iterations": (50, 300),
    "depth": (3, 10),
    "learning_rate": (0.01, 0.3, "log-uniform"),
    "l2_leaf_reg": (1, 10),
    "bagging_temperature": (0.1, 1.0),
}

# Initialize BayesSearchCV
bayes_search = BayesSearchCV(
    estimator=catboost_classifier,
    search_spaces=param_space,
    n_iter=50,
    cv=10,
    scoring="roc_auc",
    n_jobs=-1,
    random_state=42,
)

# Fit the model using BayesSearchCV on selected features
bayes_search.fit(X_rfe, y)

# Print the best parameters and best score
print(f"Best parameters: {bayes_search.best_params_}")
print(f"Best AUC score from BayesSearch: {bayes_search.best_score_}")

# Use the best estimator to make predictions
best_catboost_classifier = bayes_search.best_estimator_

# Predict on the same data for submission (since no separate test set is provided)
predictions = best_catboost_classifier.predict_proba(X_rfe)[:, 1]

# Save predictions to submission file
submission = pd.DataFrame(
    {"id": np.arange(len(predictions)), "prediction": predictions}
)
submission.to_csv("results.csv", index=False)
