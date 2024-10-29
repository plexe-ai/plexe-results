import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import RFE
import lightgbm as lgb

# Load the dataset
data = pd.read_csv("dataset.csv")

# Prepare features and target
X = data.drop(columns=["class"])
y = data["class"].apply(lambda x: 1 if x == "b'2'" else 0)

# Create polynomial features
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(X)

# Initialize the model for feature selection
base_model = lgb.LGBMClassifier()

# Perform Recursive Feature Elimination
rfe = RFE(estimator=base_model, n_features_to_select=10)  # Select top 10 features
X_selected = rfe.fit_transform(X_poly, y)

# Initialize the model for training
model = lgb.LGBMClassifier()

# Prepare cross-validation
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
auc_scores = []

# Perform cross-validation
for train_index, val_index in kf.split(X_selected, y):
    X_train, X_val = X_selected[train_index], X_selected[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred)
    auc_scores.append(auc)

# Print the mean AUC score
print("Mean AUC Score:", np.mean(auc_scores))

# Save predictions for test set (assuming a separate test set is available)
# Here we assume the same data is used for demonstration
test_predictions = model.predict_proba(X_selected)[:, 1]
submission = pd.DataFrame({"Id": data.index, "Prediction": test_predictions})
submission.to_csv("results.csv", index=False)
