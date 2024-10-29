import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from sklearn.preprocessing import PolynomialFeatures

# Load the dataset
data = pd.read_csv("dataset.csv")

# Encode the target variable
data["c"] = data["c"].apply(lambda x: 1 if x == "b'TRUE'" else 0)

# Separate features and target
X = data.drop(columns=["c"])
y = data["c"]

# Apply Polynomial Features for feature engineering
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(X)

# Initialize the LightGBM classifier
model = lgb.LGBMClassifier()

# Initialize KFold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)
auc_scores = []

# Perform cross-validation
for train_index, val_index in kf.split(X_poly):
    X_train, X_val = X_poly[train_index], X_poly[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    # Train the model
    model.fit(X_train, y_train)

    # Predict probabilities
    y_pred = model.predict_proba(X_val)[:, 1]

    # Calculate AUC score
    auc = roc_auc_score(y_val, y_pred)
    auc_scores.append(auc)

# Print the mean AUC score
print("Mean AUC score:", np.mean(auc_scores))

# Predict on the entire dataset (for submission purpose)
y_test_pred = model.predict_proba(X_poly)[:, 1]

# Save the predictions to a CSV file
submission = pd.DataFrame({"Id": data.index, "Predicted": y_test_pred})
submission.to_csv("results.csv", index=False)
