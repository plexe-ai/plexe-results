import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from lightgbm import LGBMClassifier
from sklearn.metrics import log_loss

# Load the dataset
data = pd.read_csv("dataset.csv")

# Split into features and target
X = data.drop(columns=["target"])
y = data["target"]

# Generate polynomial features
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
X_poly = poly.fit_transform(X)

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Initialize LightGBM Classifier
model = LGBMClassifier()

# Perform 10-fold cross-validation
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
logloss_scores = cross_val_score(
    model, X_poly, y_encoded, cv=cv, scoring="neg_log_loss"
)

# Print mean log loss
mean_logloss = -logloss_scores.mean()
print(f"Mean Log Loss: {mean_logloss}")

# Train on the entire dataset
model.fit(X_poly, y_encoded)

# Predict on the same dataset for demonstration (no separate test set provided)
predictions = model.predict_proba(X_poly)

# Prepare submission file
submission = pd.DataFrame(predictions, columns=label_encoder.classes_)
submission.to_csv("results.csv", index=False)
