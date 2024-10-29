import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import numpy as np

# Load the dataset
data = pd.read_csv("dataset.csv")

# Preprocess the data
# Convert class labels to binary (1 and 0)
data["Class"] = data["Class"].apply(lambda x: 1 if x == "b'1'" else 0)

# Split features and target
X = data.drop("Class", axis=1)
y = data["Class"]

# Initialize LightGBM model
model = lgb.LGBMClassifier()

# Evaluate with 10-fold cross-validation
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
auc_scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")

# Print the mean AUC score
print("Mean AUC score:", np.mean(auc_scores))

# Train the model on the entire dataset
model.fit(X, y)

# Predict on the same dataset (as no separate test set is provided)
predictions = model.predict_proba(X)[:, 1]

# Save the predictions to a submission file
submission = pd.DataFrame({"id": range(len(predictions)), "Class": predictions})
submission.to_csv("results.csv", index=False)
