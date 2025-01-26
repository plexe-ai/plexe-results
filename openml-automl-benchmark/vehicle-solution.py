import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    StackingClassifier,
)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import numpy as np

# Load the dataset
data = pd.read_csv("dataset.csv")

# Encode the target variable
le = LabelEncoder()
data["Class"] = le.fit_transform(data["Class"])

# Split the dataset into features and target
X = data.drop("Class", axis=1)
y = data["Class"]

# Define base models
base_models = [
    ("rf", RandomForestClassifier(n_estimators=100, random_state=42)),
    ("gb", GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ("svm", SVC(probability=True, kernel="linear", random_state=42)),
]

# Define the meta-model
meta_model = LogisticRegression(random_state=42)

# Initialize the Stacking Classifier
stacking_clf = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5,
    stack_method="predict_proba",
)

# Perform 10-fold cross-validation
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
auc_scores = []

for train_idx, test_idx in cv.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    stacking_clf.fit(X_train, y_train)
    y_pred_proba = stacking_clf.predict_proba(X_test)
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3])
    auc = roc_auc_score(y_test_bin, y_pred_proba, multi_class="ovr")
    auc_scores.append(auc)

mean_auc_score = np.mean(auc_scores)
print(f"Mean AUC Score: {mean_auc_score}")

# Train the model on the entire dataset
stacking_clf.fit(X, y)

# Generate predictions for the test data
test_data = pd.read_csv(
    "./input/dataset.csv"
)  # Assuming the test data is in the same format
test_data["Class"] = le.transform(test_data["Class"])
X_test = test_data.drop("Class", axis=1)
y_test_pred_proba = stacking_clf.predict_proba(X_test)

# Save predictions to submission.csv
submission = pd.DataFrame(y_test_pred_proba, columns=le.classes_)
submission.to_csv("results.csv", index=False)
