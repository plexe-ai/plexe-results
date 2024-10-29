import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# Load the dataset
data = pd.read_csv("dataset.csv")

# Encode categorical features
label_encoders = {}
for column in data.columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Split data into features and target
X = data.drop("Class", axis=1)
y = data["Class"]

# Initialize RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Perform 10-fold cross-validation
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
auc_scores = cross_val_score(rf_model, X, y, cv=cv, scoring="roc_auc")

# Print the mean AUC score
print(f"Mean AUC Score: {auc_scores.mean():.3f}")

# Train the model on the full dataset
rf_model.fit(X, y)

# Make predictions on the test data (assuming test data is the same as train for this example)
test_predictions = rf_model.predict_proba(X)[:, 1]

# Save predictions to submission.csv
submission = pd.DataFrame(
    {"Id": range(len(test_predictions)), "Prediction": test_predictions}
)
submission.to_csv("results.csv", index=False)
