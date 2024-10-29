import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

# Load the dataset
data = pd.read_csv("dataset.csv")

# Separate features and target
X = data.drop(columns=["A15"])
y = data["A15"].apply(lambda x: int(x.strip("b'")))

# Encode categorical features
for col in X.select_dtypes(include=["object"]).columns:
    X[col] = X[col].apply(lambda x: x.strip("b'"))
    X[col] = LabelEncoder().fit_transform(X[col])

# Initialize the LightGBM model
lgb_model = lgb.LGBMClassifier()

# Prepare cross-validation
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
auc_scores = []

# Perform cross-validation
for train_index, val_index in kf.split(X, y):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    lgb_model.fit(X_train, y_train)
    y_pred = lgb_model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred)
    auc_scores.append(auc)

# Print the average AUC score
print("Average AUC score from 10-fold CV:", sum(auc_scores) / len(auc_scores))

# Train on the entire dataset and predict for submission
lgb_model.fit(X, y)
y_test_pred = lgb_model.predict_proba(X)[:, 1]

# Save the predictions
submission = pd.DataFrame({"Id": data.index, "Prediction": y_test_pred})
submission.to_csv("results.csv", index=False)
