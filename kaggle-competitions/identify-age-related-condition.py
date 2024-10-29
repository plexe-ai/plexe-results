import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Load data
train = pd.read_csv("/kaggle/input/icr-identify-age-related-conditions/train.csv")
greeks = pd.read_csv("/kaggle/input/icr-identify-age-related-conditions/greeks.csv")
test = pd.read_csv("/kaggle/input/icr-identify-age-related-conditions/test.csv")

# Merge train with greeks on 'Id'
train = train.merge(greeks, on="Id", how="left")

# Prepare features and target
X = train.drop(columns=["Id", "Class"])
y = train["Class"]

# Identify categorical columns
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

# Preprocessing for numerical data
numerical_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]
)

# Preprocessing for categorical data
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, X.select_dtypes(exclude=["object"]).columns),
        ("cat", categorical_transformer, categorical_cols),
    ]
)

# Create a pipeline that includes preprocessing and model
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000)),
    ]
)

# 10-fold cross-validation
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
y_pred_proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")

# Calculate balanced log loss
loss = log_loss(y, y_pred_proba)
print(f"Balanced Logarithmic Loss: {loss}")

# Train model on full data
model.fit(X, y)

# Prepare test data
test = test.merge(greeks, on="Id", how="left")
X_test = test.drop(columns=["Id"])

# Predict test probabilities
test_pred_proba = model.predict_proba(X_test)

# Prepare submission
submission = pd.DataFrame(test[["Id"]])
submission["class_0"] = test_pred_proba[:, 0]
submission["class_1"] = test_pred_proba[:, 1]
submission.to_csv("submission.csv", index=False)