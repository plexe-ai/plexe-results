import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

# Load the data
train_data = pd.read_csv("/kaggle/input/playground-series-s4e10/train.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s4e10/test.csv")

# Split features and target
X = train_data.drop(["loan_status", "id", ], axis=1)
y = train_data["loan_status"]
X_test = test_data.drop("id", axis=1)

# Preprocessing for numerical data
numerical_cols = [
    cname for cname in X.columns if X[cname].dtype in ["int64", "float64"]
]
numerical_transformer = StandardScaler()

# Preprocessing for categorical data
categorical_cols = [cname for cname in X.columns if X[cname].dtype == "object"]
categorical_transformer = OneHotEncoder(handle_unknown="ignore")
# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols),
    ]
)

# Define the model
gbm_model = GradientBoostingClassifier()

# Create the pipeline
pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", gbm_model)])

# Define the parameter grid
param_grid = {
    "model__n_estimators": [50, 100, 150],
    "model__learning_rate": [0.05, 0.1, 0.2],
    "model__max_depth": [3, 5, 7],
}

# Create the GridSearchCV object
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring="roc_auc", verbose=1)

# Fit the grid search to the data
grid_search.fit(X, y)

# Best model
best_model = grid_search.best_estimator_

# Split data into train and validation subsets
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# Preprocessing of validation data, get predictions
preds = best_model.predict_proba(X_valid)[:, 1]

# Evaluate the model
score = roc_auc_score(y_valid, preds)
print(f"Optimized ROC-AUC score: {score}")

# Preprocess, predict and prepare submission file
test_preds = best_model.predict_proba(X_test)[:, 1]
output = pd.DataFrame({"id": test_data.id, "loan_status": test_preds})
output.to_csv("submission.csv", index=False)
# Fitting 5 folds for each of 27 candidates, totalling 135 fits
# Optimized ROC-AUC score: 0.9794836768468407