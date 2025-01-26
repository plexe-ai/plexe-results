import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Load data
train_data = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
test_data = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")

# Feature engineering: Adding interaction terms
train_data["Bsmt_x_1stFlrSF"] = train_data["TotalBsmtSF"] * train_data["1stFlrSF"]
train_data["GrLivArea_x_TotRmsAbvGrd"] = (
    train_data["GrLivArea"] * train_data["TotRmsAbvGrd"]
)
test_data["Bsmt_x_1stFlrSF"] = test_data["TotalBsmtSF"] * test_data["1stFlrSF"]
test_data["GrLivArea_x_TotRmsAbvGrd"] = (
    test_data["GrLivArea"] * test_data["TotRmsAbvGrd"]
)

# Prepare the data
X = train_data.drop(["SalePrice", "Id"], axis=1)
y = np.log(train_data["SalePrice"])  # Use log transformation
X_test = test_data.drop(["Id"], axis=1)

# Preprocessing for numerical data
numerical_cols = [
    cname for cname in X.columns if X[cname].dtype in ["int64", "float64"]
]
numerical_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)

# Preprocessing for categorical data
categorical_cols = [cname for cname in X.columns if X[cname].dtype == "object"]
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols),
    ]
)

# Define the model
model = GradientBoostingRegressor(random_state=0)

# Create a pipeline
pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

# Define GridSearchCV
param_grid = {
    "model__n_estimators": [100, 200, 300],
    "model__learning_rate": [0.01, 0.05, 0.1],
    "model__max_depth": [3, 4, 5],
    "model__max_features": ["sqrt", "log2", None],  # Adding max_features to grid search
    "model__subsample": [0.8, 0.9, 1.0],  # Adding subsample to grid search
}
search = GridSearchCV(
    pipeline, param_grid, cv=5, scoring="neg_mean_squared_error", verbose=1
)

# Fit the model
search.fit(X, y)

# Best model score
best_score = np.sqrt(-search.best_score_)
print("Best RMSE:", best_score)

# Preprocess test data, fit model
preds_test = search.predict(X_test)

# Save test predictions to file
output = pd.DataFrame({"Id": test_data.Id, "SalePrice": np.exp(preds_test)})
output.to_csv("submission.csv", index=False)
