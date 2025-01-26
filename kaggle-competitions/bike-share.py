import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_log_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Load data
train = pd.read_csv("/kaggle/input/bike-sharing-demand/train.csv")
test = pd.read_csv("/kaggle/input/bike-sharing-demand/test.csv")


# Feature engineering
def preprocess_data(df):
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["hour"] = df["datetime"].dt.hour
    df["day"] = df["datetime"].dt.day
    df["month"] = df["datetime"].dt.month
    df["year"] = df["datetime"].dt.year
    df.drop("datetime", axis=1, inplace=True)
    return df


train = preprocess_data(train)
test = preprocess_data(test)

# Define features and target
X = train.drop(["casual", "registered", "count"], axis=1)
y = np.log1p(train["count"])


# Define RMSLE scorer
def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(y_true, y_pred))


rmsle_scorer = make_scorer(rmsle, greater_is_better=False)

# Initialize model
model = GradientBoostingRegressor()

# Hyperparameter tuning
param_grid = {
    "model__n_estimators": [100, 200],
    "model__learning_rate": [0.05, 0.1],
    "model__max_depth": [3, 5],
}

# Create a pipeline with scaling and model
pipeline = Pipeline([("scaler", StandardScaler()), ("model", model)])

grid_search = GridSearchCV(
    estimator=pipeline, param_grid=param_grid, cv=5, scoring=rmsle_scorer, n_jobs=-1
)
grid_search.fit(X, y)

# Best model
best_model = grid_search.best_estimator_

# Cross-validation with best model
kf = KFold(n_splits=10, shuffle=True, random_state=42)
rmsle_scores = []

for train_index, val_index in kf.split(X):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_val)
    rmsle = np.sqrt(mean_squared_log_error(y_val, y_pred))
    rmsle_scores.append(rmsle)

# Print evaluation metric
print("Mean RMSLE:", np.mean(rmsle_scores))

# Train on full data and predict on test set
best_model.fit(X, y)
test_predictions = np.expm1(best_model.predict(test))

# Prepare submission
submission = pd.DataFrame(
    {"datetime": pd.read_csv("/kaggle/input/bike-sharing-demand/test.csv")["datetime"], "count": test_predictions}
)
submission.to_csv("submission.csv", index=False)
