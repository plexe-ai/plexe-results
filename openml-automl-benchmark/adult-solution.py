import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
import optuna

# Load dataset
data = pd.read_csv("dataset.csv")

# Preprocessing data
X = data.drop("class", axis=1)
y = (data["class"] == "b'>50K'").astype(int)

# Splitting data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Selecting categorical columns
categorical_cols = X_train.select_dtypes(include=["object"]).columns

# Creating a preprocessor for one-hot encoding of categorical variables
preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)],
    remainder="passthrough",
)


# Defining the objective function for Optuna
def objective(trial):
    param = {
        "num_leaves": trial.suggest_int("num_leaves", 20, 200),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
    }

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LGBMClassifier(**param, random_state=42)),
        ]
    )

    model.fit(X_train, y_train)
    y_pred_prob = model.predict_proba(X_val)[:, 1]
    auc_score = roc_auc_score(y_val, y_pred_prob)
    return auc_score


# Create a study object and optimize the objective function
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

# Best parameters and score
best_params = study.best_params
best_score = study.best_value
print(f"Best AUC Score: {best_score}")
print(f"Best Parameters: {best_params}")
