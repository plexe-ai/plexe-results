import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

# Load the dataset
data = pd.read_csv("dataset.csv")

# Preprocessing the data
X = data.drop("Class", axis=1)
y = data["Class"].apply(lambda x: 1 if x == "b'1'" else 0)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Identifying categorical and numerical features
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
numerical_features = X.select_dtypes(exclude=["object"]).columns.tolist()

# Creating transformers for numerical and categorical data
numerical_transformer = Pipeline(steps=[("scaler", StandardScaler())])

categorical_transformer = Pipeline(
    steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
)

# Column transformer to apply the transformations to the respective columns
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# Creating a pipeline that first transforms the data and then applies the classifier
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(use_label_encoder=False, eval_metric="logloss")),
    ]
)

# Setting up the parameter grid
param_grid = {
    "classifier__max_depth": [3, 5, 7, 9, 11],
    "classifier__n_estimators": [50, 100, 150, 200, 250],
    "classifier__learning_rate": [0.01, 0.05, 0.1, 0.2],
    "classifier__min_child_weight": [1, 2, 3, 4, 5],
}

# Grid search with cross-validation
grid_search = GridSearchCV(model, param_grid, scoring="roc_auc", cv=5, verbose=1)
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_

# Predicting the test set results
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

# Calculating the AUC score
auc_score = roc_auc_score(y_test, y_pred_proba)

print(f"Validation AUC Score: {auc_score}")
