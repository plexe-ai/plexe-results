import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    StackingClassifier,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC

# Load data
data = pd.read_csv("dataset_3_kr-vs-kp.csv")

# Encode categorical variables
label_encoders = {}
for column in data.columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Separate features and target
X = data.drop("class", axis=1)
y = data["class"]

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define base models and their parameter grids
model_params = {
    "rf": {
        "model": RandomForestClassifier(random_state=42),
        "params": {"n_estimators": [100, 200, 300], "max_depth": [10, 20, 30]},
    },
    "gb": {
        "model": GradientBoostingClassifier(random_state=42),
        "params": {"n_estimators": [100, 200, 300], "learning_rate": [0.05, 0.1, 0.2]},
    },
    "svm": {
        "model": SVC(probability=True, kernel="linear", random_state=42),
        "params": {"C": [0.1, 1, 10]},
    },
}

# Perform GridSearchCV and collect best estimators
best_estimators = []
for model_key, model_info in model_params.items():
    grid_search = GridSearchCV(
        model_info["model"], model_info["params"], cv=5, scoring="roc_auc"
    )
    grid_search.fit(X_train, y_train)
    best_estimators.append((model_key, grid_search.best_estimator_))

# Define advanced meta-model
meta_model = GradientBoostingClassifier(
    n_estimators=100, learning_rate=0.1, random_state=42
)

# Create the stacking model with optimized base models
stacking_model = StackingClassifier(
    estimators=best_estimators, final_estimator=meta_model, cv=5
)

# Fit the stacking model
stacking_model.fit(X_train, y_train)

# Predict probabilities for the test set
y_probs = stacking_model.predict_proba(X_test)[:, 1]

# Calculate AUC
auc_score = roc_auc_score(y_test, y_probs)
print(f"AUC Score: {auc_score}")
