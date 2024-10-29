import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import RFE

# Load data
train_data = pd.read_csv("/kaggle/input/spaceship-titanic/train.csv")
test_data = pd.read_csv("/kaggle/input/spaceship-titanic/test.csv")

# Preprocessing
train_data["CryoSleep"] = train_data["CryoSleep"].astype("category")
train_data["VIP"] = train_data["VIP"].astype("category")
train_data["HomePlanet"] = train_data["HomePlanet"].astype("category")
train_data["Destination"] = train_data["Destination"].astype("category")

# Fill missing values
train_data.fillna(
    {
        "Age": train_data["Age"].median(),
        "FoodCourt": 0,
        "RoomService": 0,
        "ShoppingMall": 0,
        "Spa": 0,
        "VRDeck": 0,
    },
    inplace=True,
)

# Convert categorical variables to numerical
label_encoders = {}
for column in ["HomePlanet", "CryoSleep", "Destination", "VIP"]:
    le = LabelEncoder()
    train_data[column] = le.fit_transform(train_data[column].astype(str))
    label_encoders[column] = le

# Creating interaction features
poly = PolynomialFeatures(interaction_only=True, include_bias=False)
interaction_features = poly.fit_transform(
    train_data.drop(["PassengerId", "Transported", "Name", "Cabin"], axis=1)
)

# Features and target
X = pd.DataFrame(interaction_features)
y = train_data["Transported"].astype(int)

# Feature selection using RFE
model = RandomForestClassifier(random_state=42)
rfe = RFE(model, n_features_to_select=20)  # Select top 20 features
X_rfe = rfe.fit_transform(X, y)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42), param_grid, cv=5, scoring="accuracy"
)
grid_search.fit(X_rfe, y)

# Best model from grid search
best_model = grid_search.best_estimator_

# Cross-validation score of the best model
cv_scores = cross_val_score(best_model, X_rfe, y, cv=5, scoring="accuracy")

# Fit model and predict on test data
best_model.fit(X_rfe, y)
test_data.fillna(
    {
        "Age": test_data["Age"].median(),
        "FoodCourt": 0,
        "RoomService": 0,
        "ShoppingMall": 0,
        "Spa": 0,
        "VRDeck": 0,
    },
    inplace=True,
)

for column in ["HomePlanet", "CryoSleep", "Destination", "VIP"]:
    test_data[column] = label_encoders[column].transform(test_data[column].astype(str))

X_test = pd.DataFrame(
    poly.transform(test_data.drop(["PassengerId", "Name", "Cabin"], axis=1))
)
X_test_rfe = rfe.transform(X_test)  # Apply the same feature selection

predictions = best_model.predict(X_test_rfe)

# Save predictions
submission = pd.DataFrame(
    {"PassengerId": test_data["PassengerId"], "Transported": predictions}
)
submission["Transported"] = submission["Transported"].astype(bool)
submission.to_csv("submission.csv", index=False)

# Print evaluation metric
print(f"Cross-Validation Accuracy: {cv_scores.mean():.4f}")
