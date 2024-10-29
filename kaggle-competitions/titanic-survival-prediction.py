import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load data
train_df = pd.read_csv("/kaggle/input/titanic/train.csv")
test_df = pd.read_csv("/kaggle/input/titanic/test.csv")

# Feature engineering: create FamilySize feature
train_df["FamilySize"] = train_df["SibSp"] + train_df["Parch"] + 1
test_df["FamilySize"] = test_df["SibSp"] + test_df["Parch"] + 1


# Feature engineering: create FamilyType feature
def family_type(size):
    if size == 1:
        return "Alone"
    elif size <= 4:
        return "Small"
    else:
        return "Large"


train_df["FamilyType"] = train_df["FamilySize"].apply(family_type)
test_df["FamilyType"] = test_df["FamilySize"].apply(family_type)

# Separate features and target
X = train_df.drop(columns=["Survived", "PassengerId", "Name", "Ticket", "Cabin"])
y = train_df["Survived"]
X_test = test_df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])

# Preprocess data
numeric_features = ["Age", "Fare", "SibSp", "Parch", "FamilySize"]
numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)

categorical_features = ["Pclass", "Sex", "Embarked", "FamilyType"]
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# Fit and transform the data
X = preprocessor.fit_transform(X)
X_test = preprocessor.transform(X_test)

# Initialize model
model = GradientBoostingClassifier(random_state=42)

# Evaluate model using 5-fold cross-validation
cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
print(f"Cross-validated accuracy: {np.mean(cv_scores):.4f}")

# Train the model on the entire training data
model.fit(X, y)

# Predict on the test set
predictions = model.predict(X_test)

# Create submission file
submission = pd.DataFrame(
    {"PassengerId": test_df["PassengerId"], "Survived": predictions}
)
submission.to_csv("submission.csv", index=False)
# Cross-validated accuracy: 0.8350