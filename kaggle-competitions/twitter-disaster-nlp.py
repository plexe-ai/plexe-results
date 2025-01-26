import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

# Load the data
train_data = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
test_data = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")

# Prepare train and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    train_data["text"], train_data["target"], test_size=0.2, random_state=42
)

# Create a pipeline with TF-IDF Vectorizer and SVM
pipeline = Pipeline(
    [
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("svm", SVC(kernel="linear", random_state=42)),
    ]
)

# Train the model
pipeline.fit(X_train, y_train)

# Validate the model
y_pred = pipeline.predict(X_val)
f1 = f1_score(y_val, y_pred)
print(f"Validation F1 Score: {f1}")

# Predict on test data
test_predictions = pipeline.predict(test_data["text"])

# Save the predictions to a CSV file
submission = pd.DataFrame({"id": test_data["id"], "target": test_predictions})
submission.to_csv("submission.csv", index=False)
