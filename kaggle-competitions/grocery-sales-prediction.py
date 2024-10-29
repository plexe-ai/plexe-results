import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load data
train_data = pd.read_csv("/kaggle/input/store-sales-time-series-forecasting/train.csv")
test_data = pd.read_csv("/kaggle/input/store-sales-time-series-forecasting/test.csv")

# Feature engineering
train_data["date"] = pd.to_datetime(train_data["date"])
test_data["date"] = pd.to_datetime(test_data["date"])

# Extract date features
train_data["month"] = train_data["date"].dt.month
train_data["day_of_week"] = train_data["date"].dt.dayofweek
test_data["month"] = test_data["date"].dt.month
test_data["day_of_week"] = test_data["date"].dt.dayofweek

# Encode categorical variables
label_enc = LabelEncoder()
train_data["family"] = label_enc.fit_transform(train_data["family"])
test_data["family"] = label_enc.transform(test_data["family"])

# Prepare features and target
X = train_data[["store_nbr", "family", "onpromotion", "month", "day_of_week"]]
y = train_data["sales"]
X_test = test_data[["store_nbr", "family", "onpromotion", "month", "day_of_week"]]

# Split data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Prediction on validation set
y_pred_val = model.predict(X_val)
y_pred_val = np.clip(y_pred_val, 0, None)  # Ensure no negative predictions

# Evaluation using RMSLE
rmsle = np.sqrt(mean_squared_log_error(y_val, y_pred_val))
print(f"Validation RMSLE: {rmsle}")

# Predict on test set
y_pred_test = model.predict(X_test)
y_pred_test = np.clip(y_pred_test, 0, None)  # Ensure no negative predictions

# Create submission file
submission = pd.DataFrame({"id": test_data["id"], "sales": y_pred_test})
submission.to_csv("submission.csv", index=False)
