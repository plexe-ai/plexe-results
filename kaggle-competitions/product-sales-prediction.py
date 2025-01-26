import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor

# Load data
sales = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv")
test = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/test.csv")

# Convert date column to datetime
sales["date"] = pd.to_datetime(sales["date"], format="%d.%m.%Y")

# Aggregate data to monthly level
monthly_sales = (
    sales.groupby(["date_block_num", "shop_id", "item_id"])
    .agg({"item_cnt_day": "sum"})
    .reset_index()
)
monthly_sales.rename(columns={"item_cnt_day": "item_cnt_month"}, inplace=True)

# Create lag features
lags = [1, 2, 3, 6, 12]
for lag in lags:
    lagged_sales = monthly_sales.copy()
    lagged_sales["date_block_num"] += lag
    lagged_sales.rename(
        columns={"item_cnt_month": f"item_cnt_month_lag_{lag}"}, inplace=True
    )
    monthly_sales = monthly_sales.merge(
        lagged_sales[
            ["date_block_num", "shop_id", "item_id", f"item_cnt_month_lag_{lag}"]
        ],
        on=["date_block_num", "shop_id", "item_id"],
        how="left",
    )

# Rolling window features
window_sizes = [3, 6, 12]
for window in window_sizes:
    monthly_sales[f"rolling_mean_{window}"] = monthly_sales.groupby(
        ["shop_id", "item_id"]
    )["item_cnt_month"].transform(
        lambda x: x.rolling(window=window, min_periods=1).mean()
    )
    monthly_sales[f"rolling_std_{window}"] = monthly_sales.groupby(
        ["shop_id", "item_id"]
    )["item_cnt_month"].transform(
        lambda x: x.rolling(window=window, min_periods=1).std()
    )

# Fill NaN values
monthly_sales.fillna(0, inplace=True)

# Clip target range
monthly_sales["item_cnt_month"] = monthly_sales["item_cnt_month"].clip(0, 20)

# Adding month and year from date_block_num
monthly_sales["month"] = monthly_sales["date_block_num"] % 12 + 1
monthly_sales["year"] = monthly_sales["date_block_num"] // 12 + 2013

# Prepare training data
X = monthly_sales.drop(["item_cnt_month"], axis=1)
y = monthly_sales["item_cnt_month"]

# Train/validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = LGBMRegressor(max_depth=8, n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_val)

# RMSE
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f"Validation RMSE: {rmse}")

# Prepare test set features
test = test.merge(
    monthly_sales.drop("item_cnt_month", axis=1), on=["shop_id", "item_id"], how="left"
)
test.fillna(0, inplace=True)

# Predict on test set
test["item_cnt_month"] = model.predict(test.drop(["ID"], axis=1)).clip(0, 20)
test[["ID", "item_cnt_month"]].to_csv("submission.csv", index=False)
