"""
train_model.py
--------------
Creates time-series features, trains a RandomForestRegressor on daily
electricity demand, evaluates it and saves it to models/electricity_model.pkl.

The helper functions here (load_data, create_features, split_data) are also
imported by app.py, so the dashboard uses exactly the same logic.

Run:
    python train_model.py
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# File paths (work on Windows, Mac and Linux)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "data", "electricity_demand.csv")
MODEL_FILE = os.path.join(BASE_DIR, "models", "electricity_model.pkl")

# Columns used as model inputs
FEATURE_COLUMNS = [
    "day",
    "day_of_week",
    "month",
    "day_of_year",
    "lag_1",
    "lag_7",
    "lag_14",
    "rolling_mean_7",
    "rolling_mean_14",
]
TARGET_COLUMN = "electricity_demand"
TRAIN_RATIO = 0.8  # 80% training, 20% testing


def load_data(path=DATA_FILE):
    """Load the processed daily CSV and sort it by date."""
    df = pd.read_csv(path, parse_dates=["datetime"])
    return df.sort_values("datetime").reset_index(drop=True)


def create_features(df):
    """
    Add calendar features and lag/rolling features.

    To avoid data leakage, every lag and rolling feature only uses PAST days:
      - lag_1  = demand of the previous day
      - lag_7  = demand of the same weekday last week
      - lag_14 = demand two weeks ago
      - rolling_mean_7 / 14 = average of the previous 7 / 14 days
        (shift(1) first, so today's value is never included)
    """
    df = df.copy()

    # Calendar (time-based) features
    df["day"] = df["datetime"].dt.day
    df["day_of_week"] = df["datetime"].dt.dayofweek  # 0 = Monday, 6 = Sunday
    df["month"] = df["datetime"].dt.month
    df["day_of_year"] = df["datetime"].dt.dayofyear

    # Lag features (values from earlier days)
    demand = df[TARGET_COLUMN]
    df["lag_1"] = demand.shift(1)
    df["lag_7"] = demand.shift(7)
    df["lag_14"] = demand.shift(14)

    # Rolling averages of past values only
    past_demand = demand.shift(1)
    df["rolling_mean_7"] = past_demand.rolling(window=7).mean()
    df["rolling_mean_14"] = past_demand.rolling(window=14).mean()

    return df


def split_data(df):
    """
    Chronological train/test split (NO shuffling).
    The first 80% of days are used for training, the last 20% for testing.
    """
    split_index = int(len(df) * TRAIN_RATIO)
    train = df.iloc[:split_index]
    test = df.iloc[split_index:]
    return train, test


def evaluate(actual, predicted):
    """Return MAE, RMSE and R² as a dictionary."""
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "R2": r2_score(actual, predicted),
    }


def main():
    # 1. Load data and create features
    df = load_data()
    df = create_features(df)

    # The first 14 days do not have enough history for lag_14 -> drop them
    df = df.dropna().reset_index(drop=True)

    # 2. Chronological split
    train, test = split_data(df)
    print(f"Training days: {len(train)} ({train['datetime'].min().date()} to {train['datetime'].max().date()})")
    print(f"Testing days:  {len(test)} ({test['datetime'].min().date()} to {test['datetime'].max().date()})")

    X_train, y_train = train[FEATURE_COLUMNS], train[TARGET_COLUMN]
    X_test, y_test = test[FEATURE_COLUMNS], test[TARGET_COLUMN]

    # 3. Train the Random Forest model
    model = RandomForestRegressor(
        n_estimators=200,     # number of decision trees
        max_depth=10,         # limit tree depth to reduce overfitting
        min_samples_leaf=3,
        random_state=42,      # makes results reproducible
        n_jobs=-1,            # use all CPU cores
    )
    model.fit(X_train, y_train)

    # 4. Evaluate on the unseen test period
    predictions = model.predict(X_test)
    metrics = evaluate(y_test, predictions)
    print("\nModel performance on test data:")
    print(f"  MAE : {metrics['MAE']:.2f} kWh")
    print(f"  RMSE: {metrics['RMSE']:.2f} kWh")
    print(f"  R2  : {metrics['R2']:.3f}")

    # Simple baseline for comparison: "tomorrow = today"
    baseline = evaluate(y_test, test["lag_1"])
    print(f"\nBaseline (yesterday's value) MAE: {baseline['MAE']:.2f} kWh")

    # 5. Save the trained model
    os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    print(f"\nModel saved to: {MODEL_FILE}")


if __name__ == "__main__":
    main()
