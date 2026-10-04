"""
preprocessing.py
----------------
Converts the raw UCI "Individual Household Electric Power Consumption"
dataset (minute-level .txt file) into a clean daily CSV file.

Output: data/electricity_demand.csv with columns
    datetime, electricity_demand

electricity_demand = total energy used in one day (kWh).

Run:
    python preprocessing.py
"""

import os

import pandas as pd

# Build file paths relative to this script so they work on Windows, Mac and Linux
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_FILE = os.path.join(BASE_DIR, "data", "household_power_consumption.txt")
OUTPUT_FILE = os.path.join(BASE_DIR, "data", "electricity_demand.csv")


def load_raw_data(path):
    """Load the raw semicolon-separated text file."""
    print(f"Loading raw data from: {path}")
    # In this dataset missing values are written as '?'
    df = pd.read_csv(
        path,
        sep=";",
        usecols=["Date", "Time", "Global_active_power"],
        na_values=["?"],
        low_memory=False,
    )
    print(f"Loaded {len(df):,} minute-level rows")
    return df


def clean_data(df):
    """Create a datetime column, convert power to numbers and sort by time."""
    # 1. Combine Date and Time into one datetime column (date format is day/month/year)
    df["datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S"
    )

    # 2. Make sure power values are numeric (anything invalid becomes NaN)
    df["Global_active_power"] = pd.to_numeric(df["Global_active_power"], errors="coerce")

    # 3. Keep only the columns we need and sort chronologically
    df = df[["datetime", "Global_active_power"]].sort_values("datetime")

    missing = df["Global_active_power"].isna().sum()
    print(f"Missing power readings: {missing:,} ({missing / len(df):.2%})")
    return df


def resample_to_daily(df):
    """
    Convert minute-level readings into daily electricity demand (kWh).

    Global_active_power is measured in kilowatts (kW). The average power of a
    day multiplied by 24 hours gives the energy used that day in kWh.
    Using the average of the *available* readings means a few missing minutes
    do not lower the daily total.
    """
    df = df.set_index("datetime")
    daily_avg_kw = df["Global_active_power"].resample("D").mean()

    # Some days have no readings at all -> fill them using the neighbouring days
    empty_days = daily_avg_kw.isna().sum()
    print(f"Days with no readings (filled by interpolation): {empty_days}")
    daily_avg_kw = daily_avg_kw.interpolate(method="linear")

    daily = pd.DataFrame(
        {
            "datetime": daily_avg_kw.index,
            "electricity_demand": (daily_avg_kw.values * 24).round(3),
        }
    )
    return daily


def main():
    raw = load_raw_data(RAW_FILE)
    clean = clean_data(raw)
    daily = resample_to_daily(clean)

    daily.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved {len(daily)} daily records to: {OUTPUT_FILE}")
    print(f"Date range: {daily['datetime'].min().date()} to {daily['datetime'].max().date()}")
    print(daily.head())


if __name__ == "__main__":
    main()
