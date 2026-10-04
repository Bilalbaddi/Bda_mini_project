# ⚡ Electricity Demand Forecasting

Mini-project — Roll No. 64

## 1. Problem Statement

Electricity providers and households need to know how much electricity will be used in the coming days. Good forecasts help plan power generation, reduce waste and control electricity bills. The task is to predict future electricity demand from past consumption data.

## 2. Objective

- Clean and prepare real household electricity consumption data.
- Create time-series features (calendar features, lag values, rolling averages).
- Train a machine learning model to predict daily electricity demand.
- Evaluate the model using MAE, RMSE and R².
- Build an interactive dashboard that shows trends and forecasts the next 7, 14 or 30 days.

## 3. Dataset

**UCI Individual Household Electric Power Consumption**
https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption

- Minute-by-minute measurements from one house near Paris, France.
- Period: December 2006 to November 2010 (about 2 million rows).
- Missing values are marked with `?` (about 1.25% of rows).
- Target column used: `Global_active_power` (in kilowatts).

After preprocessing, the data is converted into **daily electricity demand (kWh)**, giving 1,442 daily records.

## 4. Technologies Used

| Tool | Purpose |
|------|---------|
| Python | Programming language |
| Pandas | Loading, cleaning and resampling data |
| NumPy | Numerical calculations |
| Scikit-learn | Random Forest model and evaluation metrics |
| Streamlit | Interactive web dashboard |
| Plotly | Interactive charts |
| Joblib | Saving and loading the trained model |

## 5. Methodology

1. **Data preprocessing** (`preprocessing.py`)
   - Combine `Date` and `Time` into one `datetime` column.
   - Convert `Global_active_power` to numbers (`?` becomes missing).
   - Sort the data chronologically.
   - Resample minute data to daily data: daily demand (kWh) = average power of the day (kW) × 24 hours.
   - Fill the 9 days with no readings using linear interpolation.
   - Save as `data/electricity_demand.csv` (`datetime`, `electricity_demand`).

2. **Feature engineering** (`train_model.py`)

   | Feature | Meaning |
   |---------|---------|
   | `day` | Day of the month (1–31) |
   | `day_of_week` | 0 = Monday … 6 = Sunday |
   | `month` | Month (1–12) |
   | `day_of_year` | 1–366, captures the season |
   | `lag_1` | Demand of the previous day |
   | `lag_7` | Demand 7 days ago (same weekday last week) |
   | `lag_14` | Demand 14 days ago |
   | `rolling_mean_7` | Average of the previous 7 days |
   | `rolling_mean_14` | Average of the previous 14 days |

   All lag and rolling features use **only past values** (`shift(1)` before rolling), so there is no data leakage.

3. **Chronological train/test split**: the first 80% of days are used for training and the last 20% for testing. The data is **not shuffled**, because the model must never learn from the future.

4. **Model training and evaluation** on the test period.

5. **Recursive forecasting** of future days in the dashboard.

## 6. Machine Learning Model

**Random Forest Regressor** (scikit-learn)

- An ensemble of 200 decision trees; the final prediction is the average of all trees.
- Handles non-linear patterns such as the seasons and weekday/weekend effects.
- Less prone to overfitting than a single decision tree.
- Settings: `n_estimators=200`, `max_depth=10`, `min_samples_leaf=3`, `random_state=42`.

### Recursive multi-day forecasting

The model predicts one day at a time. For each future day:

1. Generate the next date.
2. Create its calendar features.
3. Compute lag and rolling features from previous demand values.
4. Predict the demand.
5. Add the prediction to the history and repeat for the next day.

This allows forecasts of 7, 14 or 30 days that follow realistic patterns instead of one flat average.

## 7. Evaluation Metrics

| Metric | Simple meaning | Result |
|--------|----------------|--------|
| MAE (Mean Absolute Error) | Average prediction error | ≈ 4.13 kWh |
| RMSE (Root Mean Squared Error) | Like MAE, but penalizes larger errors more | ≈ 5.66 kWh |
| R² (Coefficient of Determination) | How much of the variation the model explains (1 = perfect) | ≈ 0.41 |

For comparison, a simple "tomorrow = today" guess has an MAE of about 4.85 kWh, so the model performs better. Daily usage of a single household is quite random (guests, holidays, appliance use), so a moderate R² is expected.

## 8. Project Structure

```text
electricity-demand-forecasting/
│
├── data/
│   ├── household_power_consumption.txt   # original UCI dataset
│   └── electricity_demand.csv            # cleaned daily data
│
├── models/
│   └── electricity_model.pkl             # trained Random Forest model
│
├── preprocessing.py   # raw data -> clean daily CSV
├── train_model.py     # features, training, evaluation, saving the model
├── app.py             # Streamlit dashboard
├── requirements.txt
└── README.md
```

## 9. How to Install

```bash
cd electricity-demand-forecasting
pip install -r requirements.txt
```

If `data/household_power_consumption.txt` is missing, download it from the UCI link above, unzip it, and place the `.txt` file in the `data/` folder.

## 10. How to Run

```bash
# Step 1: create the clean daily CSV (only needed once)
python preprocessing.py

# Step 2: train and save the model (only needed once)
python train_model.py

# Step 3: start the dashboard
streamlit run app.py
```

The dashboard opens in the browser at http://localhost:8501.

## 11. Expected Output

- **`preprocessing.py`** prints the number of rows, missing values and saves `data/electricity_demand.csv` (1,442 daily records).
- **`train_model.py`** prints the train/test periods, MAE, RMSE, R² and saves `models/electricity_model.pkl`.
- **The dashboard** shows:
  - 📊 **Dataset Overview**: total records, average, maximum and minimum demand, and the historical demand chart.
  - 📈 **Historical Demand**: daily, weekly and monthly trend charts.
  - 🤖 **Model Performance**: MAE, RMSE, R² with simple explanations and an Actual vs Predicted chart.
  - 🔮 **Future Electricity Demand Forecast**: choose 7/14/30 days in the sidebar, click **Generate Forecast**, and see a table of predicted values plus a chart of historical and forecasted demand.
