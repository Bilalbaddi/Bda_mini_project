"""
app.py
------
Streamlit dashboard for Electricity Demand Forecasting.

Run:
    streamlit run app.py
"""

import os

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Reuse the same feature logic that was used during training
from train_model import (
    DATA_FILE,
    FEATURE_COLUMNS,
    MODEL_FILE,
    TARGET_COLUMN,
    create_features,
    evaluate,
    load_data,
    split_data,
)

st.set_page_config(page_title="Electricity Demand Forecasting", page_icon="⚡", layout="wide")


# ---------------------------------------------------------------------------
# Loading data and model (cached so they are loaded only once)
# ---------------------------------------------------------------------------
@st.cache_data
def get_data():
    return load_data(DATA_FILE)


@st.cache_resource
def get_model():
    return joblib.load(MODEL_FILE)


def recursive_forecast(model, history, days):
    """
    Forecast `days` future values one day at a time.

    For each future day:
      1. create the next date
      2. build its calendar features
      3. compute lag / rolling features from the history
      4. predict the demand
      5. add the prediction to the history so the next day can use it
    """
    history = history[["datetime", TARGET_COLUMN]].copy()
    forecasts = []

    for _ in range(days):
        next_date = history["datetime"].iloc[-1] + pd.Timedelta(days=1)

        # Add the new day with an unknown (NaN) demand value
        new_row = pd.DataFrame({"datetime": [next_date], TARGET_COLUMN: [float("nan")]})
        history = pd.concat([history, new_row], ignore_index=True)

        # Build features; only the last 30 rows are needed for lags/rolling means
        features = create_features(history.tail(30)).iloc[[-1]][FEATURE_COLUMNS]
        prediction = float(model.predict(features)[0])

        # Store the prediction so it becomes "history" for the next step
        history.loc[history.index[-1], TARGET_COLUMN] = prediction
        forecasts.append({"Date": next_date.date(), "Predicted Electricity Demand (kWh)": round(prediction, 2)})

    return pd.DataFrame(forecasts)


def line_chart(x, y, name, title, color="#1f77b4"):
    """Small helper to draw a simple Plotly line chart."""
    fig = go.Figure(go.Scatter(x=x, y=y, mode="lines", name=name, line=dict(color=color)))
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Electricity Demand (kWh)",
        height=380,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


# ---------------------------------------------------------------------------
# Check that the required files exist
# ---------------------------------------------------------------------------
if not os.path.exists(DATA_FILE):
    st.error("Processed data not found. Please run:  python preprocessing.py")
    st.stop()
if not os.path.exists(MODEL_FILE):
    st.error("Trained model not found. Please run:  python train_model.py")
    st.stop()

df = get_data()
model = get_model()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("⚡ Electricity Demand Forecasting")
st.write("Forecast future electricity demand using historical electricity consumption data.")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.header("📁 Dataset")
st.sidebar.write(f"**Number of records:** {len(df):,} days")
st.sidebar.write(
    f"**Date range:** {df['datetime'].min():%d %b %Y} to {df['datetime'].max():%d %b %Y}"
)
st.sidebar.write(f"**Average demand:** {df[TARGET_COLUMN].mean():.2f} kWh/day")
st.sidebar.caption("Source: UCI Individual Household Electric Power Consumption dataset")

st.sidebar.header("⚙️ Forecast Settings")
forecast_days = st.sidebar.selectbox("Forecast period", [7, 14, 30], format_func=lambda d: f"{d} days")

# ---------------------------------------------------------------------------
# 1. Dataset Overview
# ---------------------------------------------------------------------------
st.header("📊 Dataset Overview")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Records", f"{len(df):,} days")
col2.metric("Average Demand", f"{df[TARGET_COLUMN].mean():.2f} kWh")
col3.metric("Maximum Demand", f"{df[TARGET_COLUMN].max():.2f} kWh")
col4.metric("Minimum Demand", f"{df[TARGET_COLUMN].min():.2f} kWh")

st.plotly_chart(
    line_chart(df["datetime"], df[TARGET_COLUMN], "Demand", "Historical Electricity Demand"),
    use_container_width=True,
)
st.caption(
    "Each point is the total electricity used by the household in one day (kWh). "
    "Demand is higher in winter and lower in summer."
)

with st.expander("Show raw data"):
    st.dataframe(df, use_container_width=True)

# ---------------------------------------------------------------------------
# 2. Historical Demand Trends
# ---------------------------------------------------------------------------
st.header("📈 Historical Demand")

trend_daily, trend_weekly, trend_monthly = st.tabs(["Daily Trend", "Weekly Trend", "Monthly Trend"])
series = df.set_index("datetime")[TARGET_COLUMN]

with trend_daily:
    # Show the most recent 90 days so the daily pattern is easy to see
    recent = series.tail(90)
    st.plotly_chart(
        line_chart(recent.index, recent.values, "Daily", "Daily Demand (last 90 days)"),
        use_container_width=True,
    )
    st.caption("Day-to-day changes in demand. Weekends usually use more electricity.")

with trend_weekly:
    weekly = series.resample("W").mean()
    st.plotly_chart(
        line_chart(weekly.index, weekly.values, "Weekly", "Average Daily Demand per Week", "#2ca02c"),
        use_container_width=True,
    )
    st.caption("Weekly averages smooth out daily noise and show the overall pattern.")

with trend_monthly:
    monthly = series.resample("MS").mean()
    st.plotly_chart(
        line_chart(monthly.index, monthly.values, "Monthly", "Average Daily Demand per Month", "#d62728"),
        use_container_width=True,
    )
    st.caption("Monthly averages clearly show the seasonal cycle: high in winter, low in summer.")

# ---------------------------------------------------------------------------
# 3. Model Performance
# ---------------------------------------------------------------------------
st.header("🤖 Model Performance")

# Recreate the same features and chronological split used in train_model.py
featured = create_features(df).dropna().reset_index(drop=True)
train, test = split_data(featured)
test_predictions = model.predict(test[FEATURE_COLUMNS])
metrics = evaluate(test[TARGET_COLUMN], test_predictions)

st.write(
    f"The Random Forest model was trained on **{len(train)} days** "
    f"({train['datetime'].min():%b %Y} – {train['datetime'].max():%b %Y}) and tested on the "
    f"next **{len(test)} days** ({test['datetime'].min():%b %Y} – {test['datetime'].max():%b %Y}) "
    "that it had never seen."
)

m1, m2, m3 = st.columns(3)
m1.metric("MAE", f"{metrics['MAE']:.2f} kWh")
m1.caption("**Average prediction error.** On average the prediction is this far from the real value.")
m2.metric("RMSE", f"{metrics['RMSE']:.2f} kWh")
m2.caption("**Penalizes larger errors.** Big mistakes increase this value more than small ones.")
m3.metric("R²", f"{metrics['R2']:.3f}")
m3.caption("**How well the model explains the variation** (1.0 = perfect, 0 = no better than the average).")

fig_test = go.Figure()
fig_test.add_trace(go.Scatter(x=test["datetime"], y=test[TARGET_COLUMN], mode="lines", name="Actual Demand"))
fig_test.add_trace(
    go.Scatter(x=test["datetime"], y=test_predictions, mode="lines", name="Predicted Demand", line=dict(dash="dash"))
)
fig_test.update_layout(
    title="Actual vs Predicted Demand (Test Period)",
    xaxis_title="Date",
    yaxis_title="Electricity Demand (kWh)",
    height=400,
    margin=dict(l=10, r=10, t=50, b=10),
)
st.plotly_chart(fig_test, use_container_width=True)
st.caption(
    "The dashed line follows the overall ups and downs of real demand. Sudden one-day spikes are hard "
    "to predict because they depend on household activities the data does not record."
)

# ---------------------------------------------------------------------------
# 4. Future Forecast
# ---------------------------------------------------------------------------
st.header("🔮 Future Electricity Demand Forecast")
st.write(
    f"Click the button to predict electricity demand for the next **{forecast_days} days** "
    f"after {df['datetime'].max():%d %b %Y}. Change the period in the sidebar."
)

if st.button("🔮 Generate Forecast"):
    st.session_state["forecast"] = recursive_forecast(model, df, forecast_days)

forecast = st.session_state.get("forecast")
if forecast is not None:
    table_col, chart_col = st.columns([1, 2])

    with table_col:
        st.subheader("Forecast Table")
        st.dataframe(forecast, use_container_width=True, hide_index=True)

    with chart_col:
        history_tail = df.tail(90)
        forecast_dates = pd.to_datetime(forecast["Date"])
        forecast_values = forecast["Predicted Electricity Demand (kWh)"]

        fig_forecast = go.Figure()
        fig_forecast.add_trace(
            go.Scatter(x=history_tail["datetime"], y=history_tail[TARGET_COLUMN], mode="lines", name="Historical Demand")
        )
        # Start the forecast line at the last real value so the two lines connect
        fig_forecast.add_trace(
            go.Scatter(
                x=[history_tail["datetime"].iloc[-1], *forecast_dates],
                y=[history_tail[TARGET_COLUMN].iloc[-1], *forecast_values],
                mode="lines+markers",
                name="Forecasted Demand",
                line=dict(color="#ff7f0e", dash="dash"),
            )
        )
        fig_forecast.add_vline(x=history_tail["datetime"].iloc[-1], line_dash="dot", line_color="gray")
        fig_forecast.update_layout(
            title="Historical Demand + Forecasted Demand",
            xaxis_title="Date",
            yaxis_title="Electricity Demand (kWh)",
            height=420,
            margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(fig_forecast, use_container_width=True)

    st.caption(
        "Blue = real past demand, orange dashed = model forecast. Each forecasted day uses the previous "
        "days' values (including earlier predictions), so the forecast changes from day to day "
        "instead of being one flat average."
    )
    st.info(
        f"Average forecasted demand: **{forecast_values.mean():.2f} kWh/day** · "
        f"Total for {len(forecast)} days: **{forecast_values.sum():.2f} kWh**"
    )
