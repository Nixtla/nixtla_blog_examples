from pathlib import Path

import pandas as pd


def load_sample_data():
    """Load built-in sample dataset for demo"""
    sample_path = Path(__file__).parent.parent.parent / "sample_data" / "m5_sample.csv"
    return pd.read_csv(sample_path)


def load_data(uploaded_file):
    """Load and validate CSV file from user upload"""
    df = pd.read_csv(uploaded_file)

    required_columns = ["unique_id", "ds", "y", "sell_price"]
    if not all(col in df.columns for col in required_columns):
        missing_cols = [col for col in required_columns if col not in df.columns]
        raise ValueError(f"Missing required columns: {missing_cols}")

    return df


def prepare_data(df, max_products=5):
    """Transform M5 data to TimeGPT format"""
    df = df.copy()

    # Convert date column to datetime
    df["ds"] = pd.to_datetime(df["ds"])

    # Sort by unique_id and date
    df = df.sort_values(["unique_id", "ds"])

    # Limit to max_products to keep demo manageable
    unique_ids = df["unique_id"].unique()[:max_products]
    df = df[df["unique_id"].isin(unique_ids)]

    # Ensure data quality
    df = df.dropna(subset=["y", "sell_price"])
    df["y"] = df["y"].astype(int)
    df["sell_price"] = df["sell_price"].astype(float)

    return df


def create_scenario_data(df, price_change):
    """Apply price change to dataset"""
    scenario_df = df.copy()
    scenario_df["sell_price"] = scenario_df["sell_price"] * (1 + price_change)
    return scenario_df


def get_data_summary(df):
    """Generate data summary statistics"""
    summary = {
        "total_products": df["unique_id"].nunique(),
        "date_range": f"{df['ds'].min().strftime('%Y-%m-%d')} to {df['ds'].max().strftime('%Y-%m-%d')}",
        "total_sales": df["y"].sum(),
        "avg_price": df["sell_price"].mean(),
        "total_days": df["ds"].nunique(),
    }
    return summary


def validate_forecast_data(df):
    """Validate data format for TimeGPT forecasting"""
    if df.empty:
        return False, "Dataset is empty"

    required_cols = ["unique_id", "ds", "y"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        return False, f"Missing required columns for forecasting: {missing_cols}"

    if df["y"].isnull().any():
        return False, "Sales data (y column) contains null values"

    if df["ds"].isnull().any():
        return False, "Date data (ds column) contains null values"

    return True, "Data is valid for forecasting"
