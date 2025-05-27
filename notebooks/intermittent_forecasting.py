# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "matplotlib==3.10.1",
#     "nixtla==0.6.6",
#     "numpy==2.2.5",
#     "openai==1.76.2",
#     "pandas==2.2.3",
#     "plotly==6.0.1",
#     "utilsforecast==0.2.12",
# ]
# ///

import marimo

__generated_with = "0.13.6"
app = marimo.App(app_title="Intermittent Forecasting")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Import required libraries for data analysis, forecasting, and evaluation metrics.""")
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    from nixtla import NixtlaClient
    from utilsforecast.evaluation import evaluate
    from utilsforecast.losses import mae
    return NixtlaClient, evaluate, mae, np, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Prerequisites""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Initialize Nixtla client with API key from environment variables.""")
    return


@app.cell
def _(NixtlaClient):
    import os

    NIXTLA_API_KEY = os.environ["NIXTLA_API_KEY"]
    client = NixtlaClient(api_key=NIXTLA_API_KEY)
    return (client,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Data Preparation""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Load and preprocess the M5 sales dataset with exogenous variables.""")
    return


@app.cell
def _(pd):
    sales_data = pd.read_csv(
        "https://raw.githubusercontent.com/Nixtla/transfer-learning-time-series/main/datasets/m5_sales_exog_small.csv"
    )
    sales_data["ds"] = pd.to_datetime(sales_data["ds"])
    sales_data.head()
    return (sales_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Visualize the sales data for the first 365 days.""")
    return


@app.cell
def _(client, sales_data):
    sales_plot = client.plot(
        sales_data,
        max_insample_length=365,
    )

    sales_plot
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Bounded Forecasts""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Apply log transformation to handle the intermittent nature of the data.""")
    return


@app.cell
def _(np, sales_data):
    log_transformed_data = sales_data.copy()
    log_transformed_data["y"] = np.log(log_transformed_data["y"] + 1)
    log_transformed_data.head()
    return (log_transformed_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Compare original and log-transformed data for a specific product.""")
    return


@app.cell
def _(client, log_transformed_data, sales_data):
    import matplotlib.pyplot as plt

    # Create a figure and axis for Matplotlib
    _, ax = plt.subplots(figsize=(10, 5))

    # Plot the original data
    client.plot(
        sales_data,
        max_insample_length=30,
        unique_ids=["FOODS_1_001"],
        engine="matplotlib",
        ax=ax,
    )

    # Plot the transformed data on the same axes
    client.plot(
        log_transformed_data,
        max_insample_length=30,
        unique_ids=["FOODS_1_001"],
        engine="matplotlib",
        ax=ax,
    )

    # Manually change the color of the second line plot
    lines = ax.get_lines()
    if len(lines) > 1:
        lines[1].set_color("#006400")  # New color for transformed data
        lines[1].set_linestyle("--")

    # Add legend with custom labels
    handles, labels = ax.get_legend_handles_labels()
    labels = ["Original Sales", "Transformed Sales"]
    ax.legend(handles, labels)

    ax
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Split data into training and test sets (last 28 days for testing).""")
    return


@app.cell
def _(log_transformed_data):
    # Select the last 28 observations for each unique_id — used as test data
    test_data = log_transformed_data.groupby("unique_id").tail(28)

    # Drop the test set indices from the original dataset to form the training set
    train_data = log_transformed_data.drop(test_data.index).reset_index(drop=True)
    return test_data, train_data


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Generating Forecasts with TimeGPT""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Generate forecasts using the base TimeGPT model with 80% confidence interval.""")
    return


@app.cell
def _(client, train_data):
    log_forecast = client.forecast(
        df=train_data,
        h=28,
        level=[80],
        model="timegpt-1-long-horizon",
        time_col="ds",
        target_col="y",
        id_col="unique_id",
    )
    return (log_forecast,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Reverse Transformation

    After obtaining predictions, we reverse the log transformation to return to the original scale.
    """
    )
    return


@app.cell
def _(log_forecast, np):
    def reverse_log_transform(df):
        df = df.copy()
        value_cols = [col for col in df if col not in ["ds", "unique_id"]]
        df[value_cols] = np.exp(df[value_cols]) - 1
        return df

    base_forecast = reverse_log_transform(log_forecast)
    base_forecast.head()
    return base_forecast, reverse_log_transform


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Evaluation""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Define functions to merge forecasts with real data and calculate MAE.""")
    return


@app.cell
def _(evaluate, mae, pd):
    def merge_forecast(real_data, forecast):
        merged_results = pd.merge(
            real_data, forecast, "left", ["unique_id", "ds"]
        )
        return merged_results

    def get_mean_mae(real_data, forecast):
        merged_results = merge_forecast(real_data, forecast)
        model_evaluation = evaluate(
            merged_results,
            metrics=[mae],
            models=["TimeGPT"],
            target_col="y",
            id_col="unique_id",
        )
        return model_evaluation.groupby("metric")["TimeGPT"].mean()["mae"]
    return (get_mean_mae,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Calculate MAE for the base model forecasts.""")
    return


@app.cell
def _(base_forecast, get_mean_mae, test_data):
    base_mae = get_mean_mae(test_data, base_forecast)
    print(base_mae)
    return (base_mae,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Finetuning the Model""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Generate forecasts using a fine-tuned TimeGPT model with 10 finetuning steps.""")
    return


@app.cell
def _(client, train_data):
    log_finetuned_forecast = client.forecast(
        df=train_data,
        h=28,
        level=[80],
        finetune_steps=10,
        finetune_loss="mae",
        model="timegpt-1-long-horizon",
        time_col="ds",
        target_col="y",
        id_col="unique_id",
    )
    return (log_finetuned_forecast,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Calculate MAE for the fine-tuned model forecasts.""")
    return


@app.cell
def _(get_mean_mae, log_finetuned_forecast, reverse_log_transform, test_data):
    finetuned_forecast = reverse_log_transform(log_finetuned_forecast)
    finedtune_mae = get_mean_mae(test_data, finetuned_forecast)
    print(finedtune_mae)
    return (finedtune_mae,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Incorporating Exogenous Variables""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Prepare exogenous variables for forecasting by removing target and price columns.""")
    return


@app.cell
def _(test_data):
    non_exogenous_variables = ["y", "sell_price"]
    futr_exog_data = test_data.drop(non_exogenous_variables, axis=1)
    futr_exog_data.head()
    return (futr_exog_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Generate forecasts using TimeGPT with exogenous variables and fine-tuning.""")
    return


@app.cell
def _(client, futr_exog_data, train_data):
    log_exogenous_forecast = client.forecast(
        df=train_data,
        X_df=futr_exog_data,
        h=28,
        level=[80],
        finetune_steps=10,
        finetune_loss="mae",
        model="timegpt-1-long-horizon",
        time_col="ds",
        target_col="y",
        id_col="unique_id",
    )
    return (log_exogenous_forecast,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Calculate MAE for the model with exogenous variables.""")
    return


@app.cell
def _(get_mean_mae, log_exogenous_forecast, reverse_log_transform, test_data):
    exogenous_forecast = reverse_log_transform(log_exogenous_forecast)
    exogenous_mae = get_mean_mae(test_data, exogenous_forecast)
    print(exogenous_mae)
    return (exogenous_mae,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Comparing MAE""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Create a comparison table of MAE values for all three model variants.""")
    return


@app.cell
def _(base_mae, exogenous_mae, finedtune_mae, pd):
    # Define the mean absolute error (MAE) values for different TimeGPT variants
    mae_values = {
        "Model Variant": ["Base TimeGPT", "Fine-Tuned TimeGPT", "TimeGPT with Exogenous"],
        "MAE": [base_mae, finedtune_mae, exogenous_mae]
    }

    mae_table = pd.DataFrame(mae_values)
    mae_table

    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
