# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "matplotlib==3.10.1",
#     "nixtla==0.6.6",
#     "pandas==2.2.3",
#     "python-dotenv==1.1.0",
# ]
# ///

import marimo

__generated_with = "0.13.6"
app = marimo.App(width="medium", app_title="Anomaly Detection")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Anomaly detection""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Import packages""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Import required libraries for data manipulation and Nixtla client initialization.""")
    return


@app.cell
def _():
    import os

    import pandas as pd
    from nixtla import NixtlaClient
    return NixtlaClient, os, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Initialize Nixtla client with API key from environment variables.""")
    return


@app.cell
def _(NixtlaClient, os):
    NIXTLA_API_KEY = os.environ["NIXTLA_API_KEY"]
    nixtla_client = NixtlaClient(api_key=NIXTLA_API_KEY)
    return (nixtla_client,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Load dataset

    Now, let's load the dataset for this tutorial.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Load the Peyton Manning Wikipedia page views dataset and display first 10 rows.""")
    return


@app.cell
def _(pd):
    # Read the dataset
    wikipedia = pd.read_csv("https://datasets-nixtla.s3.amazonaws.com/peyton-manning.csv", parse_dates=["ds"])
    wikipedia.head(10)
    return (wikipedia,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Plot the time series data to visualize the patterns.""")
    return


@app.cell
def _(nixtla_client, wikipedia):
    wikipedia_plot = nixtla_client.plot(wikipedia)
    wikipedia_plot
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Anomaly detection""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Detect anomalies in the time series using TimeGPT model with default settings.""")
    return


@app.cell
def _(nixtla_client, wikipedia):
    anomalies_df = nixtla_client.detect_anomalies(
        wikipedia,
        freq="D",
        model="timegpt-1",
    )
    anomalies_df.head()
    return (anomalies_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Visualize the detected anomalies on the time series plot.""")
    return


@app.cell
def _(anomalies_df, nixtla_client, wikipedia):
    anomaly_plot = nixtla_client.plot(wikipedia, anomalies_df)
    anomaly_plot
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Anomaly detection with exogenous features""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Detect anomalies using TimeGPT with additional date-based features (month and year).""")
    return


@app.cell
def _(nixtla_client, wikipedia):
    anomalies_df_exogenous = nixtla_client.detect_anomalies(
        wikipedia,
        freq="D",
        date_features=["month", "year"],
        date_features_to_one_hot=True,
        model="timegpt-1",
    )
    return (anomalies_df_exogenous,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Plot the feature importance weights to understand which features contribute most to anomaly detection.""")
    return


@app.cell
def _(nixtla_client):
    feature_plot = nixtla_client.weights_x.plot.barh(
        x="features",
        y="weights"
    )

    feature_plot
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Compare the number of anomalies detected with and without exogenous features.""")
    return


@app.cell
def _(anomalies_df, anomalies_df_exogenous):
    # Without exogenous features
    print("Number of anomalies without exogenous features:", anomalies_df.anomaly.sum())

    # With exogenous features
    print("Number of anomalies with exogenous features:", anomalies_df_exogenous.anomaly.sum())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Visualize the anomalies detected using the model with exogenous features.""")
    return


@app.cell
def _(anomalies_df_exogenous, nixtla_client, wikipedia):
    anomalies_exogenous_plot = nixtla_client.plot(wikipedia, anomalies_df_exogenous)
    anomalies_exogenous_plot
    return (anomalies_exogenous_plot,)


@app.cell
def _(anomalies_exogenous_plot):
    anomalies_exogenous_plot.savefig("images/anomalies_exogenous_plot.svg", format="svg", bbox_inches="tight")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Modifying the confidence intervals""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Detect anomalies using a lower confidence interval (70%) to see how it affects the results.""")
    return


@app.cell
def _(nixtla_client, wikipedia):
    anomalies_df_70 = nixtla_client.detect_anomalies(wikipedia, freq="D", level=70)
    return (anomalies_df_70,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Compare the number of anomalies detected with different confidence intervals (99% vs 70%).""")
    return


@app.cell
def _(anomalies_df, anomalies_df_70):
    # Print and compare anomaly counts
    print("Number of anomalies with 99% confidence interval:", anomalies_df.anomaly.sum())
    print("Number of anomalies with 70% confidence interval:", anomalies_df_70.anomaly.sum())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Visualize the anomalies detected using the 70% confidence interval.""")
    return


@app.cell
def _(anomalies_df_70, nixtla_client, wikipedia):
    anomalies_70_plot = nixtla_client.plot(wikipedia, anomalies_df_70)
    anomalies_70_plot
    return (anomalies_70_plot,)


@app.cell
def _(anomalies_70_plot):
    anomalies_70_plot.savefig("images/anomalies_70_plot.svg", format="svg", bbox_inches="tight")
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
