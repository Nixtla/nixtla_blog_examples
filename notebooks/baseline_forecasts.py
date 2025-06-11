# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "matplotlib==3.10.3",
#     "numpy==2.2.6",
#     "pandas==2.2.3",
#     "statsforecast==2.0.1",
#     "utilsforecast==0.2.12",
# ]
# ///

import marimo

__generated_with = "0.13.9"
app = marimo.App(width="medium", app_title="Baseline Forecasts")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Effortless Accuracy: Unlocking the Power of Baseline Forecasts""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Required Packages""")
    return


@app.cell
def _():
    import os
    import sys
    import pathlib 

    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    from statsforecast import StatsForecast
    from statsforecast.models import (
        HistoricAverage,
        Naive,
        SeasonalNaive,
        WindowAverage,
    )
    from utilsforecast.evaluation import evaluate
    from utilsforecast.losses import rmse

    os.environ["NIXTLA_ID_AS_COL"] = "1"
    return (
        HistoricAverage,
        Naive,
        SeasonalNaive,
        StatsForecast,
        WindowAverage,
        evaluate,
        pathlib,
        pd,
        rmse,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Getting the data
    Tourism data (from the R-tsibble package), but with only 3 regions included.
    """
    )
    return


@app.cell
def _(pathlib, pd):

    libpath_computed = str(pathlib.Path.cwd().parent.absolute())


    df = pd.read_parquet(f'{libpath_computed}/data/tourism.parquet', engine='pyarrow')

    # df = pd.read_csv("data/tourism.csv")
    df["ds"] = pd.PeriodIndex(df["ds"], freq="Q").to_timestamp()
    df
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Splitting data into test and train""")
    return


@app.cell
def _(df):
    test_df = df.groupby("unique_id", group_keys=False).tail(4)
    train_df = df[~df.index.isin(test_df.index)]
    return test_df, train_df


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Setting up the baseline models and training them on the train data.""")
    return


@app.cell
def _(
    HistoricAverage,
    Naive,
    SeasonalNaive,
    StatsForecast,
    WindowAverage,
    train_df,
):
    # Define forecast horizon
    h = 4

    models = [
        HistoricAverage(),
        Naive(),
        SeasonalNaive(season_length = 4), # Quarterly data seasonality = 4
        WindowAverage(window_size=4)
    ]

    sf = StatsForecast(
        models=models,
        freq="QS",
    )

    sf.fit(train_df)
    return h, sf


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Predicting for each of the model""")
    return


@app.cell
def _(h, sf):
    pred_df = sf.predict(h=h)
    return (pred_df,)


@app.cell
def _(df, pred_df, sf):
    sf.plot(df, pred_df)
    return


@app.cell(hide_code=True)
def _():
    ### Evaluate the models
    return


@app.cell
def _(evaluate, pd, pred_df, rmse, test_df):
    accuracy_df =  pd.merge(test_df, pred_df, how = "left", on = ["unique_id", "ds"])
    evaluate(accuracy_df, metrics=[rmse])
    return


if __name__ == "__main__":
    app.run()
