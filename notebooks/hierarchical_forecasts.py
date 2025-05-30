import marimo

__generated_with = "0.13.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
    # Top-Down, Bottom-Up: Making Sense of Hierarchical Forecasts

    Load the required libraries first.
    """
    )
    return


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import os
    import sys
    import pathlib 

    # For base forecasting
    from statsforecast import StatsForecast
    from statsforecast.models import AutoETS, Naive

    # For hierarchical reconciliation
    from hierarchicalforecast.core import HierarchicalReconciliation
    from hierarchicalforecast.methods import BottomUp, TopDown, MiddleOut, MinTrace
    from hierarchicalforecast.utils import aggregate, HierarchicalPlot
    from hierarchicalforecast.evaluation import evaluate

    # For evaluation
    from hierarchicalforecast.evaluation import HierarchicalEvaluation
    from utilsforecast.losses import mae, rmse

    os.environ["NIXTLA_ID_AS_COL"] = "1"
    return (
        AutoETS,
        BottomUp,
        HierarchicalPlot,
        HierarchicalReconciliation,
        MiddleOut,
        MinTrace,
        StatsForecast,
        TopDown,
        aggregate,
        evaluate,
        pathlib,
        pd,
        rmse,
    )


@app.cell
def _():
    # Define the levels of hierarchy.
    hierarchy_levels = [['Country'],
                        ['Country', 'Region'], 
                        ['Country', 'Region', 'Citynoncity'], 
                        ]
    return (hierarchy_levels,)


@app.cell
def _(aggregate, hierarchy_levels, pathlib, pd):
    # Get required data and pre-process.

    libpath_computed = str(pathlib.Path.cwd().parent.absolute())

    df = pd.read_csv(f'{libpath_computed}/data/HierarchicalForecasting_TourismData_2Region.csv')
    df['ds'] = pd.to_datetime(df['ds'])

    Y_df, S_df, tags = aggregate(df=df, spec=hierarchy_levels)

    # print(type(tags))
    # tags.update({'middle_level': 'Country/Region'})
    # print(type(tags))

    Y_df['ds'] = pd.to_datetime(Y_df['ds'])
    Y_df['unique_id'] = Y_df['unique_id'].astype(str)
    Y_df = Y_df.sort_values(by = ['unique_id', 'ds'], axis = 0)

    # print("Y_df shape:", Y_df.shape)
    # print("S_df shape:", S_df.shape)
    # print('Y\n', Y_df.head())
    # print('S\n', S_df.head())
    # print(tags)
    return S_df, Y_df, tags


@app.cell
def _():
    # Define hierarchical levels of the data
    # tags = {
    #     "Country": ['total'], 
    #     "Country/State": ['nsw', 'nt'],
    #     "Country/State/CityNonCity": ['nsw-city', 'nsw-noncity', 'nt-city', 'nt-noncity']
    # }
    return


@app.cell
def _(Y_df):
    # Split data into training and testing sets
    # The forecast horizon 'h' depends on the data frequency and desired forecast length.
    # TourismSmall is quarterly, so h=4 means forecasting 1 year ahead.
    h = 4
    Y_test_df = Y_df.groupby('unique_id').tail(h)
    Y_train_df = Y_df.drop(Y_test_df.index)
    return Y_test_df, Y_train_df, h


@app.cell
def _(AutoETS, StatsForecast, Y_train_df):
    # Initialize StatsForecast object
    models = [
        AutoETS(season_length=4, model = 'ZZZ')
    ]

    sf = StatsForecast(
        models=models,
        freq='Q',
        n_jobs=-1  
    )

    sf.fit(Y_train_df)
    return (sf,)


@app.cell
def _(h, sf):
    # Generate base forecasts for the horizon 'h'
    Y_hat_df = sf.predict(h=h)

    # Optionally, retrieve fitted values if needed by some reconciliation methods (e.g., for WLS-var)
    # Y_fitted_df = sf.predict_in_sample()
    return (Y_hat_df,)


@app.cell
def _(
    BottomUp,
    HierarchicalReconciliation,
    MiddleOut,
    MinTrace,
    S_df,
    TopDown,
    Y_hat_df,
    Y_train_df,
    tags,
):
    reconcilers = [
        BottomUp(),
        TopDown(method='proportion_averages'),
        MiddleOut(middle_level='Country/Region',
                  top_down_method='proportion_averages'),
        MinTrace(method='ols')
    ]
    hrec = HierarchicalReconciliation(reconcilers=reconcilers)
    Y_rec_df = hrec.reconcile(Y_hat_df=Y_hat_df, 
                              Y_df=Y_train_df,
                              S=S_df, 
                              tags=tags)
    return (Y_rec_df,)


@app.cell
def _(Y_rec_df, Y_test_df):
    Y_rec_df_with_y = Y_rec_df.merge(Y_test_df, on=['unique_id', 'ds'], how='left')
    Y_rec_df_with_y = Y_rec_df_with_y.rename(columns = {
        'AutoETS' : 'Base Forecasts', 
        'AutoETS/BottomUp' : 'Bottom-Up',
        'AutoETS/TopDown_method-proportion_averages' : 'Top-Down',
        'AutoETS/MiddleOut_middle_level-Country/Region_top_down_method-proportion_averages': 'Middle-Out',
        'AutoETS/MinTrace_method-ols': 'Min Trace'
    })
    Y_rec_df_with_y
    return (Y_rec_df_with_y,)


@app.cell
def _(HierarchicalPlot, S_df, tags):
    hplot = HierarchicalPlot(S=S_df, tags=tags)
    return (hplot,)


@app.cell
def _(Y_rec_df_with_y, hplot):
    # fig = hplot.plot_series(
    #     series = 'A',
    #     Y_df=Y_rec_df_with_y,
    #     models=['Base Forecasts', 'Bottom-Up', 'Top-Down', 'Middle-Out', 'Min Trace', 'y'],
    # )
    fig = hplot.plot_hierarchically_linked_series(
        bottom_series = 'A/nsw/city',
        Y_df=Y_rec_df_with_y,
        models=['Base Forecasts', 'Bottom-Up', 'Top-Down', 'Middle-Out', 'Min Trace', 'y'],
    )
    # fig.savefig("hierarchical_forecast_plot.svg", format="svg", bbox_inches="tight")
    return (fig,)


@app.cell
def _(fig):
    print(fig) # why are we getting none here?
    return


@app.cell
def _(Y_rec_df_with_y, Y_train_df, evaluate, rmse, tags):
    eval_tags = {}
    eval_tags['Country'] = tags['Country']
    eval_tags['Region'] = tags['Country/Region']
    eval_tags['City Non-City'] = tags['Country/Region/Citynoncity']

    evaluation = evaluate(df = Y_rec_df_with_y, 
                          metrics = [rmse], 
                          tags = eval_tags,
                          train_df = Y_train_df
                         )

    numeric_cols = evaluation.select_dtypes(include="number").columns
    evaluation[numeric_cols] = evaluation[numeric_cols].map('{:.2f}'.format)
    evaluation
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
