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
    import pyarrow as pa
    import pyarrow.parquet as pq

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
        HierarchicalReconciliation,
        MiddleOut,
        MinTrace,
        StatsForecast,
        TopDown,
        aggregate,
        evaluate,
        np,
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


    df = pd.read_parquet(f'{libpath_computed}/data/HierarchicalForecasting_TourismData_2Region.parquet', engine='pyarrow')
    df['ds'] = pd.to_datetime(df['ds'])

    Y_df, S_df, tags = aggregate(df=df, spec=hierarchy_levels)

    Y_df['ds'] = pd.to_datetime(Y_df['ds'])
    Y_df['unique_id'] = Y_df['unique_id'].astype(str)
    Y_df = Y_df.sort_values(by = ['unique_id', 'ds'], axis = 0)
    return S_df, Y_df, libpath_computed, tags


@app.cell
def _(Y_df):
    Y_df.head()
    return


@app.cell
def _(S_df):
    S_df
    return


@app.cell
def _(tags):
    tags
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
def _(AutoETS, StatsForecast, Y_train_df, h):
    # Initialize StatsForecast object
    models = [
        AutoETS(season_length=4, model = 'ZZZ')
    ]

    sf = StatsForecast(
        models=models,
        freq='Q',
        n_jobs=-1  
    )

    Y_hat_df = sf.forecast(h=h, df = Y_train_df, fitted = True)
    Y_fitted_df = sf.forecast_fitted_values()
    return Y_fitted_df, Y_hat_df, sf


@app.cell
def _(Y_fitted_df, Y_hat_df, libpath_computed, sf):
    fig = sf.plot(df = Y_fitted_df, forecasts_df=Y_hat_df)

    fig.savefig(f'{libpath_computed}/images/baselineForecasts_baseForecasts.svg', format="svg") 
    sf.plot(df = Y_fitted_df, forecasts_df=Y_hat_df)
    return


@app.cell
def _(
    BottomUp,
    HierarchicalReconciliation,
    MiddleOut,
    MinTrace,
    S_df,
    TopDown,
    Y_fitted_df,
    Y_hat_df,
    tags,
):
    reconcilers = [
        BottomUp(),
        TopDown(method='proportion_averages'),
        MiddleOut(middle_level='Country/Region',
                    top_down_method='proportion_averages'),
        MinTrace(method='ols'),
        MinTrace(method='wls_struct'),
        MinTrace(method='wls_var'),
        MinTrace(method='mint_shrink'),
        MinTrace(method='mint_cov')
    ]
    hrec = HierarchicalReconciliation(reconcilers=reconcilers)
    Y_rec_df = hrec.reconcile(Y_hat_df=Y_hat_df, 
                              Y_df=Y_fitted_df,
                              S=S_df, 
                              tags=tags)
    return (Y_rec_df,)


@app.cell
def _(Y_rec_df, Y_test_df):
    Y_rec_df_with_y = Y_rec_df.merge(Y_test_df, on=['unique_id', 'ds'], how='left')
    Y_rec_df_with_y = Y_rec_df_with_y.rename(columns = {
        'AutoETS' : 'Base', 
        'AutoETS/BottomUp' : 'Bottom-Up',
        'AutoETS/TopDown_method-proportion_averages' : 'Top-Down',
        'AutoETS/MiddleOut_middle_level-Country/Region_top_down_method-proportion_averages': 'Middle-Out',
        'AutoETS/MinTrace_method-ols': 'MT-ols',
        'AutoETS/MinTrace_method-wls_struct': 'MT-struct',
        'AutoETS/MinTrace_method-wls_var': 'MT-var',
        'AutoETS/MinTrace_method-mint_shrink': 'MT-shrink',
        'AutoETS/MinTrace_method-mint_cov': 'MT-cov'
    })
    Y_rec_df_with_y
    return (Y_rec_df_with_y,)


@app.cell
def _(Y_rec_df_with_y, np):
    import matplotlib.pyplot as plt

    uid = 'A'
    df_subset = Y_rec_df_with_y[Y_rec_df_with_y['unique_id'] == uid]

    value_columns = [col for col in Y_rec_df_with_y.columns if col not in ['unique_id', 'ds']]

    # Color map
    color_map = {
        'Base Forecasts': 'khaki',
        'Bottom-Up': 'lightblue',
        'Top-Down': 'cornflowerblue',
        'Middle-Out': 'royalblue',
        'MinTrace-ols': 'lightgreen',
        'MinTrace-struct': 'mediumseagreen',
        'MinTrace-var': 'seagreen',
        'MinTrace-shrink': 'forestgreen',
        'MinTrace-cov': 'darkgreen',
        'y': 'tan'
    }

    # Plotting
    x = np.arange(len(df_subset['ds']))  
    width = 0.8 / len(value_columns)    
    figh, ax = plt.subplots(figsize=(14, 6))

    for i, col in enumerate(value_columns):
        ax.bar(x + i*width, df_subset[col], width=width, label=col, color=color_map.get(col, 'gray'))

    # X-axis labels
    ax.set_xticks(x + width*(len(value_columns)-1)/2)
    ax.set_xticklabels(df_subset['ds'], rotation=45)

    # Titles and labels
    ax.set_title(f"Hierarchical Forecasts for {uid}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Units")
    ax.legend(loc='best', fontsize='small', ncol=2)

    plt.tight_layout()
    plt.savefig('baselineForecasts_hierarchicalForecasts.svg', format="svg")
    plt.show()

    return


@app.cell
def _():
    # import plotly.express as px
    # import plotly.graph_objects as go
    # import plotly.io as pio

    # value_columns = [col for col in Y_rec_df_with_y.columns if col not in ['unique_id', 'ds']]

    # # Define groups for coloring
    # bottom_up_group = ['Bottom-Up', 'Top-Down', 'Middle-Out']
    # mintrace_group = [col for col in df.columns if col.startswith('MinTrace')]
    # others = ['Base Forecasts', 'y']

    # # Assign colors to groups
    # color_map = {
    #     'Base Forecasts': 'yellow',
    #     'Bottom-Up': 'lightblue',
    #     'Top-Down': 'cornflowerblue',
    #     'Middle-Out': 'royalblue',
    #     'MinTrace-ols': 'lightgreen',
    #     'MinTrace-struct': 'mediumseagreen',
    #     'MinTrace-var': 'seagreen',
    #     'MinTrace-shrink': 'forestgreen',
    #     'MinTrace-cov': 'darkgreen',
    #     'y': 'orange'
    # }

    # Y_rec_df_with_y['unique_id'].unique()

    # uid = 'A'
    # df_subset = Y_rec_df_with_y[Y_rec_df_with_y['unique_id'] == uid]

    # fig_h = go.Figure()

    # for col in value_columns:
    #     fig_h.add_trace(go.Bar(
    #         x=df_subset['ds'],
    #         y=df_subset[col],
    #         name=col,
    #         marker_color=color_map.get(col, 'gray')
    #     ))

    # fig_h.update_layout(
    #     title=f"Hierarchical Forecasts for {uid}",
    #     xaxis_title="Date",
    #     yaxis_title="Units",
    #     barmode='group'
    # )

    # # pio.write_image(fig_h, f'{libpath_computed}/images/baselineForecasts_hierarchicalForecasts.svg', format="svg")

    # fig_h.show()
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
    evaluation[numeric_cols] = evaluation[numeric_cols].map('{:.1f}'.format)
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
