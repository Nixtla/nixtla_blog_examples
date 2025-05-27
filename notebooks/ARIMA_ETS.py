# /// script
# [tool.marimo.runtime]
# auto_instantiate = false
# ///

import marimo

__generated_with = "0.13.10"
app = marimo.App(width="medium")


@app.cell
def _():
    # Import necessary libraries
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from statsmodels.tsa.arima_process import ArmaProcess
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA
    from statsforecast.arima import ARIMASummary
    from copy import deepcopy
    import statsmodels.api as sm
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsforecast.models import ARIMA
    from statsmodels.tsa.stattools import adfuller
    from utilsforecast.plotting import plot_series
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsforecast.models import AutoETS
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    import warnings
    warnings.filterwarnings("ignore")

    return (
        ARIMA,
        ARIMASummary,
        ArmaProcess,
        AutoARIMA,
        AutoETS,
        ExponentialSmoothing,
        StatsForecast,
        acorr_ljungbox,
        adfuller,
        np,
        pd,
        plot_acf,
        plot_pacf,
        plot_series,
        plt,
        seasonal_decompose,
        sm,
    )


@app.cell
def _(ArmaProcess, np, pd):
    # ======================
    # Generate Enhanced Synthetic Time Series (Daily Quantity sold in a retail Company)
    # ======================

    # Set random seed for reproducibility
    np.random.seed(0)

    # Define ARIMA parameters
    ar = np.array([1, -0.5])  # AR(1) coefficient
    ma = np.array([1, 0.4])   # MA(1) coefficient
    d = 1                     # Differencing order

    # Create ARMA process
    arma_process = ArmaProcess(ar, ma)

    # Number of samples
    n_samples = 365  # 1 year of daily data

    # Generate ARMA(1,1) data
    arma_sample = arma_process.generate_sample(nsample=n_samples, scale=1)

    # Integrate to get ARIMA(1,1,1)
    arima_sample = np.cumsum(arma_sample)  # Cumulative sum to introduce differencing (d=1)

    # Generate Trend Component (e.g., linear trend)
    trend = np.linspace(10, 20, n_samples)  # Increasing from 10 to 20 over the year

    # Generate Seasonality Component (e.g., weekly seasonality)
    seasonality = 10 + 5 * np.sin(2 * np.pi * np.arange(n_samples) / 7)  # Weekly pattern

    # Combine components to form the final synthetic time series
    synthetic_series = trend + seasonality + arima_sample

    # Generate Date Range
    dates = pd.date_range(start='2024-01-01', periods=n_samples, freq='D')

    # Create Pandas Series
    time_series = pd.Series(synthetic_series, index=dates, name='Daily_Quantity')

    return (time_series,)


@app.cell
def _(pd, time_series):
    # transform dataframe for statsforecast
    data = time_series.copy()
    data= data.reset_index()
    df=pd.DataFrame(data)
    df = df.rename(columns={'index': 'ds', 'Daily_Quantity': 'y'})
    df["unique_id"] = "1"
    df.columns=["ds", "y", "unique_id"]

    return (df,)


@app.cell
def _(df, plot_series):
    # use plot function from utilforecast
    plot_series(df)
    return


@app.cell
def _():
    #---------------------------
    # ARIMA
    #---------------------------
    return


@app.cell
def _(adfuller, plot_acf, plot_pacf, plt, time_series):
    # ======================
    # Box-Jenkins Methodology
    # ======================

    # Plot ACF and PACF for correlation
    plt.figure(figsize=(14,5))

    plt.subplot(1,2,1)
    plot_acf(time_series.diff().dropna(), lags=28, ax=plt.gca())
    plt.title('ACF (Autocorrelation Plot)')

    plt.subplot(1,2,2)
    plot_pacf(time_series.diff().dropna(), lags=28, ax=plt.gca(), method='ywm')
    plt.title('PACF (Partial Autocorrelation Plot)')

    plt.tight_layout()
    plt.show()

    # Augmented Dickey-Fuller Test
    # Test on original series
    adf_result_original = adfuller(time_series)
    print('Augmented Dickey-Fuller Test on Original Series:')
    print(f'ADF Statistic: {adf_result_original[0]:.4f}')
    print(f'p-value: {adf_result_original[1]:.4f}')
    for key, value in adf_result_original[4].items():
        print(f'Critical Value {key}: {value:.4f}')
    if adf_result_original[1] < 0.05:
        print("Conclusion: The original time series is stationary.")
    else:
        print("Conclusion: The original time series is non-stationary.")

    print('\n' + '-'*50 + '\n')

    # Test on differenced series
    adf_result_diff = adfuller(time_series.diff().dropna())
    print('Augmented Dickey-Fuller Test on Differenced Series:')
    print(f'ADF Statistic: {adf_result_diff[0]:.4f}')
    print(f'p-value: {adf_result_diff[1]:.4f}')
    for key, value in adf_result_diff[4].items():
        print(f'Critical Value {key}: {value:.4f}')
    if adf_result_diff[1] < 0.05:
        print("Conclusion: The differenced series is stationary.")
    else:
        print("Conclusion: The differenced series is non-stationary.")

    return


@app.cell
def _(ARIMASummary, AutoARIMA, StatsForecast, df):
    #--------------------------
    # Fit ARIMA
    #--------------------------

    def _():
        # Find the suitable model with AutoARIMA
        models = [AutoARIMA(allowmean=True)]

        # Initialize StatsForecast
        sf=StatsForecast(models=models,freq='D', n_jobs=-1)

        # Fit the model
        sf.fit(df=df[["ds", "y", "unique_id"]])

        # Retrieve the fitted ARIMA model
        fitted_arima_model = sf.fitted_[0, 0].model_
        # Print ARIMA Model Summary
        print("\nARIMA Model Summary:")
        return print(ARIMASummary(fitted_arima_model))


    _()
    return


@app.cell
def _(ARIMA, ARIMASummary, StatsForecast, df, pd):
    def _():
        # Compare the AutoARIMA model with other ARIMA models
        models = [
            ARIMA(order=(1, 1, 1), alias="arima111"),
            ARIMA(order=(2, 1, 2), alias="arima212"),
            ARIMA(order=(2, 1, 3), alias="arima213"),
            ARIMA(order=(2, 1, 4), alias="arima214"),
            ARIMA(order=(3, 1, 4), alias="arima314"),
        ]

        sf = StatsForecast(models=models, freq="D", n_jobs=-1)

        sf.fit(df=df[["ds", "y", "unique_id"]])

        summaries = []
        for model in sf.fitted_[0]:
            summary_model = {
                "model": model,
                "Orders": ARIMASummary(model.model_),
                "aic": model.model_["aic"],
                "aicc": model.model_["aicc"],
                "bic": model.model_["bic"],
            }
            summaries.append(summary_model)
        return pd.DataFrame(sorted(summaries, key=lambda d: d["aicc"]))


    _()
    return


@app.cell
def _(acorr_ljungbox, df, plot_acf, plot_pacf, plt, sm):
    def _():
        # ======================
        # Residuals Diagnostics Using statsmodels
        # ======================

        # Fit the best ARIMA model using statsmodels for diagnostics
        model_sm = sm.tsa.ARIMA(df.set_index('ds')['y'], order=(2, 1, 4))
        fitted_model_sm = model_sm.fit()

        # Print statsmodels ARIMA summary
        print("\nStatsmodels ARIMA Model Summary:")
        print(fitted_model_sm.summary())

        # Extract residuals
        residuals = fitted_model_sm.resid

        # Plot residuals and diagnostics
        plt.figure(figsize=(14,8))

        # Plot Residuals
        plt.subplot(2,2,1)
        plt.plot(residuals)
        plt.title('Residuals')
        plt.xlabel('Date')
        plt.ylabel('Residuals')

        # Plot ACF of Residuals
        plt.subplot(2,2,2)
        plot_acf(residuals, lags=21, ax=plt.gca())
        plt.title('ACF of Residuals')

        # Plot PACF of Residuals
        plt.subplot(2,2,3)
        plot_pacf(residuals, lags=28, ax=plt.gca(), method='ywm')
        plt.title('PACF of Residuals')

        # QQ-Plot
        plt.subplot(2,2,4)
        sm.qqplot(residuals, line='s', ax=plt.gca())
        plt.title('QQ-Plot of Residuals')

        plt.tight_layout()
        plt.show()

        # Perform Ljung-Box test for autocorrelation
        lb_test = acorr_ljungbox(residuals, lags=[28], return_df=True)
        print("\nLjung-Box Test Results:")
        print(lb_test)

        # Interpretation
        if lb_test['lb_pvalue'].iloc[-1] > 0.05:
            print("\nConclusion: Residuals are independently distributed (fail to reject H₀). Good fit.")
        else:
            return print("\nConclusion: Residuals are not independently distributed (reject H₀). Consider revising the model.")


    _()
    return


@app.cell
def _(df, plot_series, sf):
    # Make 1 month prediction
    forecast_ARIMA = sf.predict(h=30)
    plot_series(df=df[["ds", "y", "unique_id"]], forecasts_df=forecast_ARIMA)


    return


@app.cell
def _():
    #---------------------------
    # ETS
    #--------------------------
    return


@app.cell
def _(df, seasonal_decompose):
    #-------------------------------
    # ETS decomposition
    #-------------------------------

    # Plot decomposition
    a = seasonal_decompose(df["y"], model = "add", period=7)
    a.plot()

    return


@app.cell
def _(AutoETS, StatsForecast, df):
    def _():
        # Find suitable model with AutoETS
        models = [AutoETS()]

        # Initialize StatsForecast
        sf = StatsForecast(
            models=models,
            freq='D',    # Daily frequency
            n_jobs=-1
        )

        # Fit the model
        sf.fit(df=df[["ds", "y", "unique_id"]])

        # Retrieve the fitted ETS model
        fitted_ets_model = sf.fitted_[0, 0].model_["method"]
        # Print ETS Model Summary
        print("\nETS Model Summary:")
        return print(fitted_ets_model)


    _()
    return


@app.cell
def _(AutoETS, StatsForecast, df, pd):
    # Compare AutoETS choice with other models
    models=[AutoETS(model="ANN", alias="SES"),
            AutoETS(model="AAN", alias="Holt"),
    ]

    sf = StatsForecast(models=models, freq="D", n_jobs=-1)

    sf.fit(df=df[["ds", "y", "unique_id"]])

    summaries = []
    for model in sf.fitted_[0]:
        summary_model = {
            "model": model,
            "Orders": model.model_["method"],
            "aic": model.model_["aic"],
            "aicc": model.model_["aicc"],
            "bic": model.model_["bic"],
        }
        summaries.append(summary_model)

    pd.DataFrame(sorted(summaries, key=lambda d: d["aicc"]))
    return (sf,)


@app.cell
def _(ExponentialSmoothing, acorr_ljungbox, df, plot_acf, plot_pacf, plt, sm):
    # ======================
    # Residuals Diagnostics Using statsmodels
    # ======================

    # Fit the same ETS model using statsmodels for diagnostics
    model_ets = ExponentialSmoothing(
        df.set_index('ds')['y'],
        trend=None,
        seasonal=None,
        seasonal_periods=7
    )
    fitted_model_ets = model_ets.fit()

    # Print ETS model summary
    print("\nETS Model Summary:")
    print(fitted_model_ets.summary())

    # Extract residuals
    residuals = fitted_model_ets.resid

    # Plot residuals and diagnostics
    plt.figure(figsize=(14,8))

    # Plot Residuals
    plt.subplot(2,2,1)
    plt.plot(residuals)
    plt.title('Residuals')
    plt.xlabel('Date')
    plt.ylabel('Residuals')

    # Plot ACF of Residuals
    plt.subplot(2,2,2)
    plot_acf(residuals, lags=28, ax=plt.gca())
    plt.title('ACF of Residuals')

    # Plot PACF of Residuals
    plt.subplot(2,2,3)
    plot_pacf(residuals, lags=28, ax=plt.gca(), method='ywm')
    plt.title('PACF of Residuals')

    # QQ-Plot
    plt.subplot(2,2,4)
    sm.qqplot(residuals, line='s', ax=plt.gca())
    plt.title('QQ-Plot of Residuals')

    plt.tight_layout()
    plt.show()

    # Perform Ljung-Box test for autocorrelation
    lb_test = acorr_ljungbox(residuals, lags=[28], return_df=True)
    print("\nLjung-Box Test Results:")
    print(lb_test)

    # Interpretation
    if lb_test['lb_pvalue'].iloc[-1] > 0.05:
        print("\nConclusion: Residuals are independently distributed (fail to reject H₀). Good fit.")
    else:
        print("\nConclusion: Residuals are not independently distributed (reject H₀). Consider revising the model.")
    model_sm = sm.tsa.ARIMA(df.set_index('ds')['y'], order=(2, 1, 4))
    fitted_model_sm = model_sm.fit()


    return


@app.cell
def _(df, plot_series, sf):
    # make 1 month prediction
    forecast_ETS = sf.predict(h=30)
    plot_series(df=df[["ds", "y", "unique_id"]], forecasts_df=forecast_ETS)
    return


if __name__ == "__main__":
    app.run()
