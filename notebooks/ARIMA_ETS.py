# /// script
# dependencies = [
#     "marimo",
#     "matplotlib==3.10.3",
#     "numpy==2.2.6",
#     "pandas==2.3.0",
#     "statsforecast==2.0.1",
#     "statsmodels==0.14.4",
#     "utilsforecast==0.2.12",
# ]
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
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA
    from statsforecast.arima import ARIMASummary
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
    warnings.simplefilter(action='ignore', category=UserWarning)
    warnings.simplefilter(action='ignore', category=FutureWarning)
    return (
        ARIMA,
        ARIMASummary,
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
def _(np, pd):
    # ==================================================================================
    # Generate Enhanced Synthetic Time Series (Daily Quantity sold in a retail Company)
    # ==================================================================================


    def generate_synthetic_retail_data(
        start_date='2024-01-01',
        n_days=365,
        base_quantity=50,
        trend_slope=0.1,
        weekly_amplitude=10,
        yearly_amplitude=25,
        noise_std_dev=5):

        # Generate date range
        dates = pd.date_range(start=start_date, periods=n_days, freq='D')
        time_index = np.arange(n_days)

        # 1. Base Quantity
        # This is the fundamental level of sales.
        base = np.full(n_days, base_quantity)

        # 2. Trend Component (Linear)
        # Simulates a gradual increase or decrease in sales over time.
        trend = trend_slope * time_index

        # 3. Seasonality Component (Weekly)
        # Simulates recurring patterns within a week (e.g., higher sales on weekends).
        # Using a sine wave for smooth weekly fluctuations.
        weekly_seasonality = weekly_amplitude * np.sin(2 * np.pi * time_index / 7)

        # 4. Seasonality Component (Yearly)
        # Simulates recurring patterns within a year (e.g., holiday sales, seasonal demand).
        # Using a sine wave for smooth yearly fluctuations.
        yearly_seasonality = yearly_amplitude * np.sin(2 * np.pi * time_index / 365)

        # 5. Random Noise Component
        # Adds unpredictable, random fluctuations to the data.
        # Using a normal distribution for noise.
        np.random.seed(42) # for reproducibility
        noise = np.random.normal(0, noise_std_dev, n_days)

        # Combine all components
        # Ensure quantities are non-negative
        synthetic_series_values = base + trend + weekly_seasonality + yearly_seasonality + noise
        synthetic_series_values[synthetic_series_values < 0] = 0 # Ensure no negative quantities

        # Create a Pandas Series
        time_series = pd.Series(synthetic_series_values, index=dates, name='Daily_Quantity_Sold')

        return time_series

    # --- Generate the synthetic data ---
    time_series = generate_synthetic_retail_data()

    return (time_series,)


@app.cell
def _(time_series):
    time_series
    return


@app.cell
def _(pd, time_series):
    # transform dataframe for statsforecast
    data = time_series.copy()
    data= data.reset_index()
    df=pd.DataFrame(data)
    df = df.rename(columns={'index': 'ds', 'Daily_Quantity_Sold': 'y'})
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

    # Find the suitable model with AutoARIMA
    models_ar = [AutoARIMA()]

    # Initialise StatsForecast
    sf_ar=StatsForecast(models=models_ar,freq='D', n_jobs=-1).fit(df=df[["ds", "y", "unique_id"]])

    # Retrieve the fitted ARIMA model
    fitted_arima_model = sf_ar.fitted_[0, 0].model_
    # Print ARIMA Model Summary
    print("\nARIMA Model Summary:")
    print(ARIMASummary(fitted_arima_model))

    return


@app.cell
def _(ARIMA, ARIMASummary, StatsForecast, df, pd):
    # Compare the AutoARIMA model with other ARIMA models
    models_ = [
        ARIMA(order=(1, 1, 1), alias="arima111"),
        ARIMA(order=(3, 1, 2), alias="arima312"),
        ARIMA(order=(2, 1, 3), alias="arima213"),
        ARIMA(order=(2, 1, 4), alias="arima214"),
        ARIMA(order=(3, 1, 4), alias="arima314"),
    ]

    sf_ar_ = StatsForecast(models=models_, freq="D", n_jobs=-1).fit(df=df[["ds", "y", "unique_id"]])

    summaries_arima = []
    for model_arima in sf_ar_.fitted_[0]:
        summary_model_arima = {
            "model": model_arima,
            "Orders": ARIMASummary(model_arima.model_),
            "aic": model_arima.model_["aic"],
            "aicc": model_arima.model_["aicc"],
            "bic": model_arima.model_["bic"],
        }
        summaries_arima.append(summary_model_arima)
    pd.DataFrame(sorted(summaries_arima, key=lambda d: d["aicc"]))
    return


@app.cell
def _(acorr_ljungbox, df, plot_acf, plot_pacf, plt, sm):

    # ======================
    # Residuals Diagnostics Using statsmodels
    # ======================

    # Fit the best ARIMA model using statsmodels for diagnostics
    fitted_model_sm_ar = sm.tsa.ARIMA(df.set_index('ds')['y'], order=(3, 1, 4)).fit()

    # Print statsmodels ARIMA summary
    print("\nStatsmodels ARIMA Model Summary:")
    print(fitted_model_sm_ar.summary())

    # Extract residuals
    residuals = fitted_model_sm_ar.resid

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
    plot_acf(residuals, lags=7, ax=plt.gca())
    plt.title('ACF of Residuals')

    # Plot PACF of Residuals
    plt.subplot(2,2,3)
    plot_pacf(residuals, lags=7, ax=plt.gca(), method='ywm')
    plt.title('PACF of Residuals')

    # QQ-Plot
    plt.subplot(2,2,4)
    sm.qqplot(residuals, line='s', ax=plt.gca())
    plt.title('QQ-Plot of Residuals')

    plt.tight_layout()
    plt.show()

    # Perform Ljung-Box test for autocorrelation
    lb_test = acorr_ljungbox(residuals, lags=[7], return_df=True)
    print("\nLjung-Box Test Results:")
    print(lb_test)

    # Interpretation
    if lb_test['lb_pvalue'].iloc[-1] > 0.05:
        print("\nConclusion: Residuals are independently distributed (fail to reject H₀). Good fit.")
    else:
        print("\nConclusion: Residuals are not independently distributed (reject H₀). Consider revising the model.")



    return


@app.cell
def _(ARIMA, StatsForecast, df, plot_series):
    # Make a 1-month prediction
    # fit the best model
    models_arima = [ARIMA(order=(3,1,4))]

    # Initialise StatsForecast
    sf_arima=StatsForecast(models=models_arima,freq='D', n_jobs=-1).fit(df=df[["ds", "y", "unique_id"]])

    forecast_ARIMA = sf_arima.predict(h=30)
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

    # Find a suitable model with AutoETS
    models_ets = [AutoETS()]

    # Initialise StatsForecast
    sf_ets = StatsForecast(models=models_ets,freq='D', n_jobs=-1).fit(df=df[["ds", "y", "unique_id"]])


    # Retrieve the fitted ETS model
    fitted_ets_model = sf_ets.fitted_[0, 0].model_["method"]
    # Print ETS Model Summary
    print("\nETS Model Summary:")
    print(fitted_ets_model)

    return (sf_ets,)


@app.cell
def _(AutoETS, StatsForecast, df, pd):
    # Compare AutoETS choice with other models
    models_ets_=[AutoETS(model="ANN", alias="SES"),
            AutoETS(model="AAN", alias="Holt"),
    ]

    sf_ets_ = StatsForecast(models=models_ets_, freq="D", n_jobs=-1).fit(df=df[["ds", "y", "unique_id"]])

    summaries_ets = []
    for model_ets in sf_ets_.fitted_[0]:
        summary_model_ets = {
            "model": model_ets,
            "Orders": model_ets.model_["method"],
            "aic": model_ets.model_["aic"],
            "aicc": model_ets.model_["aicc"],
            "bic": model_ets.model_["bic"],
        }
        summaries_ets.append(summary_model_ets)

    pd.DataFrame(sorted(summaries_ets, key=lambda d: d["aicc"]))
    return


@app.cell
def _(ExponentialSmoothing, acorr_ljungbox, df, plot_acf, plot_pacf, plt, sm):
    # ======================
    # Residuals Diagnostics Using statsmodels
    # ======================

    # Fit the same ETS model using statsmodels for diagnostics
    fitted_model_ets = ExponentialSmoothing(
        df.set_index('ds')['y'],
        trend=None,
        seasonal=None,
        seasonal_periods=7).fit()

    # Print ETS model summary
    print("\nETS Model Summary:")
    print(fitted_model_ets.summary())

    # Extract residuals
    residuals_ets = fitted_model_ets.resid

    # Plot residuals and diagnostics
    plt.figure(figsize=(14,8))

    # Plot Residuals
    plt.subplot(2,2,1)
    plt.plot(residuals_ets)
    plt.title('Residuals')
    plt.xlabel('Date')
    plt.ylabel('Residuals')

    # Plot ACF of Residuals
    plt.subplot(2,2,2)
    plot_acf(residuals_ets, lags=7, ax=plt.gca())
    plt.title('ACF of Residuals')

    # Plot PACF of Residuals
    plt.subplot(2,2,3)
    plot_pacf(residuals_ets, lags=7, ax=plt.gca(), method='ywm')
    plt.title('PACF of Residuals')

    # QQ-Plot
    plt.subplot(2,2,4)
    sm.qqplot(residuals_ets, line='s', ax=plt.gca())
    plt.title('QQ-Plot of Residuals')

    plt.tight_layout()
    plt.show()

    # Perform Ljung-Box test for autocorrelation
    lb_test_ets = acorr_ljungbox(residuals_ets, lags=[7], return_df=True)
    print("\nLjung-Box Test Results:")
    print(lb_test_ets)

    # Interpretation
    if lb_test_ets['lb_pvalue'].iloc[-1] > 0.05:
        print("\nConclusion: Residuals are independently distributed (fail to reject H₀). Good fit.")
    else:
        print("\nConclusion: Residuals are not independently distributed (reject H₀). Consider revising the model.")


    return


@app.cell
def _(df, plot_series, sf_ets):
    # make 1-month prediction
    forecast_ETS = sf_ets.predict(h=30)
    plot_series(df=df[["ds", "y", "unique_id"]], forecasts_df=forecast_ETS)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
