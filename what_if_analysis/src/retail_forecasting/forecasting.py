import streamlit as st
from nixtla import NixtlaClient


def initialize_client(api_key):
    """Setup TimeGPT client"""
    client = NixtlaClient(api_key=api_key)
    return client


def generate_forecast(client, data, horizon=14):
    """Generate single forecast using TimeGPT"""
    forecast_data = data[["unique_id", "ds", "y"]].copy()

    try:
        forecast = client.forecast(
            df=forecast_data, h=horizon, freq="D", time_col="ds", target_col="y"
        )
        return forecast
    except Exception as e:
        st.error(f"Forecast generation failed: {str(e)}")
        return None


def calculate_metrics(forecast, scenario_df, horizon):
    """Calculate business metrics from forecast"""
    if forecast is None or forecast.empty:
        return {}

    # Get average price for each product in the scenario
    price_by_product = scenario_df.groupby("unique_id")["sell_price"].mean()

    # Calculate projected revenue for each product
    forecast["sell_price"] = forecast["unique_id"].map(price_by_product).fillna(0)
    forecast["projected_revenue"] = forecast["TimeGPT"] * forecast["sell_price"]

    # Aggregate metrics
    metrics = {
        "total_units": forecast["TimeGPT"].sum(),
        "total_revenue": forecast["projected_revenue"].sum(),
        "avg_daily_units": forecast["TimeGPT"].sum() / horizon,
        "avg_daily_revenue": forecast["projected_revenue"].sum() / horizon,
        "products_count": forecast["unique_id"].nunique(),
        "forecast_period_days": horizon,
    }

    # Product-level metrics
    product_metrics = (
        forecast.groupby("unique_id")
        .agg({"TimeGPT": "sum", "projected_revenue": "sum", "sell_price": "first"})
        .round(2)
    )

    metrics["by_product"] = product_metrics.to_dict("index")

    return metrics


def compare_scenarios(baseline_metrics, scenario_metrics):
    """Compare scenario metrics against baseline"""
    if not baseline_metrics or not scenario_metrics:
        return {}

    comparison = {
        "revenue_change": scenario_metrics["total_revenue"]
        - baseline_metrics["total_revenue"],
        "revenue_change_pct": (
            (scenario_metrics["total_revenue"] / baseline_metrics["total_revenue"]) - 1
        )
        * 100,
        "units_change": scenario_metrics["total_units"]
        - baseline_metrics["total_units"],
        "units_change_pct": (
            (scenario_metrics["total_units"] / baseline_metrics["total_units"]) - 1
        )
        * 100,
        "scenario_revenue": scenario_metrics["total_revenue"],
        "baseline_revenue": baseline_metrics["total_revenue"],
    }

    return comparison


def generate_recommendation(comparison_results):
    """Generate business recommendation based on scenario comparison"""
    if not comparison_results:
        return "Unable to generate recommendation - insufficient data"

    revenue_change = comparison_results.get("revenue_change", 0)
    revenue_change_pct = comparison_results.get("revenue_change_pct", 0)

    if revenue_change_pct > 5:
        return f"✅ Strong Recommendation: Implement this pricing strategy. Projected revenue increase of ${revenue_change:,.0f} ({revenue_change_pct:.1f}%)"
    elif revenue_change_pct > 0:
        return f"⚠️ Moderate Opportunity: This strategy shows positive impact with ${revenue_change:,.0f} increase ({revenue_change_pct:.1f}%)"
    elif revenue_change_pct > -5:
        return f"⚠️ Caution: Minimal revenue impact of ${revenue_change:,.0f} ({revenue_change_pct:.1f}%). Consider other factors"
    else:
        return f"❌ Not Recommended: Significant revenue decrease of ${revenue_change:,.0f} ({revenue_change_pct:.1f}%)"


def validate_api_key(api_key):
    """Validate TimeGPT API key by testing connection"""
    if not api_key or len(api_key.strip()) == 0:
        return False, "API key is required"

    try:
        NixtlaClient(api_key=api_key)
        # Test with a simple validation call
        return True, "API key is valid"
    except Exception as e:
        return False, f"Invalid API key: {str(e)}"
