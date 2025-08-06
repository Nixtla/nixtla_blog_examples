from unittest.mock import Mock, patch

import pandas as pd
import pytest

from src.forecasting import (
    calculate_metrics,
    compare_scenarios,
    generate_recommendation,
    initialize_client,
    validate_api_key,
)


class TestCalculateMetrics:
    def test_calculate_metrics_basic(self):
        # Mock forecast data
        forecast = pd.DataFrame(
            {
                "unique_id": ["PRODUCT_1", "PRODUCT_1", "PRODUCT_2", "PRODUCT_2"],
                "ds": pd.date_range("2023-01-01", periods=4, freq="D")[:4],
                "TimeGPT": [10, 12, 8, 9],
            }
        )

        # Mock scenario data
        scenario_df = pd.DataFrame(
            {"unique_id": ["PRODUCT_1", "PRODUCT_2"], "sell_price": [5.0, 3.0]}
        )

        metrics = calculate_metrics(forecast, scenario_df, horizon=2)

        assert metrics["total_units"] == 39  # Sum of TimeGPT values
        assert metrics["products_count"] == 2
        assert metrics["forecast_period_days"] == 2
        assert "by_product" in metrics

    def test_calculate_metrics_empty_forecast(self):
        forecast = pd.DataFrame()
        scenario_df = pd.DataFrame()

        metrics = calculate_metrics(forecast, scenario_df, horizon=14)

        assert metrics == {}

    def test_calculate_metrics_none_forecast(self):
        scenario_df = pd.DataFrame()

        metrics = calculate_metrics(None, scenario_df, horizon=14)

        assert metrics == {}


class TestCompareScenarios:
    def test_compare_scenarios_basic(self):
        baseline = {"total_revenue": 1000, "total_units": 200}

        scenario = {"total_revenue": 1100, "total_units": 180}

        comparison = compare_scenarios(baseline, scenario)

        assert comparison["revenue_change"] == 100
        assert pytest.approx(comparison["revenue_change_pct"], rel=1e-10) == 10.0
        assert comparison["units_change"] == -20
        assert pytest.approx(comparison["units_change_pct"], rel=1e-10) == -10.0

    def test_compare_scenarios_empty_data(self):
        comparison = compare_scenarios({}, {})
        assert comparison == {}

        comparison = compare_scenarios(None, None)
        assert comparison == {}


class TestGenerateRecommendation:
    def test_generate_recommendation_strong_positive(self):
        comparison = {"revenue_change": 150, "revenue_change_pct": 8.5}

        recommendation = generate_recommendation(comparison)

        assert "✅ Strong Recommendation" in recommendation
        assert "8.5%" in recommendation

    def test_generate_recommendation_moderate_positive(self):
        comparison = {"revenue_change": 30, "revenue_change_pct": 2.3}

        recommendation = generate_recommendation(comparison)

        assert "⚠️ Moderate Opportunity" in recommendation
        assert "2.3%" in recommendation

    def test_generate_recommendation_minimal_impact(self):
        comparison = {"revenue_change": -10, "revenue_change_pct": -1.2}

        recommendation = generate_recommendation(comparison)

        assert "⚠️ Caution" in recommendation
        assert "-1.2%" in recommendation

    def test_generate_recommendation_strong_negative(self):
        comparison = {"revenue_change": -100, "revenue_change_pct": -8.7}

        recommendation = generate_recommendation(comparison)

        assert "❌ Not Recommended" in recommendation
        assert "-8.7%" in recommendation

    def test_generate_recommendation_empty_data(self):
        recommendation = generate_recommendation({})

        assert "Unable to generate recommendation" in recommendation


class TestValidateApiKey:
    def test_validate_api_key_empty(self):
        is_valid, message = validate_api_key("")

        assert not is_valid
        assert "required" in message.lower()

    def test_validate_api_key_none(self):
        is_valid, message = validate_api_key(None)

        assert not is_valid
        assert "required" in message.lower()

    @patch("src.forecasting.NixtlaClient")
    def test_validate_api_key_valid(self, mock_client):
        # Mock successful client initialization
        mock_client.return_value = Mock()

        is_valid, message = validate_api_key("test_api_key")

        assert is_valid
        assert "valid" in message.lower()

    @patch("src.forecasting.NixtlaClient")
    def test_validate_api_key_invalid(self, mock_client):
        # Mock client initialization failure
        mock_client.side_effect = Exception("Invalid API key")

        is_valid, message = validate_api_key("invalid_key")

        assert not is_valid
        assert "Invalid API key" in message


class TestInitializeClient:
    @patch("src.forecasting.NixtlaClient")
    def test_initialize_client_success(self, mock_client):
        mock_instance = Mock()
        mock_client.return_value = mock_instance

        client = initialize_client("test_api_key")

        assert client == mock_instance
        mock_client.assert_called_once_with(api_key="test_api_key")


class TestBusinessLogic:
    def test_revenue_calculation_logic(self):
        # Test the core business logic for revenue calculations
        forecast = pd.DataFrame(
            {
                "unique_id": ["PRODUCT_A", "PRODUCT_A", "PRODUCT_B"],
                "ds": pd.date_range("2023-01-01", periods=3, freq="D"),
                "TimeGPT": [5, 7, 3],
            }
        )

        scenario_df = pd.DataFrame(
            {"unique_id": ["PRODUCT_A", "PRODUCT_B"], "sell_price": [10.0, 15.0]}
        )

        metrics = calculate_metrics(forecast, scenario_df, horizon=3)

        # Product A: (5 + 7) * 10 = 120
        # Product B: 3 * 15 = 45
        # Total: 165
        expected_revenue = (5 + 7) * 10.0 + 3 * 15.0

        assert metrics["total_revenue"] == expected_revenue
        assert metrics["total_units"] == 15

    def test_price_elasticity_scenario(self):
        # Test a realistic price elasticity scenario
        baseline = {"total_revenue": 1000, "total_units": 200}

        # 10% price decrease leads to 15% unit increase but 5% revenue decrease
        discounted_scenario = {
            "total_revenue": 950,  # 5% decrease
            "total_units": 230,  # 15% increase
        }

        comparison = compare_scenarios(baseline, discounted_scenario)

        assert pytest.approx(comparison["revenue_change_pct"], rel=1e-10) == -5.0
        assert pytest.approx(comparison["units_change_pct"], rel=1e-10) == 15.0

        recommendation = generate_recommendation(comparison)
        assert "Not Recommended" in recommendation
