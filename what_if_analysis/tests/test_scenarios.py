from unittest.mock import patch

import pandas as pd
import pytest

from src.data import create_scenario_data, load_sample_data
from src.forecasting import calculate_metrics, compare_scenarios
from src.main import get_scenario_multipliers


class TestScenarioIntegration:
    def test_scenario_multipliers_configuration(self):
        """Test that scenario multipliers are properly configured"""
        multipliers = get_scenario_multipliers()

        expected_scenarios = [
            "15% Discount",
            "5% Discount",
            "Current Pricing",
            "10% Premium",
        ]
        assert all(scenario in multipliers for scenario in expected_scenarios)

        assert multipliers["15% Discount"] == -0.15
        assert multipliers["5% Discount"] == -0.05
        assert multipliers["Current Pricing"] == 0.0
        assert multipliers["10% Premium"] == 0.10

    def test_end_to_end_scenario_processing(self):
        """Test complete scenario processing workflow"""
        # Load sample data
        df = load_sample_data()

        # Test each scenario
        multipliers = get_scenario_multipliers()

        for _, price_change in multipliers.items():
            # Create scenario data
            scenario_df = create_scenario_data(df, price_change)

            # Verify price changes are applied correctly
            original_prices = df["sell_price"].values
            new_prices = scenario_df["sell_price"].values
            expected_prices = original_prices * (1 + price_change)

            assert pytest.approx(new_prices, rel=1e-10) == expected_prices

            # Verify data structure is preserved
            assert scenario_df.shape == df.shape
            assert list(scenario_df.columns) == list(df.columns)
            assert scenario_df["unique_id"].equals(df["unique_id"])
            assert scenario_df["ds"].equals(df["ds"])
            assert scenario_df["y"].equals(df["y"])

    def test_discount_scenario_price_changes(self):
        """Test specific discount scenarios with realistic data"""
        df = pd.DataFrame(
            {
                "unique_id": ["PRODUCT_1", "PRODUCT_2"],
                "ds": pd.to_datetime(["2023-01-01", "2023-01-01"]),
                "y": [10, 5],
                "sell_price": [100.0, 50.0],
            }
        )

        # Test 15% discount
        discount_df = create_scenario_data(df, -0.15)
        expected_prices = [85.0, 42.5]  # 15% off
        assert (
            pytest.approx(discount_df["sell_price"].tolist(), rel=1e-10)
            == expected_prices
        )

        # Test 10% premium
        premium_df = create_scenario_data(df, 0.10)
        expected_prices = [110.0, 55.0]  # 10% markup
        assert (
            pytest.approx(premium_df["sell_price"].tolist(), rel=1e-10)
            == expected_prices
        )

    @patch("src.forecasting.generate_forecast")
    def test_scenario_comparison_workflow(self, mock_forecast):
        """Test the complete scenario comparison workflow"""
        # Mock forecast data
        mock_forecast_data = pd.DataFrame(
            {
                "unique_id": ["PRODUCT_1", "PRODUCT_1", "PRODUCT_2", "PRODUCT_2"],
                "ds": pd.date_range("2023-01-01", periods=4, freq="D"),
                "TimeGPT": [10, 12, 8, 9],
            }
        )
        mock_forecast.return_value = mock_forecast_data

        # Create test data
        df = pd.DataFrame(
            {
                "unique_id": ["PRODUCT_1", "PRODUCT_2"],
                "ds": pd.to_datetime(["2023-01-01", "2023-01-01"]),
                "y": [10, 8],
                "sell_price": [5.0, 3.0],
            }
        )

        # Test baseline vs discount scenario
        baseline_df = create_scenario_data(df, 0.0)  # Current pricing
        discount_df = create_scenario_data(df, -0.10)  # 10% discount

        # Calculate metrics for both scenarios
        baseline_metrics = calculate_metrics(mock_forecast_data, baseline_df, horizon=2)
        discount_metrics = calculate_metrics(mock_forecast_data, discount_df, horizon=2)

        # Compare scenarios
        comparison = compare_scenarios(baseline_metrics, discount_metrics)

        # Verify discount scenario has lower revenue (due to lower prices)
        assert comparison["revenue_change"] < 0
        assert comparison["revenue_change_pct"] < 0

        # Units should remain the same (same forecast)
        assert comparison["units_change"] == 0


class TestScenarioDataIntegrity:
    def test_scenario_data_independence(self):
        """Test that scenario modifications don't affect original data"""
        original_df = pd.DataFrame(
            {
                "unique_id": ["PRODUCT_1"],
                "ds": pd.to_datetime(["2023-01-01"]),
                "y": [10],
                "sell_price": [100.0],
            }
        )

        # Create multiple scenarios
        scenarios = [-0.20, -0.10, 0.0, 0.10, 0.20]
        scenario_dfs = []

        for price_change in scenarios:
            scenario_df = create_scenario_data(original_df, price_change)
            scenario_dfs.append(scenario_df)

        # Verify original data is unchanged
        assert original_df["sell_price"].iloc[0] == 100.0

        # Verify each scenario has correct prices
        expected_prices = [80.0, 90.0, 100.0, 110.0, 120.0]
        for scenario_df, expected_price in zip(scenario_dfs, expected_prices):
            assert (
                pytest.approx(scenario_df["sell_price"].iloc[0], rel=1e-10)
                == expected_price
            )

    def test_scenario_processing_with_sample_data(self):
        """Test scenario processing with actual sample data"""
        df = load_sample_data()

        # Verify sample data has expected structure
        assert not df.empty
        assert df["unique_id"].nunique() == 5

        # Test all scenarios
        multipliers = get_scenario_multipliers()

        for _, multiplier in multipliers.items():
            scenario_df = create_scenario_data(df, multiplier)

            # Verify scenario data integrity
            assert scenario_df.shape[0] == df.shape[0]
            assert scenario_df["unique_id"].nunique() == df["unique_id"].nunique()

            # Verify price changes
            if multiplier == 0.0:  # Current pricing
                assert scenario_df["sell_price"].equals(df["sell_price"])
            else:
                price_ratio = (
                    scenario_df["sell_price"].iloc[0] / df["sell_price"].iloc[0]
                )
                expected_ratio = 1 + multiplier
                assert pytest.approx(price_ratio, rel=1e-10) == expected_ratio


class TestEdgeCases:
    def test_zero_price_handling(self):
        """Test handling of zero prices in scenarios"""
        df = pd.DataFrame(
            {
                "unique_id": ["PRODUCT_1"],
                "ds": pd.to_datetime(["2023-01-01"]),
                "y": [10],
                "sell_price": [0.0],
            }
        )

        # Test various price changes with zero base price
        scenarios = [-0.10, 0.0, 0.10]

        for price_change in scenarios:
            scenario_df = create_scenario_data(df, price_change)
            # All should result in 0.0 price
            assert scenario_df["sell_price"].iloc[0] == 0.0

    def test_extreme_price_changes(self):
        """Test extreme price change scenarios"""
        df = pd.DataFrame(
            {
                "unique_id": ["PRODUCT_1"],
                "ds": pd.to_datetime(["2023-01-01"]),
                "y": [10],
                "sell_price": [100.0],
            }
        )

        # Test extreme discount (90% off)
        extreme_discount_df = create_scenario_data(df, -0.90)
        assert (
            pytest.approx(extreme_discount_df["sell_price"].iloc[0], rel=1e-10) == 10.0
        )

        # Test extreme premium (200% markup)
        extreme_premium_df = create_scenario_data(df, 2.0)
        assert (
            pytest.approx(extreme_premium_df["sell_price"].iloc[0], rel=1e-10) == 300.0
        )

    def test_single_product_scenario(self):
        """Test scenario processing with single product"""
        df = pd.DataFrame(
            {
                "unique_id": ["PRODUCT_SINGLE"] * 7,
                "ds": pd.date_range("2023-01-01", periods=7, freq="D"),
                "y": [5, 3, 8, 2, 6, 4, 7],
                "sell_price": [25.0] * 7,
            }
        )

        # Test 5% discount scenario
        scenario_df = create_scenario_data(df, -0.05)

        # Verify all prices are reduced by 5%
        expected_price = 25.0 * 0.95
        assert all(price == expected_price for price in scenario_df["sell_price"])
