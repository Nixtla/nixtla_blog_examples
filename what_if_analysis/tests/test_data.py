import os
import tempfile

import pandas as pd
import pytest

from src.retail_forecasting.data import (
    create_scenario_data,
    get_data_summary,
    load_data,
    load_sample_data,
    prepare_data,
    validate_forecast_data,
)


class TestDataLoading:
    def test_load_sample_data(self):
        df = load_sample_data()
        assert not df.empty
        assert all(col in df.columns for col in ["unique_id", "ds", "y", "sell_price"])
        assert df["unique_id"].nunique() == 5

    def test_load_data_valid_csv(self):
        # Create a temporary CSV file
        test_data = pd.DataFrame(
            {
                "unique_id": ["PRODUCT_1", "PRODUCT_1", "PRODUCT_2"],
                "ds": ["2023-01-01", "2023-01-02", "2023-01-01"],
                "y": [10, 15, 8],
                "sell_price": [5.99, 5.99, 3.49],
            }
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            test_data.to_csv(f.name, index=False)
            result = load_data(f.name)
            os.unlink(f.name)

        assert not result.empty
        assert list(result.columns) == ["unique_id", "ds", "y", "sell_price"]

    def test_load_data_missing_columns(self):
        # Create CSV with missing columns
        test_data = pd.DataFrame(
            {
                "unique_id": ["PRODUCT_1"],
                "ds": ["2023-01-01"],
                "y": [10],
                # Missing sell_price column
            }
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            test_data.to_csv(f.name, index=False)

            with pytest.raises(ValueError, match="Missing required columns"):
                load_data(f.name)

            os.unlink(f.name)


class TestDataPreparation:
    def test_prepare_data_basic(self):
        df = pd.DataFrame(
            {
                "unique_id": ["PRODUCT_1", "PRODUCT_1", "PRODUCT_2", "PRODUCT_2"],
                "ds": ["2023-01-01", "2023-01-02", "2023-01-01", "2023-01-02"],
                "y": [10, 15, 8, 12],
                "sell_price": [5.99, 5.99, 3.49, 3.49],
            }
        )

        result = prepare_data(df)

        assert not result.empty
        assert pd.api.types.is_datetime64_any_dtype(result["ds"])
        assert result["y"].dtype == int
        assert result["sell_price"].dtype == float

    def test_prepare_data_max_products_limit(self):
        df = pd.DataFrame(
            {
                "unique_id": ["P1", "P2", "P3", "P4", "P5", "P6"] * 2,
                "ds": ["2023-01-01"] * 6 + ["2023-01-02"] * 6,
                "y": [10] * 12,
                "sell_price": [5.0] * 12,
            }
        )

        result = prepare_data(df, max_products=3)

        assert result["unique_id"].nunique() <= 3

    def test_create_scenario_data(self):
        df = pd.DataFrame(
            {
                "unique_id": ["PRODUCT_1", "PRODUCT_2"],
                "ds": ["2023-01-01", "2023-01-01"],
                "y": [10, 8],
                "sell_price": [5.0, 3.0],
            }
        )

        # Test 10% price increase
        result = create_scenario_data(df, 0.10)

        expected_prices = [5.5, 3.3]
        assert (
            pytest.approx(result["sell_price"].tolist(), rel=1e-10) == expected_prices
        )

        # Original dataframe should be unchanged
        assert df["sell_price"].tolist() == [5.0, 3.0]


class TestDataValidation:
    def test_validate_forecast_data_valid(self):
        df = pd.DataFrame(
            {
                "unique_id": ["PRODUCT_1", "PRODUCT_2"],
                "ds": ["2023-01-01", "2023-01-01"],
                "y": [10, 8],
            }
        )

        is_valid, message = validate_forecast_data(df)
        assert is_valid
        assert "valid" in message.lower()

    def test_validate_forecast_data_empty(self):
        df = pd.DataFrame()

        is_valid, message = validate_forecast_data(df)
        assert not is_valid
        assert "empty" in message.lower()

    def test_validate_forecast_data_missing_columns(self):
        df = pd.DataFrame(
            {
                "unique_id": ["PRODUCT_1"],
                "ds": ["2023-01-01"],
                # Missing y column
            }
        )

        is_valid, message = validate_forecast_data(df)
        assert not is_valid
        assert "missing" in message.lower()

    def test_validate_forecast_data_null_values(self):
        df = pd.DataFrame(
            {
                "unique_id": ["PRODUCT_1", "PRODUCT_2"],
                "ds": ["2023-01-01", "2023-01-01"],
                "y": [10, None],  # Null value
            }
        )

        is_valid, message = validate_forecast_data(df)
        assert not is_valid
        assert "null" in message.lower()


class TestDataSummary:
    def test_get_data_summary(self):
        df = pd.DataFrame(
            {
                "unique_id": ["PRODUCT_1", "PRODUCT_1", "PRODUCT_2"],
                "ds": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-01"]),
                "y": [10, 15, 8],
                "sell_price": [5.0, 5.0, 3.0],
            }
        )

        summary = get_data_summary(df)

        assert summary["total_products"] == 2
        assert summary["total_sales"] == 33
        assert summary["avg_price"] == pytest.approx(4.33, rel=1e-2)
        assert summary["total_days"] == 2
        assert "2023-01-01" in summary["date_range"]
