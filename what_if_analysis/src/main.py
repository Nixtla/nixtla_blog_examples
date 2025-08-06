import sys
from pathlib import Path

import streamlit as st

# Add parent directory to Python path for direct imports
sys.path.append(str(Path(__file__).parent))

from data import (  # noqa: E402
    load_data,
    load_sample_data,
    prepare_data,
    validate_forecast_data,
)
from forecasting import (  # noqa: E402
    calculate_metrics,
    compare_scenarios,
    generate_forecast,
    generate_recommendation,
    initialize_client,
    validate_api_key,
)
from ui import (  # noqa: E402
    display_data_overview,
    render_header,
    render_sidebar,
    show_error_message,
    show_info_message,
    show_scenario_results,
    show_success_message,
)


def get_scenario_multipliers():
    """Define price change multipliers for each scenario"""
    return {
        "15% Discount": -0.15,
        "5% Discount": -0.05,
        "Current Pricing": 0.0,
        "10% Premium": 0.10,
    }


def run_scenario_analysis(client, df, scenarios, horizon):
    """Coordinate forecast generation for all scenarios"""
    results = {}
    scenario_multipliers = get_scenario_multipliers()

    total_scenarios = sum(scenarios.values())
    if total_scenarios == 0:
        show_error_message("Please select at least one pricing scenario")
        return results

    progress_container = st.container()

    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()

        current_scenario = 0
        baseline_metrics = None

        for scenario_name, is_selected in scenarios.items():
            if not is_selected:
                continue

            current_scenario += 1
            progress = current_scenario / total_scenarios
            status_text.text(
                f"Processing {scenario_name}... ({current_scenario}/{total_scenarios})"
            )
            progress_bar.progress(progress)

            # Apply price change to create scenario data
            price_change = scenario_multipliers[scenario_name]
            scenario_df = df.copy()
            scenario_df["sell_price"] = scenario_df["sell_price"] * (1 + price_change)

            # Generate forecast
            forecast = generate_forecast(client, scenario_df, horizon)

            if forecast is not None:
                # Calculate metrics
                metrics = calculate_metrics(forecast, scenario_df, horizon)

                # Store baseline for comparison
                if scenario_name == "Current Pricing":
                    baseline_metrics = metrics

                results[scenario_name] = {
                    "forecast": forecast,
                    "metrics": metrics,
                    "scenario_df": scenario_df,
                }
            else:
                show_error_message(f"Failed to generate forecast for {scenario_name}")

        # Add comparisons to baseline
        for scenario_name in results:
            if baseline_metrics and scenario_name != "Current Pricing":
                comparison = compare_scenarios(
                    baseline_metrics, results[scenario_name]["metrics"]
                )
                results[scenario_name]["comparison"] = comparison
                results[scenario_name]["recommendation"] = generate_recommendation(
                    comparison
                )
            elif scenario_name == "Current Pricing":
                results[scenario_name]["comparison"] = {}
                results[scenario_name][
                    "recommendation"
                ] = "📊 Baseline scenario for comparison"

        progress_bar.progress(1.0)
        status_text.text("✅ Analysis complete!")

    return results


def main():
    """Streamlit app entry point"""
    st.set_page_config(
        page_title="TimeGPT Retail What-If Analysis", page_icon="🛒", layout="wide"
    )

    render_header()

    # Get configuration from sidebar
    config = render_sidebar()

    # Validate API key
    if not config["api_key"]:
        show_info_message(
            "Please enter your TimeGPT API key in the sidebar to get started"
        )
        return

    # Load and prepare data
    try:
        if config["data_source"] == "Use Sample Data":
            df = load_sample_data()
            show_success_message("Sample data loaded successfully")
        else:
            if config["uploaded_file"] is None:
                show_info_message("Please upload a CSV file to continue")
                return
            df = load_data(config["uploaded_file"])
            show_success_message("Custom data loaded successfully")

        # Prepare data for TimeGPT
        df = prepare_data(df)

        # Validate data
        is_valid, validation_message = validate_forecast_data(df)
        if not is_valid:
            show_error_message(f"Data validation failed: {validation_message}")
            return

        # Display data overview
        display_data_overview(df)

        st.divider()

        # Run analysis button
        if st.button("🚀 Run Scenario Analysis", type="primary"):
            # Validate API key
            is_valid_key, key_message = validate_api_key(config["api_key"])
            if not is_valid_key:
                show_error_message(f"API key validation failed: {key_message}")
                return

            # Initialize TimeGPT client
            try:
                client = initialize_client(config["api_key"])
            except Exception as e:
                show_error_message(f"Failed to initialize TimeGPT client: {str(e)}")
                return

            # Run scenario analysis
            with st.spinner("Running scenario analysis..."):
                results = run_scenario_analysis(
                    client, df, config["scenarios"], config["horizon"]
                )

            if results:
                # Display results
                show_scenario_results(results)

                # Store results in session state for persistence
                st.session_state.results = results
            else:
                show_error_message(
                    "No results generated. Please check your configuration and try again."
                )

        # Show previous results if available
        elif hasattr(st.session_state, "results") and st.session_state.results:
            st.subheader("📋 Previous Analysis Results")
            show_scenario_results(st.session_state.results)

    except Exception as e:
        show_error_message(f"Application error: {str(e)}")


if __name__ == "__main__":
    main()
