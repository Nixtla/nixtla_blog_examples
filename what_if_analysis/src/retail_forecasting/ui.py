import pandas as pd
import plotly.express as px
import streamlit as st


def render_sidebar():
    """Configuration sidebar"""
    st.sidebar.title("⚙️ Configuration")

    # API Key input
    api_key = st.sidebar.text_input(
        "TimeGPT API Key", type="password", help="Enter your Nixtla TimeGPT API key"
    )

    # Data source selection
    st.sidebar.subheader("Data Source")
    data_source = st.sidebar.radio(
        "Choose data source:", ["Use Sample Data", "Upload Custom File"]
    )

    uploaded_file = None
    if data_source == "Upload Custom File":
        uploaded_file = st.sidebar.file_uploader(
            "Upload CSV file",
            type="csv",
            help="File must contain columns: unique_id, ds, y, sell_price",
        )

    # Forecast horizon
    st.sidebar.subheader("Forecast Settings")
    horizon = st.sidebar.slider(
        "Forecast Horizon (days)",
        min_value=7,
        max_value=56,
        value=14,
        help="Number of days to forecast",
    )

    # Pricing scenarios
    st.sidebar.subheader("Pricing Scenarios")
    scenarios = {}

    scenarios["15% Discount"] = st.sidebar.checkbox("15% Discount", value=True)
    scenarios["5% Discount"] = st.sidebar.checkbox("5% Discount", value=True)
    scenarios["Current Pricing"] = st.sidebar.checkbox("Current Pricing", value=True)
    scenarios["10% Premium"] = st.sidebar.checkbox("10% Premium", value=True)

    return {
        "api_key": api_key,
        "data_source": data_source,
        "uploaded_file": uploaded_file,
        "horizon": horizon,
        "scenarios": scenarios,
    }


def display_data_overview(df):
    """Show data summary and charts"""
    st.subheader("📊 Data Overview")

    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Products", df["unique_id"].nunique())
    with col2:
        st.metric("Total Sales", f"{df['y'].sum():,}")
    with col3:
        st.metric("Avg Price", f"${df['sell_price'].mean():.2f}")
    with col4:
        st.metric("Date Range", f"{df['ds'].nunique()} days")

    # Historical sales chart
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Historical Sales by Product")
        sales_by_product = df.groupby(["unique_id", "ds"])["y"].sum().reset_index()
        fig = px.line(
            sales_by_product,
            x="ds",
            y="y",
            color="unique_id",
            title="Daily Sales Volume",
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Price Distribution")
        price_by_product = df.groupby("unique_id")["sell_price"].first().reset_index()
        fig = px.bar(
            price_by_product,
            x="unique_id",
            y="sell_price",
            title="Current Prices by Product",
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)


def show_scenario_results(results):
    """Display results table and visualizations"""
    if not results:
        st.warning("No scenario results to display")
        return

    st.subheader("🎯 Scenario Analysis Results")

    # Create results dataframe for display
    results_data = []
    for scenario_name, data in results.items():
        if "metrics" in data and data["metrics"]:
            metrics = data["metrics"]
            comparison = data.get("comparison", {})

            results_data.append(
                {
                    "Scenario": scenario_name,
                    "Projected Units": f"{metrics.get('total_units', 0):,.0f}",
                    "Projected Revenue": f"${metrics.get('total_revenue', 0):,.0f}",
                    "Revenue Change": f"${comparison.get('revenue_change', 0):,.0f}",
                    "Revenue Change %": f"{comparison.get('revenue_change_pct', 0):+.1f}%",
                    "Recommendation": data.get("recommendation", ""),
                }
            )

    if results_data:
        results_df = pd.DataFrame(results_data)
        st.dataframe(results_df, use_container_width=True)

        # Revenue comparison chart
        st.subheader("Revenue Comparison")
        revenue_data = [
            {
                "Scenario": row["Scenario"],
                "Revenue": float(
                    row["Projected Revenue"].replace("$", "").replace(",", "")
                ),
            }
            for row in results_data
        ]
        revenue_df = pd.DataFrame(revenue_data)

        fig = px.bar(
            revenue_df,
            x="Scenario",
            y="Revenue",
            title="Projected Revenue by Scenario",
            color="Revenue",
            color_continuous_scale="viridis",
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

        # Best scenario highlight
        if len(results_data) > 1:
            best_scenario = max(
                results_data,
                key=lambda x: float(
                    x["Projected Revenue"].replace("$", "").replace(",", "")
                ),
            )
            st.success(
                f"🏆 **Best Scenario**: {best_scenario['Scenario']} with {best_scenario['Projected Revenue']} projected revenue"
            )


def display_progress_bar(progress, status_text):
    """Show progress bar for scenario processing"""
    progress_bar = st.progress(progress)
    st.text(status_text)
    return progress_bar


def show_error_message(error_msg):
    """Display error message"""
    st.error(f"❌ {error_msg}")


def show_success_message(message):
    """Display success message"""
    st.success(f"✅ {message}")


def show_info_message(message):
    """Display info message"""
    st.info(f"ℹ️ {message}")


def display_forecast_details(forecast_data, scenario_name):
    """Show detailed forecast information"""
    if forecast_data is None or forecast_data.empty:
        st.warning(f"No forecast data available for {scenario_name}")
        return

    st.subheader(f"📈 Forecast Details - {scenario_name}")

    # Forecast chart
    fig = px.line(
        forecast_data,
        x="ds",
        y="TimeGPT",
        color="unique_id",
        title=f"Forecast for {scenario_name}",
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    # Summary stats
    st.write("**Forecast Summary:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Forecast Units", f"{forecast_data['TimeGPT'].sum():,.0f}")
    with col2:
        st.metric("Products", forecast_data["unique_id"].nunique())
    with col3:
        st.metric("Forecast Days", len(forecast_data["ds"].unique()))


def render_header():
    """Render application header"""
    st.title("🛒 TimeGPT Retail What-If Analysis")
    st.markdown(
        """
	Transform retail pricing decisions with AI-powered demand forecasting.
	Upload your data, configure pricing scenarios, and get actionable revenue insights in minutes.
	"""
    )
    st.divider()
