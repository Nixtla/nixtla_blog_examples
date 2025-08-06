# TimeGPT Retail What-If Analysis

Transform retail pricing decisions with AI-powered demand forecasting. Upload your data, configure pricing scenarios, and get actionable revenue insights in minutes.

![TimeGPT Retail Forecasting](https://img.shields.io/badge/TimeGPT-Retail%20Forecasting-blue)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-dashboard-red.svg)

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- [uv](https://github.com/astral-sh/uv) package manager
- TimeGPT API key from [Nixtla](https://nixtla.io/)

### Installation

1. **Clone and navigate to the project:**
   ```bash
   cd what_if_analysis
   ```

2. **Install dependencies:**
   ```bash
   uv sync
   ```

3. **Run the application:**
   ```bash
   uv run streamlit run src/main.py
   ```

4. **Open your browser:** Navigate to `http://localhost:8501`

## 🎯 Features

### Business Intelligence
- **Revenue Impact Analysis:** Compare projected revenue across pricing strategies
- **Demand Elasticity:** Understand how price changes affect unit sales
- **Business Recommendations:** AI-generated insights for optimal pricing

### Technical Capabilities
- **TimeGPT Integration:** State-of-the-art forecasting without model training
- **Multiple Scenarios:** Test 15% discount, 5% discount, current pricing, and 10% premium
- **Interactive Dashboard:** Professional visualizations and easy-to-use interface
- **Data Validation:** Built-in checks for data quality and completeness

## 📊 How It Works

### 1. Data Input
- **Use sample data:** Built-in M5 retail dataset (5 products, 30 days)
- **Upload custom data:** CSV with columns: `unique_id`, `ds`, `y`, `sell_price`

### 2. Configure Scenarios
- Select pricing strategies to test
- Set forecast horizon (7-56 days)
- Enter your TimeGPT API key

### 3. Get Results
- Revenue projections for each scenario
- Business recommendations
- Interactive visualizations

## 🏢 Business Use Cases

### Retail Pricing Strategy
> "Should we run a 15% holiday promotion or maintain current prices?"

**Answer in 5 minutes:** Compare revenue impact, understand demand response, get clear recommendation.

### Product Launch Pricing
> "What's the optimal price point for maximum revenue?"

**Test multiple price points** simultaneously and identify the sweet spot.

### Competitive Response
> "How should we respond to competitor price cuts?"

**Model different responses** and choose the strategy that protects revenue.

## 📁 Project Structure

```
what_if_analysis/
├── pyproject.toml              # Dependencies and configuration
├── README.md                   # This file
├── src/
│   ├── __init__.py            # Package initialization
│   ├── main.py                # Streamlit app entry point
│   ├── data.py                # Data loading and processing
│   ├── forecasting.py         # TimeGPT integration
│   └── ui.py                  # Streamlit UI components
├── sample_data/
│   └── m5_sample.csv          # Demo dataset
└── tests/
    ├── conftest.py            # Test configuration
    ├── test_data.py           # Data processing tests
    ├── test_forecasting.py    # Forecasting logic tests
    └── test_scenarios.py      # Integration tests
```

## 🔧 Development

### Running Tests
```bash
# Run all tests
uv run pytest tests/

# Run specific test file
uv run pytest tests/test_data.py

# Run with coverage
uv run pytest tests/ --cov=src
```

### Code Quality
```bash
# Install pre-commit hooks
pre-commit install

# Run linting manually
uv run ruff check src/

# Format code
uv run ruff format src/
```

### Adding New Scenarios
To add custom pricing scenarios, modify `get_scenario_multipliers()` in `src/main.py`:

```python
def get_scenario_multipliers():
    return {
        "20% Flash Sale": -0.20,        # New scenario
        "Current Pricing": 0.0,
        "Premium Strategy": 0.15,       # Modified scenario
    }
```

## 📋 Data Format

### Required CSV Columns
- `unique_id`: Product identifier (string)
- `ds`: Date (YYYY-MM-DD format)
- `y`: Sales quantity (integer)
- `sell_price`: Unit price (float)

### Example Data
```csv
unique_id,ds,y,sell_price
FOODS_1_008_CA_1,2011-01-29,3,1.97
FOODS_1_008_CA_1,2011-01-30,0,1.97
FOODS_1_013_CA_1,2011-01-29,1,3.26
```

## 🔑 API Configuration

### Getting TimeGPT API Key
1. Sign up at [Nixtla](https://nixtla.io/)
2. Navigate to your dashboard
3. Copy your API key
4. Enter it in the Streamlit sidebar

### API Usage Notes
- API key is stored in session (not persisted)
- Each forecast consumes TimeGPT credits
- Failed requests are handled gracefully

## ⚡ Performance Tips

### For Large Datasets
- Limit to 5-10 products initially
- Use shorter forecast horizons (7-14 days)
- Consider data sampling for proof-of-concept

### For Production Use
- Implement API key environment variables
- Add caching for repeated scenarios
- Consider batch processing for multiple stores

## 🛠️ Troubleshooting

### Common Issues

**"Missing required columns" error:**
- Ensure CSV has: `unique_id`, `ds`, `y`, `sell_price`
- Check for typos in column names

**API key validation failed:**
- Verify your Nixtla account is active
- Check API key is copied correctly
- Ensure sufficient credits in your account

**No forecast generated:**
- Verify data has recent dates
- Check for null values in sales data
- Ensure products have sufficient history

### Getting Help
- Check the [Nixtla documentation](https://docs.nixtla.io/)
- Review test files for usage examples
- Open an issue for bugs or feature requests

## 🎓 Educational Value

This project demonstrates:
- **Time series forecasting** with modern AI
- **Business scenario modeling** and comparison
- **Production-ready code structure** with tests
- **Interactive data applications** with Streamlit
- **API integration** patterns

## 📄 License

This project is provided as an educational example. TimeGPT usage subject to [Nixtla's terms of service](https://nixtla.io/terms).

---

**Ready to optimize your pricing strategy?**
Run `uv run streamlit run src/main.py` and start forecasting!
