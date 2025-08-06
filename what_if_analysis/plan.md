# TimeGPT Retail What-If Analysis - Implementation Plan

## Project Overview

Simple Streamlit application for retail demand forecasting and pricing scenario analysis using TimeGPT. Designed for business technical managers who need to perform sophisticated what-if analysis without deep technical expertise.

**Core Functionality:**
- Upload M5 retail dataset
- Configure pricing scenarios
- Generate TimeGPT forecasts
- Compare revenue impact
- Get business recommendations

## Executive Summary

### The Business Challenge
Retail pricing decisions directly impact revenue, but traditional forecasting methods are slow, require specialized expertise, and often miss market dynamics. Business teams need quick, reliable answers to critical questions:

- "How will a 10% price increase affect demand next month?"
- "Should we run a promotion during the holiday season?"
- "What pricing strategy maximizes revenue for our top products?"

### The Solution
TimeGPT Retail What-If Analysis transforms complex forecasting into simple business decisions. Upload your sales data, configure pricing scenarios, and get AI-powered revenue projections in minutes—no data science team required.

### Key Business Benefits

**Immediate ROI Visibility**
- Compare revenue impact across multiple pricing strategies
- Identify optimal pricing with clear profit projections
- Make data-driven decisions in under 5 minutes

**Risk-Free Decision Making**
- Test pricing strategies before implementation
- Understand demand elasticity for each product
- Avoid costly pricing mistakes through simulation

**Competitive Advantage**
- React quickly to market changes with rapid scenario analysis
- Optimize pricing based on AI predictions, not guesswork
- Scale pricing decisions across product portfolios

### Real-World Impact
A typical use case: Testing a 15% holiday discount vs. maintaining current prices might reveal that the discount increases unit sales by 30% but reduces overall revenue by 8%. This insight prevents a costly pricing mistake while identifying the optimal strategy.

**Time Investment**: 5 minutes to upload data and configure scenarios
**Expected Outcome**: Clear revenue projections with actionable recommendations
**Business Value**: Prevent revenue losses, identify profit opportunities, optimize pricing strategy

## Required Libraries

### Core Dependencies
```python
# Web application framework
import streamlit as st

# Data processing
import pandas as pd
import numpy as np

# TimeGPT integration
from nixtla import NixtlaClient

# Visualization
import plotly.express as px
import plotly.graph_objects as go

# Date handling
from datetime import datetime, timedelta
```

### Installation
Using pyproject.toml for proper Python package management:

```toml
[project]
name = "retail-forecasting-timegpt"
version = "0.1.0"
description = "TimeGPT-powered retail forecasting and pricing scenario analysis"
requires-python = ">=3.8"
dependencies = [
    "streamlit",
    "pandas",
    "numpy",
    "nixtla",
    "plotly",
    "hydra-core",
    "python-dotenv"
]

[project.optional-dependencies]
dev = [
    "pytest",
    "pre-commit",
    "ruff"
]

[tool.ruff]
# Exclude a variety of commonly ignored directories.
exclude = [
    ".bzr",
    ".direnv",
    ".eggs",
    ".git",
    ".git-rewrite",
    ".hg",
    ".mypy_cache",
    ".nox",
    ".pants.d",
    ".pytype",
    ".ruff_cache",
    ".svn",
    ".tox",
    ".venv",
    "__pypackages__",
    "_build",
    "buck-out",
    "build",
    "dist",
    "node_modules",
    "venv",
]

# Same as Black.
line-length = 88

[tool.ruff.lint]
ignore = ["E501"]
select = ["B","C","E","F","W","B9", "I", "Q"]

[tool.ruff.format]
quote-style = "double"
indent-style = "tab"
skip-magic-trailing-comma = false

[tool.ruff.lint.mccabe]
max-complexity = 10
```

## Application Architecture

Following **KISS** (Keep It Simple, Stupid) and **SRP** (Single Responsibility Principle):

### Module Structure

**data.py** - Data handling and processing
```python
def load_sample_data():
    """Load built-in sample dataset for demo"""

def load_data(uploaded_file):
    """Load and validate CSV file from user upload"""

def prepare_data(df, max_products=5):
    """Transform M5 data to TimeGPT format"""

def create_scenario_data(df, price_change):
    """Apply price change to dataset"""
```

**forecasting.py** - TimeGPT integration
```python
def initialize_client(api_key):
    """Setup TimeGPT client"""

def generate_forecast(client, data, horizon):
    """Generate single forecast"""

def calculate_metrics(forecast, scenario_df, horizon):
    """Calculate business metrics"""
```

**ui.py** - Streamlit components
```python
def render_sidebar():
    """Configuration sidebar"""

def display_data_overview(df):
    """Show data summary and charts"""

def show_scenario_results(results):
    """Display results table and visualizations"""
```

**main.py** - Application orchestration
```python
def run_scenario_analysis(client, df, scenarios, horizon):
    """Coordinate forecast generation for all scenarios"""

def main():
    """Streamlit app entry point"""
```

## File Structure

```
/what_if_analysis/
├── pyproject.toml (dependencies and project config)
├── plan.md (this file)
├── README.md (usage instructions)
├── sample_data/
│   └── m5_sample.csv (demo dataset: 5 products, 30 days)
├── tests/
│   ├── __init__.py
│   ├── test_data.py (data processing tests)
│   ├── test_forecasting.py (forecasting logic tests)
│   └── test_scenarios.py (integration tests)
└── src/
    └── retail_forecasting/
        ├── __init__.py
        ├── main.py (Streamlit app entry point)
        ├── data.py (data loading and processing)
        ├── forecasting.py (TimeGPT integration)
        └── ui.py (Streamlit UI components)
```

## Running the Application

Using pyproject.toml with proper Python package management:

```bash
# Install in development mode
uv sync

# Run the application
uv run streamlit run src/retail_forecasting/main.py

# Run tests
uv run pytest tests/
```

## Sample Dataset

### Demo Data Specifications
- **Products**: 5 FOODS category items from M5 dataset
- **Timeframe**: 30 days (enough for meaningful patterns)
- **Stores**: Single California store (CA_1)
- **Format**: Ready-to-use TimeGPT format (unique_id, ds, y, sell_price)
- **Size**: ~150 rows (5 products × 30 days)

### Sample Data Structure
```csv
unique_id,ds,y,sell_price
FOODS_1_008_CA_1,2011-01-29,3,1.97
FOODS_1_008_CA_1,2011-01-30,0,1.97
FOODS_1_013_CA_1,2011-01-29,1,3.26
...
```

## Implementation Principles

### KISS (Keep It Simple)
- Single file application for easy deployment
- Minimal configuration required
- Clear, intuitive interface
- Straightforward data flow

### SRP (Single Responsibility)
- Each function has one clear purpose
- Modular design for easy testing
- Separate concerns (data, UI, forecasting)
- Clean function interfaces

### Error Handling Philosophy
- **Avoid unnecessary try/except blocks**
- Only handle expected failures (API calls, file uploads)
- Let Streamlit handle UI errors gracefully
- Use validation functions instead of exception handling

## Key Features

### 1. Configuration Panel (Sidebar)
- API key input (password field)
- Data source selection: "Use Sample Data" or "Upload Custom File"
- File upload widget (when custom selected)
- Forecast horizon slider (7-56 days)
- Pricing scenario checkboxes

### 2. Data Overview (Main Panel)
- Dataset statistics
- Historical sales visualization
- Price distribution analysis
- Product selection interface

### 3. Scenario Analysis Engine
- Multiple pricing strategies:
  - 15% Discount
  - 5% Discount
  - Current Pricing
  - 10% Premium
- Real-time forecast generation
- Progress tracking
- Revenue calculations

### 4. Results Dashboard
- Scenario comparison table
- Revenue impact charts
- Business recommendations
- Optimal strategy identification

## Business Value

### For Technical Managers
- **Immediate ROI**: Clear revenue impact calculations
- **Decision Support**: Data-driven pricing recommendations
- **Stakeholder Communication**: Visual results for presentations
- **Operational Efficiency**: Automated scenario testing

### For Business Users
- **No Coding Required**: Point-and-click interface
- **Professional Output**: Publication-ready visualizations
- **Actionable Insights**: Clear business recommendations
- **Scalable Analysis**: Handle multiple products simultaneously

## Success Metrics

1. **Usability**: Non-technical users can generate forecasts in under 5 minutes
2. **Accuracy**: TimeGPT forecasts help identify optimal pricing strategies
3. **Adoption**: Business teams use the tool for regular pricing decisions
4. **Value**: Demonstrable revenue impact from pricing optimizations


### Module Responsibilities
- **data.py**: All data loading, validation, and transformation
- **forecasting.py**: TimeGPT client management and forecast generation
- **ui.py**: Streamlit components and visualization
- **main.py**: Application orchestration and entry point

## Testing Strategy

### Focus Areas
Unit tests focus on core business logic and data processing, avoiding complex UI testing that provides limited value for the development time investment.

**Test Coverage Priorities:**
1. **Data Processing Logic** - Sample data loading, price scenario application
2. **Revenue Calculations** - Pricing multipliers, forecast-based revenue projections
3. **Business Logic** - Scenario comparison, recommendation logic
4. **Skip**: Streamlit UI components, API integration mocking, edge cases

### Test Execution
- **Run tests**: `pytest tests/`
- **Target**: ~15-20 focused tests covering critical business logic
- **Philosophy**: Test behavior, not implementation details

## Pre-commit Configuration

### Setup and Installation

Pre-commit hooks ensure code quality and consistency before commits. Install development dependencies and initialize hooks:

```bash
# Install pre-commit hooks
pre-commit install
```

### Pre-commit Configuration File

Create `.pre-commit-config.yaml` in the project root:

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.6
    hooks:
      # Run the linter
      - id: ruff
        args: ["--fix"]
      # Run the formatter
      - id: ruff-format
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
```
