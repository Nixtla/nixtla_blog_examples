# Contribution Guidelines

## Table of Contents

- [Writing Checklist](#writing-checklist)
- [Write Article Draft](#write-article-draft)
- [Write Code](#write-code)
- [Pull Request Process](#pull-request-process)

## Writing Checklist

### Writing Style Checklist


- [ ] **Use action verbs instead of passive voice**
  ❌ *The model was trained on the dataset.*
  ✅ *Train the model on the dataset.*

- [ ] **Explain what the code does before presenting it**
  Define the baseline models to use for forecasting:

  ```python
  models = [
      HistoricAverage(),
      Naive(),
      SeasonalNaive(season_length = 4),
      WindowAverage(window_size=4)
  ]
  ```

- [ ] **Provide a clear explanation of the code immediately after presenting it**
  This list includes four common baseline models:

  * `HistoricAverage()`: Mean Forecast
  * `Naive()`: Naive Forecast
  * `SeasonalNaive(season_length=4)`: Seasonal Naive Forecast for quarterly seasonality
  * `WindowAverage(window_size=4)`: Averages the last 4 quarters to forecast future values

- [ ] **Avoid filler words (just, really, actually, basically, in order to)**
  ❌ *You just need to install the library to get started.*
  ✅ *Install the library to get started.*

- [ ] **Avoid phrases that hide the subject (there is, there are)**
  ❌ *There are several reasons to use Nixtla tools.*
  ✅ *Nixtla tools offer several advantages.*

- [ ] **Avoid ambiguous pronouns (it, this, that)**
  ❌ *This improves accuracy.*
  ✅ *The seasonal adjustment improves accuracy.*

- [ ] **Avoid -ing verb forms when possible (using, having, going)**
  ❌ *Using this model helps reduce error.*
  ✅ *This model helps reduce error.*

- [ ] **Avoid culture-specific references or idioms**
  ❌ *There's no silver bullet for intermittent demand.*
  ✅ *No single model solves all intermittent demand problems.*

- [ ] **Structure content for quick scanning with clear headings and bullet points**
- [ ] **Keep paragraphs short (2–4 sentences maximum)**

### Audience Assumptions Checklist

- [ ] Write for data scientists who are familiar with basic time series concepts
- [ ] Explain Nixtla tools as if readers are new to them
- [ ] Include enough examples for quick understanding of concepts

### Content Checklist

- [ ] Begin with a real-world time series problem or use case
- [ ] Present a solution that addresses the problem, making it the central focus of your article
- [ ] Include clear explanations of time series concepts and terminology
- [ ] When mentioning install commands or configuration flags, keep them minimal and link out to official docs for details

## Write Article Draft

1. Create your blog post in [HackMD](https://hackmd.io)
2. Follow [these instructions](https://hackmd.io/c/tutorials/%2F%40docs%2Finvite-others-to-a-private-note-en) to share your draft with khuyentran@nixtla.io for review

## Write Code

### Environment Setup

#### Install uv

[uv](https://github.com/astral.sh/uv) is a fast Python package installer and resolver.

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Verify installation
uv --version
```

#### Install Dependencies

```bash
# Install dependencies from pyproject.toml
uv sync
```

#### Install Pre-commit Hooks

We use pre-commit to ensure code quality and consistency.

```bash
# Install pre-commit hooks
uv run pre-commit install
```

### Working with Marimo Notebooks

#### Creating a New Notebook

Create a new notebook in the `notebooks` directory using marimo:

```bash
uv run marimo edit notebooks/your_notebook_name.py --sandbox
```

#### Notebook Creation Guidelines

- [ ] Use snake_case for notebook names (e.g., `anomaly_detection.py`, `intermittent_forecasting.py`)
- [ ] Keep notebook names short but descriptive
- [ ] Create headings as markdown blocks (Command/Control + Shift + M)
- [ ] Hide markdown code of headings (Command/Control + H)

#### Publishing Notebooks

To export your marimo notebooks to HTML locally:

1. Make sure the `export_notebook.sh` script is executable:

   ```bash
   chmod +x export_notebook.sh
   ```

2. Run the script with your notebook name:

   ```bash
   # Either format works:
   ./export_notebook.sh notebooks/notebook_name
   ./export_notebook.sh notebooks/notebook_name.py
   ```

   For example:

   ```bash
   ./export_notebook.sh notebooks/anomaly_detection
   ./export_notebook.sh notebooks/intermittent_forecasting
   ```

The exported HTML files will be automatically deployed to GitHub Pages through the GitHub Actions workflow.

### Pull Request Process

1. Fork the repository
2. Create a new branch for your feature
3. Make your changes
4. Submit a pull request with a clear description of changes
