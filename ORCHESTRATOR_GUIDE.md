## Orchestrator Function Documentation

### `generate_auto_eda_report(df: pd.DataFrame, target_col: str, max_features: int = 5) → dict`

**Purpose**: Complete end-to-end automated EDA generation with a single function call.

#### Parameters
- **`df`** (pd.DataFrame): The input dataset to analyze
- **`target_col`** (str): Name of the target column
- **`max_features`** (int): Maximum number of features to include in univariate/bivariate plots (default: 5)

#### Returns
A structured dictionary containing:

```python
{
    'structure': {
        'problem_type': str,          # 'BINARY', 'MULTICLASS', or 'REGRESSION'
        'feature_types': {
            'numeric': [list of numeric columns],
            'categorical': [list of categorical columns],
            'datetime': [list of datetime columns],
            'geo': [list of geographic columns]
        }
    },
    'univariate_plots': {
        'column_name': plotly.Figure,  # Plots for individual features
        ...
    },
    'bivariate_plots': {
        'column_name': plotly.Figure,  # Feature vs target relationship plots
        ...
    },
    'summary_plots': {
        'correlation_heatmap': plotly.Figure,  # Optional: if numeric features exist
        'geo_map': plotly.Figure               # Optional: if geographic features exist
    },
    'report_metadata': {
        'features_processed': int,     # Number of successful plots
        'features_skipped': int,       # Number of failed plots (gracefully handled)
        'errors': [list of error messages]  # Detailed error information
    }
}
```

#### Workflow
The function performs the following steps:

1. **Dataset Analysis**
   - Calls `analyze_dataset_structure()` to identify problem type and feature categories
   - Extracts numeric, categorical, datetime, and geographic features

2. **Univariate Analysis**
   - Generates plots for top `max_features` numeric features using `plot_univariate_numeric()`
   - Generates plots for top `max_features` categorical features using `plot_univariate_categorical()`

3. **Bivariate Analysis**
   - For each top feature, creates relationship plot with target using `plot_bivariate_vs_target()`
   - Plot type adapts based on problem type:
     - **REGRESSION + numeric**: Scatter plot with OLS trendline
     - **BINARY/MULTICLASS + numeric**: Grouped box plot
     - **categorical**: Grouped/stacked bar chart

4. **Multivariate Analysis**
   - Generates `plot_correlation_heatmap()` if numeric features exist
   - Generates `plot_geo_map()` if geographic features are detected

5. **Error Handling**
   - Gracefully skips any failed plots without crashing
   - Records all errors in metadata for transparency
   - Continues processing remaining features

#### Error Handling
The orchestrator implements robust error handling:
- Individual plot failures don't crash the entire report
- All errors are logged with context information
- Metadata tracks which features succeeded vs. failed
- Failed plots are simply excluded from results

#### Example Usage

```python
from eda_engine import generate_auto_eda_report, print_eda_report_summary
import pandas as pd

# Load your data
df = pd.read_csv('your_data.csv')

# Generate complete EDA report
report = generate_auto_eda_report(df, target_col='target', max_features=5)

# Print summary to console
print_eda_report_summary(report)

# Access individual plots
age_plot = report['univariate_plots']['age']
age_plot.show()

# Access feature-target relationship
age_target_plot = report['bivariate_plots']['age']
age_target_plot.show()

# Display correlation heatmap
correlation_plot = report['summary_plots']['correlation_heatmap']
correlation_plot.show()

# Check for errors
errors = report['report_metadata']['errors']
if errors:
    print(f"Encountered {len(errors)} errors")
    for error in errors:
        print(f"  - {error}")
```

### `print_eda_report_summary(report: Dict[str, Any]) → None`

**Purpose**: Print a human-readable summary of the generated EDA report.

#### Example Output
```
======================================================================
AUTOMATED EDA REPORT SUMMARY
======================================================================

Dataset Structure:
  Problem Type: BINARY
  Feature Types:
    numeric: 3 features
    categorical: 2 features

Plots Generated:
  Univariate Plots: 4
  Bivariate Plots: 4
  Summary Plots: 1

Report Metadata:
  Features Processed: 8
  Features Skipped: 0
  Errors: None

======================================================================
```

## Complete EDA Engine API

### Analysis Functions
- **`analyze_dataset_structure(df, target_col)`** → Analyze dataset and identify problem type
- **`generate_auto_eda_report(df, target_col, max_features)`** → Complete automated EDA
- **`print_eda_report_summary(report)`** → Pretty-print report summary

### Univariate Visualization
- **`plot_univariate_numeric(df, col)`** → Histogram + Box Plot
- **`plot_univariate_categorical(df, col, top_n)`** → Bar chart of top categories

### Bivariate Visualization
- **`plot_bivariate_vs_target(df, feature_col, target_col, problem_type, feature_type)`**
  - Adaptive plots based on feature/problem type

### Multivariate Visualization
- **`plot_correlation_heatmap(df, numeric_cols)`** → Correlation matrix heatmap
- **`plot_geo_map(df, lat_col, lon_col, color_col)`** → Geographic scatter plot

## Features Highlights

✅ **End-to-End Automation**: Single function call for complete EDA
✅ **Adaptive Visualization**: Plot types adjust to data characteristics
✅ **Robust Error Handling**: Fails gracefully on problematic features
✅ **Comprehensive Metadata**: Track what succeeded and what failed
✅ **Geographic Support**: Automatic detection and plotting of location data
✅ **Configurable**: Control feature count via `max_features` parameter
✅ **Datetime Handling**: Supports datetime columns gracefully
✅ **Low-Cardinality Detection**: Numeric columns with ≤5 unique values treated as categorical
✅ **Interactive Plots**: All visualizations are interactive Plotly figures
✅ **Type Hints**: Full type annotations for IDE support

## Dependencies
- pandas
- numpy
- plotly
- scipy
- statsmodels (for OLS trendlines)
- seaborn
- matplotlib

## Installation
```bash
pip install -r requirements.txt
```

## File Structure
```
/workspaces/auto-ml-tool/
├── eda_engine.py                    # Main module with all functions
├── main_test.py                     # Comprehensive test suite (18 tests)
├── example_orchestrator_usage.py    # 5 detailed usage examples
├── requirements.txt                 # Package dependencies
└── README.md
```

## Testing
Run the comprehensive test suite:
```bash
python main_test.py
```

Run usage examples:
```bash
python example_orchestrator_usage.py
```

All 18 tests pass with ✓ success.
