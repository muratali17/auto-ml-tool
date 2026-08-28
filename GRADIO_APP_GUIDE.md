## Gradio Web Application - User Guide

### Overview
The Auto-ML EDA Engine Gradio application provides an interactive web interface for performing comprehensive exploratory data analysis on any dataset. Users can upload CSV or Excel files, select a target column, and generate a complete EDA report with visualizations.

### Getting Started

#### Installation
```bash
# Install all dependencies
pip install -r requirements.txt

# Or install just gradio if you have dependencies already
pip install gradio
```

#### Launch the Application
```bash
python app.py
```

The application will start on `http://localhost:7860`

### User Interface Layout

#### 1. Header
- **Title**: "🔍 Auto-ML EDA Engine"
- **Description**: Subtitle explaining the purpose

#### 2. Data Upload & Configuration Section

**File Upload**
- Upload CSV or Excel files (.csv, .xlsx, .xls)
- Status indicator shows file information
- Shows shape and number of columns once loaded

**Target Column Dropdown**
- Dynamically populated after file upload
- Select the column you want to predict/analyze
- Updates automatically when new file is loaded

**Max Features Slider**
- Range: 1-15 (default: 5)
- Controls how many features to include in analysis
- Separate limits for numeric and categorical features

**Generate EDA Report Button**
- Large "🚀 Generate EDA Report" button
- Triggers complete EDA pipeline
- Populates all tabs with results

#### 3. Output Dashboard (4 Tabs)

##### Tab 1: Overview & Summary
Shows:
- **Dataset Structure**
  - Problem Type (BINARY/MULTICLASS/REGRESSION)
  - Total rows and columns
  - Target column name
  
- **Feature Breakdown**
  - Count of numeric features
  - Count of categorical features
  - Count of datetime features (if any)
  - Count of geographic features (if any)

- **Report Generation Stats**
  - Number of univariate plots generated
  - Number of bivariate plots generated
  - Number of summary plots
  - Features successfully processed
  - Features skipped
  - Errors encountered

- **Detailed Feature List**
  - Shows all features in each category
  - Organized by feature type

##### Tab 2: Summary Plots
Contains:
- **Correlation Heatmap**
  - Shows Pearson correlations between numeric features
  - Interactive Plotly chart
  - Color scale from -1 to +1

- **Geographic Map** (optional)
  - Appears if latitude/longitude columns detected
  - Interactive scatter plot with geographic coordinates
  - Optional color coding by target or numeric feature

##### Tab 3: Univariate Analysis
Features:
- **Feature Selector Dropdown**
  - Lists all features with generated univariate plots
  - Select a feature to view its visualization

- **Dynamic Plot Display**
  - **For Numeric Features**: Histogram + Box Plot
    - Shows distribution and outliers
    - Interactively explore ranges and statistics
  
  - **For Categorical Features**: Bar Chart
    - Shows frequency of top categories
    - Color gradient by count

##### Tab 4: Bivariate vs Target
Features:
- **Feature Selector Dropdown**
  - Lists all features with generated bivariate plots
  - Select a feature to view its relationship with target

- **Adaptive Visualization** (based on problem type & feature type):
  - **Regression + Numeric Feature**: Scatter plot with OLS trendline
  - **Classification + Numeric Feature**: Grouped box plots
  - **Categorical Feature**: Grouped/stacked bar chart

### Common Workflows

#### Workflow 1: Quick EDA on New Dataset
1. Click "Upload Dataset" and select your CSV/Excel file
2. Wait for confirmation "✅ File loaded successfully"
3. The Target Column dropdown will auto-populate
4. Select your target column
5. Click "🚀 Generate EDA Report"
6. View results in tabs:
   - Overview → See dataset structure
   - Summary Plots → Check correlations
   - Univariate → Explore individual features
   - Bivariate → See relationships with target

#### Workflow 2: Detailed Feature Analysis
1. Complete steps 1-5 from Workflow 1
2. Go to Tab 3: Univariate Analysis
3. Use the dropdown to explore each feature
4. Investigate distributions, outliers, and data quality
5. Go to Tab 4: Bivariate Analysis
6. Explore how each feature relates to your target

#### Workflow 3: Correlation Analysis
1. Complete report generation (steps 1-5)
2. Go to Tab 2: Summary Plots
3. Examine the correlation heatmap
4. Identify highly correlated features (close to +1 or -1)
5. Consider feature engineering or removing redundant features

#### Workflow 4: Geographic Data Exploration
1. Upload file with 'latitude' and 'longitude' columns
2. Follow standard workflow steps 1-5
3. Go to Tab 2: Summary Plots
4. Scroll to Geographic Map section
5. View spatial distribution of data points
6. Optional: Color-coded by target or numeric feature

### Features & Capabilities

✅ **File Support**: CSV and Excel files
✅ **Automatic Type Detection**: Numeric, categorical, datetime, geographic
✅ **Adaptive Analysis**: Different plots based on data types
✅ **Problem Type Detection**: Auto-detects BINARY, MULTICLASS, or REGRESSION
✅ **Interactive Plots**: All visualizations are interactive Plotly charts
✅ **Robust Error Handling**: Failed plots don't crash the analysis
✅ **Dynamic Dropdowns**: Column selection updates with uploaded file
✅ **Configurable Analysis**: Control feature count via slider

### Understanding the Reports

#### Problem Type Classification
- **BINARY**: Target has exactly 2 unique values
- **MULTICLASS**: Target has >2 unique values
- **REGRESSION**: Numeric target with >10 unique values

#### Feature Categories
- **Numeric**: Continuous or discrete numeric columns
- **Categorical**: String, object, or numeric with ≤5 unique values
- **Datetime**: Datetime-typed columns
- **Geographic**: Columns with 'latitude', 'longitude', 'lat', 'lon' in name

#### Plot Interpretation

**Univariate - Numeric** (Histogram + Box Plot):
- Left: Distribution shape and frequency
- Right: Quartiles, median, and outliers
- Look for skewness, bimodality, or outliers

**Univariate - Categorical** (Bar Chart):
- Height shows frequency of each category
- Color intensity indicates relative frequency
- Identify dominant categories and imbalances

**Bivariate - Regression** (Scatter + Trendline):
- Points show individual data points
- Blue line shows linear relationship
- Steeper line = stronger relationship
- Scattered points = more noise in relationship

**Bivariate - Classification** (Grouped Box Plot):
- Each box represents one target class
- Overlapping boxes suggest similar feature values across classes
- Separated boxes suggest feature discriminates between classes

**Bivariate - Categorical** (Grouped Bar Chart):
- Groups show different target classes
- Height differences indicate imbalance across categories
- Color separation shows target distribution

**Correlation Heatmap**:
- Red (positive 1.0): Perfect positive correlation
- White (0.0): No correlation
- Blue (negative -1.0): Perfect negative correlation
- Look for correlations >0.8 or <-0.8 as potentially problematic

### Tips & Best Practices

1. **Start with Overview Tab**
   - Understand your dataset structure first
   - Check the problem type classification
   - Review feature breakdown

2. **Check for Imbalances**
   - In categorical features, look for dominant categories
   - In target variable, check class balance
   - May require resampling or weighted models

3. **Identify Outliers**
   - Use univariate box plots to spot outliers
   - Decide whether to remove, cap, or transform
   - Check if outliers make sense for your domain

4. **Explore Relationships**
   - Bivariate plots show feature-target relationships
   - Stronger relationships = more predictive power
   - Weak relationships may need feature engineering

5. **Review Correlations**
   - High correlation between features may indicate redundancy
   - Consider removing one of correlated features
   - Geographic coordinates often highly correlated

6. **Use Appropriate max_features**
   - Start with default (5) for quick overview
   - Increase for datasets with many features
   - Decrease for cleaner, focused analysis

### Troubleshooting

**Problem**: "No file uploaded" message
- **Solution**: Click file upload and select a CSV or Excel file

**Problem**: Target column dropdown is empty
- **Solution**: First upload a file, dropdown auto-populates after success

**Problem**: "Error generating report"
- **Possible Causes**:
  - Target column has only 1 unique value
  - Dataset has no numeric or categorical columns
  - File is corrupted
- **Solution**: Check data quality, ensure valid target column

**Problem**: Geographic map not appearing
- **Solution**: File must have 'latitude' and 'longitude' columns

**Problem**: Some plots missing in univariate/bivariate tabs
- **Solution**: These features had errors during processing. Check "Overview" tab for error details

### Performance Considerations

- **Small datasets** (<10K rows): Near-instant results
- **Medium datasets** (10K-100K rows): 2-5 seconds
- **Large datasets** (>100K rows): 10-30 seconds
- Report generation is CPU-bound; performance depends on your machine

### Advanced Usage

#### Accessing Raw Plots Programmatically
```python
from app import app_state, generate_report

# Generate report
generate_report('target_col', max_features=5)

# Access individual plots
univariate_plots = app_state.report['univariate_plots']
bivariate_plots = app_state.report['bivariate_plots']
summary_plots = app_state.report['summary_plots']

# Each plot is a Plotly Figure object
fig = univariate_plots['column_name']
fig.show()
```

#### Customizing the Interface
- Edit `app.py` to modify colors, themes, or layout
- Gradio supports custom CSS for styling
- Add additional tabs or sections as needed

### Known Limitations

- Excel files may be slower to load than CSV
- Very wide datasets (>100 columns) may take longer
- Geographic features require specific naming conventions
- Maximum file upload size depends on server configuration

### Support & Feedback

For issues or feature requests:
1. Check the error messages in the "Overview" tab
2. Verify your data format and content
3. Try with a sample dataset first
4. Review the eda_engine.py documentation for technical details

---

**Status**: Production Ready | **Version**: 1.0 | **Last Updated**: 2026-08-28
