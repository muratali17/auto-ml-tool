#!/usr/bin/env python3
"""
Quick Start Guide for Auto-ML EDA Engine
Provides examples of how to use both CLI and web interface
"""

# ============================================================================
# SECTION 1: WEB APPLICATION (RECOMMENDED FOR NON-TECHNICAL USERS)
# ============================================================================

"""
START THE WEB APPLICATION
==========================

1. Open your terminal
2. Navigate to the project directory:
   cd /workspaces/auto-ml-tool

3. Run the application:
   python app.py

4. Open your browser and go to:
   http://localhost:7860

5. You should see:
   - 📤 File upload area (top left)
   - 🎯 Target column dropdown (should say "Select after upload")
   - 🎚️ Max Features slider (default 5)
   - 🚀 Generate EDA Report button (main action)
   - 📊 4 tabs for results (Overview, Summary Plots, Univariate, Bivariate)

QUICK WORKFLOW:
================
1. Click "Upload Dataset" and select a CSV or Excel file
2. Wait for "✅ File loaded successfully"
3. Select your target column from the dropdown
4. Adjust max_features slider if needed (1-15)
5. Click "🚀 Generate EDA Report"
6. Explore results in the 4 tabs

EXPECTED RESULTS:
=================
- Overview Tab: Dataset structure and feature breakdown
- Summary Plots Tab: Correlation heatmap and (optional) geographic map
- Univariate Tab: Individual feature distributions
- Bivariate Tab: Feature relationships with target variable

SUPPORTED FILE FORMATS:
=======================
✓ CSV (.csv)
✓ Excel (.xlsx, .xls)

FILE REQUIREMENTS:
==================
- CSV/Excel with proper headers
- First row should be column names
- At least 2 columns (1 feature + 1 target)
- Numeric and/or categorical columns
"""

# ============================================================================
# SECTION 2: PROGRAMMATIC USAGE (FOR DEVELOPERS)
# ============================================================================

"""
USE THE EDA ENGINE DIRECTLY IN PYTHON
======================================

Example 1: Basic Usage
----------------------
"""

import pandas as pd
from eda_engine import generate_auto_eda_report, print_eda_report_summary

# Load your dataset
df = pd.read_csv('your_data.csv')

# Generate complete EDA report
report = generate_auto_eda_report(
    df=df,
    target_col='target_column_name',
    max_features=5  # Limit features to 5
)

# Print summary to console
print_eda_report_summary(report)

# Access individual results
structure = report['structure']  # Problem type, features
univariate = report['univariate_plots']  # Feature distributions
bivariate = report['bivariate_plots']  # Feature vs target
summary = report['summary_plots']  # Correlations, geo maps
metadata = report['report_metadata']  # Processing info, errors

"""

Example 2: Accessing Specific Plots
------------------------------------
"""

# Get correlation heatmap
if 'correlation_heatmap' in summary:
    fig = summary['correlation_heatmap']
    fig.show()  # Display in Jupyter/browser

# Get univariate plot for specific feature
feature_name = 'age'
if feature_name in univariate:
    fig = univariate[feature_name]
    fig.show()

# Get bivariate plot for specific feature vs target
if feature_name in bivariate:
    fig = bivariate[feature_name]
    fig.show()

"""

Example 3: Analyzing Report Structure
--------------------------------------
"""

# Check problem type
problem_type = report['structure']['problem_type']
print(f"Problem Type: {problem_type}")
# Output: BINARY, MULTICLASS, or REGRESSION

# Get feature categorization
feature_types = report['structure']['feature_types']
print(f"Numeric features: {feature_types['numeric']}")
print(f"Categorical features: {feature_types['categorical']}")
print(f"Datetime features: {feature_types['datetime']}")
print(f"Geographic features: {feature_types['geographic']}")

# Check processing errors
errors = report['report_metadata']['errors']
if errors:
    print("Errors encountered:")
    for error in errors:
        print(f"  - {error}")

"""

Example 4: Custom Analysis Pipeline
------------------------------------
"""

from eda_engine import (
    analyze_dataset_structure,
    plot_univariate_numeric,
    plot_univariate_categorical,
    plot_bivariate_vs_target,
    plot_correlation_heatmap,
    plot_geo_map
)

# 1. Analyze dataset
structure = analyze_dataset_structure(df, target_col='promoted')
problem_type = structure['problem_type']
feature_types = structure['feature_types']

# 2. Plot individual numeric feature
fig = plot_univariate_numeric(df, 'age')
fig.show()

# 3. Plot individual categorical feature
fig = plot_univariate_categorical(df, 'department', top_n=10)
fig.show()

# 4. Plot feature vs target (adaptive based on types)
fig = plot_bivariate_vs_target(
    df=df,
    feature_col='age',
    target_col='promoted',
    problem_type=problem_type,
    feature_type='numeric'  # or 'categorical'
)
fig.show()

# 5. Plot correlation heatmap
numeric_cols = ['age', 'salary', 'performance_score']
fig = plot_correlation_heatmap(df, numeric_cols=numeric_cols)
fig.show()

# 6. Plot geographic distribution (if lat/lon available)
if 'latitude' in df.columns and 'longitude' in df.columns:
    fig = plot_geo_map(df, lat_col='latitude', lon_col='longitude')
    fig.show()

"""

Example 5: Batch Processing Multiple Datasets
---------------------------------------------
"""

import os
from pathlib import Path

# Process all CSV files in a directory
data_dir = Path('data_files')
results = {}

for csv_file in data_dir.glob('*.csv'):
    df = pd.read_csv(csv_file)
    
    # Assume last column is target
    target_col = df.columns[-1]
    
    # Generate report
    report = generate_auto_eda_report(df, target_col=target_col)
    
    # Store results
    results[csv_file.name] = report
    
    print(f"✓ Processed {csv_file.name}")
    print(f"  Problem Type: {report['structure']['problem_type']}")
    print(f"  Features: {len(report['univariate_plots'])}")

"""

# ============================================================================
# SECTION 3: TESTING & VALIDATION
# ============================================================================

"""
RUN THE TEST SUITE
===================

1. Backend Tests (EDA Engine):
   python main_test.py
   
   Expected: 18 tests pass ✓
   Tests: Structure analysis, plotting functions, orchestrator

2. Web App Tests:
   python test_app.py
   
   Expected: 6 tests pass ✓
   Tests: File loading, report generation, plot creation, workflow

3. Individual Function Tests:
   python -m pytest eda_engine.py -v
   
   (If pytest is installed)

"""

# ============================================================================
# SECTION 4: CONFIGURATION & CUSTOMIZATION
# ============================================================================

"""
CUSTOMIZING THE WEB APPLICATION
================================

Edit app.py to customize:

1. UI Appearance:
   - Change title, description
   - Modify colors, fonts in Gradio theme
   - Add custom CSS

2. Feature Limits:
   - Modify max_features slider range
   - Change default value
   - Add additional filters

3. Output Display:
   - Add more tabs
   - Customize output formatting
   - Add export functionality

4. File Upload:
   - Add additional file formats
   - Set file size limits
   - Add data validation

CUSTOMIZING THE EDA ENGINE
===========================

Edit eda_engine.py to:

1. Add Custom Plots:
   - Create new plotting functions
   - Add to generate_auto_eda_report()
   - Follow existing patterns

2. Change Analysis Parameters:
   - Modify plot sizes, colors
   - Adjust correlation threshold
   - Add statistical tests

3. Extend Feature Detection:
   - Add new feature types
   - Create custom categorization
   - Handle special data types

"""

# ============================================================================
# SECTION 5: DEPLOYMENT
# ============================================================================

"""
DEPLOYING THE WEB APPLICATION
==============================

LOCAL DEPLOYMENT (Already Configured):
--------------------------------------
python app.py
# Access at http://localhost:7860

CLOUD DEPLOYMENT:
-----------------

Option 1: Gradio Share (Quick, Temporary)
------------------------------------------
1. Add to app.py:
   if __name__ == "__main__":
       interface = create_interface()
       interface.launch(share=True)  # ← Add share=True

2. Run: python app.py
3. Gradio will generate a public URL (expires in 72 hours)

Option 2: HuggingFace Spaces
-----------------------------
1. Create account at huggingface.co
2. Create new Space
3. Upload all project files
4. Set app.py as entrypoint

Option 3: Docker
----------------
1. Create Dockerfile:
   FROM python:3.9
   WORKDIR /app
   COPY . .
   RUN pip install -r requirements.txt
   CMD ["python", "app.py"]

2. Build and run:
   docker build -t eda-engine .
   docker run -p 7860:7860 eda-engine

Option 4: AWS/GCP/Azure
-----------------------
1. Deploy Docker container to cloud platform
2. Configure environment variables
3. Set public URL/domain
4. Configure CORS if needed

"""

# ============================================================================
# SECTION 6: TROUBLESHOOTING
# ============================================================================

"""
COMMON ISSUES & SOLUTIONS
==========================

Issue 1: Port 7860 Already in Use
----------------------------------
# Kill existing process:
lsof -i :7860
kill -9 <PID>

# Or use different port:
# Edit app.py and change:
interface.launch(server_name="127.0.0.1", server_port=7861)

Issue 2: Module Import Errors
------------------------------
# Reinstall dependencies:
pip install -r requirements.txt --force-reinstall

# Check Python version (needs 3.8+):
python --version

Issue 3: File Upload Not Working
---------------------------------
# Check file permissions:
chmod 755 app.py

# Ensure temp directory exists:
mkdir -p /tmp

Issue 4: Slow Performance
--------------------------
# Reduce max_features slider max value
# Use smaller CSV files for testing
# Upgrade machine resources

Issue 5: Error: "No module named 'eda_engine'"
----------------------------------------------
# Ensure you're in correct directory:
cd /workspaces/auto-ml-tool

# Check file exists:
ls -la eda_engine.py

"""

# ============================================================================
# SECTION 7: QUICK REFERENCE
# ============================================================================

"""
COMMAND QUICK REFERENCE
======================

Start Web App:
  python app.py

Run Tests:
  python main_test.py    # EDA engine tests
  python test_app.py     # Web app tests

Use in Python:
  from eda_engine import generate_auto_eda_report
  report = generate_auto_eda_report(df, 'target')

Check Dependencies:
  pip list | grep -E 'pandas|plotly|gradio'

View Logs:
  # Gradio logs print to console
  python app.py  # Watch output

Get Help:
  python -c "from eda_engine import *; help(generate_auto_eda_report)"

"""

print(__doc__)
