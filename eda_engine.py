"""
Exploratory Data Analysis Engine Module

This module provides functions to analyze dataset structure and characteristics,
helping identify problem types and feature categorization for automated ML pipelines.
It also includes modular plotting functions for univariate, bivariate, correlation,
and geographic visualizations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats


def analyze_dataset_structure(df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
    """
    Analyze the structure of a DataFrame and identify problem type and feature types.
    
    This function examines the target column to determine if the problem is binary
    classification, multiclass classification, or regression. It also categorizes
    all features (excluding the target) into types: numeric, categorical, datetime,
    and geographic.
    
    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame to analyze.
    target_col : str
        The name of the target column in the DataFrame.
    
    Returns
    -------
    Dict[str, Any]
        A dictionary containing:
        - 'problem_type' (str): One of 'BINARY', 'MULTICLASS', or 'REGRESSION'
        - 'feature_types' (dict): A dictionary with keys 'numeric', 'categorical',
          'datetime', and 'geo', each containing a list of column names belonging
          to that category.
    
    Raises
    ------
    KeyError
        If the target_col is not found in the DataFrame.
    ValueError
        If the DataFrame is empty or target_col is empty.
    
    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'age': [25, 30, 35, 40],
    ...     'salary': [50000, 60000, 70000, 80000],
    ...     'target': [0, 1, 0, 1]
    ... })
    >>> result = analyze_dataset_structure(df, 'target')
    >>> result['problem_type']
    'BINARY'
    >>> 'age' in result['feature_types']['numeric']
    True
    """
    
    # Validate inputs
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame")
    
    target_series = df[target_col]
    
    if target_series.empty:
        raise ValueError(f"Target column '{target_col}' is empty")
    
    # Determine problem type
    problem_type = _determine_problem_type(target_series)
    
    # Get feature columns (all except target)
    feature_columns = [col for col in df.columns if col != target_col]
    
    # Categorize features
    feature_types = _categorize_features(df, feature_columns)
    
    return {
        'problem_type': problem_type,
        'feature_types': feature_types
    }


def _determine_problem_type(target_series: pd.Series) -> str:
    """
    Determine the problem type based on the target column.
    
    Parameters
    ----------
    target_series : pd.Series
        The target column series.
    
    Returns
    -------
    str
        One of 'BINARY', 'MULTICLASS', or 'REGRESSION'.
    """
    # Remove NaN values for analysis
    target_clean = target_series.dropna()
    
    # Check data type
    is_numeric = pd.api.types.is_numeric_dtype(target_clean)
    
    unique_count = target_clean.nunique()
    
    # If numeric with many unique values, treat as regression
    if is_numeric and unique_count > 10:
        return 'REGRESSION'
    
    # If numeric with few unique values, treat as classification
    if is_numeric and unique_count == 2:
        return 'BINARY'
    
    if is_numeric and unique_count > 2:
        return 'MULTICLASS'
    
    # If non-numeric, treat as classification based on unique count
    if unique_count == 2:
        return 'BINARY'
    
    if unique_count > 2:
        return 'MULTICLASS'
    
    # Default case (only 1 unique value)
    return 'BINARY'


def _categorize_features(df: pd.DataFrame, feature_columns: List[str]) -> Dict[str, List[str]]:
    """
    Categorize features into numeric, categorical, datetime, geo, and high_cardinality types.
    
    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    feature_columns : List[str]
        List of feature column names to categorize.
    
    Returns
    -------
    Dict[str, List[str]]
        A dictionary with keys 'numeric', 'categorical', 'datetime', 'geo', 'high_cardinality',
        each containing a list of column names.
    """
    feature_types = {
        'numeric': [],
        'categorical': [],
        'datetime': [],
        'geo': [],
        'high_cardinality': []
    }
    
    geo_keywords = ['latitude', 'longitude', 'lat', 'lon', 'latitude_', 'longitude_']
    
    for col in feature_columns:
        col_lower = col.lower()
        
        # Check for geographic columns
        if any(keyword in col_lower for keyword in geo_keywords):
            feature_types['geo'].append(col)
            continue
        
        # Check for datetime columns
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            feature_types['datetime'].append(col)
            continue
        
        # Check for numeric columns
        if pd.api.types.is_numeric_dtype(df[col]):
            # Edge case: numeric columns with very few unique values (≤5)
            # are treated as categorical
            unique_count = df[col].nunique()
            
            if unique_count <= 5:
                feature_types['categorical'].append(col)
            else:
                feature_types['numeric'].append(col)
            continue
        
        # Check for high cardinality categorical columns (ID, Ticket, Name, etc.)
        # Filters: unique values > 40% of row count AND > 20 unique values
        unique_count = df[col].nunique()
        cardinality_threshold = len(df) * 0.4
        
        if unique_count > cardinality_threshold and unique_count > 20:
            feature_types['high_cardinality'].append(col)
            continue
        
        # Default: treat as categorical (object, string types, etc.)
        feature_types['categorical'].append(col)
    
    return feature_types


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_univariate_numeric(df: pd.DataFrame, col: str) -> go.Figure:
    """
    Create a combined Histogram + BoxPlot visualization for a numeric feature.
    
    This function displays the distribution and outliers of a numeric column
    using a histogram with an overlay boxplot, allowing for quick identification
    of skewness, outliers, and distribution shape.
    
    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    col : str
        The name of the numeric column to visualize.
    
    Returns
    -------
    go.Figure
        A Plotly figure object with histogram and box plot.
    
    Raises
    ------
    KeyError
        If the column is not found in the DataFrame.
    ValueError
        If the column contains no numeric data.
    
    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'age': [25, 30, 35, 40, 45, 50, 100]})
    >>> fig = plot_univariate_numeric(df, 'age')
    >>> fig.show()
    """
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found in DataFrame")
    
    if not pd.api.types.is_numeric_dtype(df[col]):
        raise ValueError(f"Column '{col}' is not numeric")
    
    # Remove NaN values
    data_clean = df[col].dropna()
    
    # Create subplots: histogram and box plot
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(f'Histogram of {col}', f'Box Plot of {col}'),
        specs=[[{"secondary_y": False}], [{"secondary_y": False}]],
        row_heights=[0.7, 0.3]
    )
    
    # Add histogram
    fig.add_trace(
        go.Histogram(
            x=data_clean,
            name='Distribution',
            nbinsx=30,
            opacity=0.7,
            marker_color='steelblue'
        ),
        row=1, col=1
    )
    
    # Add box plot
    fig.add_trace(
        go.Box(
            x=data_clean,
            name='Quartiles',
            marker_color='darkblue'
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_xaxes(title_text=col, row=2, col=1)
    fig.update_yaxes(title_text='Frequency', row=1, col=1)
    fig.update_layout(
        title_text=f'Univariate Analysis: {col}',
        height=600,
        showlegend=True,
        hovermode='x unified'
    )
    
    return fig


def plot_univariate_categorical(df: pd.DataFrame, col: str, top_n: int = 10) -> go.Figure:
    """
    Create a Bar Chart showing the count/frequency of the top N categories.
    
    This function displays a horizontal bar chart of category frequencies,
    useful for understanding the distribution of categorical variables.
    
    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    col : str
        The name of the categorical column to visualize.
    top_n : int, optional
        The number of top categories to display (default=10).
    
    Returns
    -------
    go.Figure
        A Plotly bar chart figure object.
    
    Raises
    ------
    KeyError
        If the column is not found in the DataFrame.
    
    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'city': ['NYC', 'LA', 'NYC', 'Chicago', 'LA', 'LA']})
    >>> fig = plot_univariate_categorical(df, 'city', top_n=5)
    >>> fig.show()
    """
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found in DataFrame")
    
    # Get top N categories by frequency
    value_counts = df[col].value_counts().head(top_n).sort_values()
    
    fig = px.bar(
        x=value_counts.values,
        y=value_counts.index,
        orientation='h',
        labels={'x': 'Count', 'y': col},
        title=f'Top {top_n} Categories: {col}',
        color=value_counts.values,
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        height=400,
        showlegend=False,
        hovermode='y unified'
    )
    
    return fig


def plot_bivariate_vs_target(
    df: pd.DataFrame,
    feature_col: str,
    target_col: str,
    problem_type: str,
    feature_type: str
) -> go.Figure:
    """
    Create a bivariate visualization showing the relationship between a feature and target.
    
    This function creates different plot types based on the problem type and feature type:
    - REGRESSION + numeric: Scatter plot with trend line
    - BINARY/MULTICLASS + numeric: Grouped BoxPlot
    - categorical: Stacked/Grouped Bar Chart
    
    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    feature_col : str
        The name of the feature column.
    target_col : str
        The name of the target column.
    problem_type : str
        One of 'BINARY', 'MULTICLASS', or 'REGRESSION'.
    feature_type : str
        One of 'numeric' or 'categorical'.
    
    Returns
    -------
    go.Figure
        A Plotly figure object appropriate for the data types.
    
    Raises
    ------
    KeyError
        If columns are not found in the DataFrame.
    ValueError
        If problem_type or feature_type are invalid.
    
    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'age': [25, 30, 35, 40],
    ...     'target': [0, 1, 0, 1]
    ... })
    >>> fig = plot_bivariate_vs_target(df, 'age', 'target', 'BINARY', 'numeric')
    >>> fig.show()
    """
    if feature_col not in df.columns:
        raise KeyError(f"Column '{feature_col}' not found in DataFrame")
    if target_col not in df.columns:
        raise KeyError(f"Column '{target_col}' not found in DataFrame")
    
    # Remove NaN values
    df_clean = df[[feature_col, target_col]].dropna()
    
    # Case 1: REGRESSION + numeric -> Scatter plot
    if problem_type == 'REGRESSION' and feature_type == 'numeric':
        fig = px.scatter(
            x=df_clean[feature_col],
            y=df_clean[target_col],
            labels={
                'x': feature_col,
                'y': target_col
            },
            title=f'{feature_col} vs {target_col} (Regression)',
            color_discrete_sequence=['steelblue']
        )
        fig.update_layout(height=500, hovermode='closest')
        return fig
    
    # Case 2: CLASSIFICATION + numeric -> Grouped BoxPlot
    if problem_type in ['BINARY', 'MULTICLASS'] and feature_type == 'numeric':
        fig = px.box(
            x=df_clean[target_col],
            y=df_clean[feature_col],
            labels={
                'x': target_col,
                'y': feature_col
            },
            title=f'{feature_col} Distribution by {target_col}',
            color=df_clean[target_col],
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig.update_layout(height=500, showlegend=False, hovermode='x unified')
        return fig
    
    # Case 3: categorical feature -> Grouped/Stacked Bar Chart
    if feature_type == 'categorical':
        # Create cross-tabulation
        crosstab = pd.crosstab(df_clean[feature_col], df_clean[target_col])
        
        fig = px.bar(
            crosstab,
            x=crosstab.index,
            y=crosstab.columns.tolist(),
            labels={'index': feature_col, 'value': 'Count'},
            title=f'{feature_col} Distribution by {target_col}',
            barmode='group',
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig.update_layout(
            height=500,
            xaxis_title=feature_col,
            yaxis_title='Count',
            hovermode='x unified'
        )
        return fig
    
    raise ValueError(f"Unsupported combination: problem_type={problem_type}, feature_type={feature_type}")


def plot_correlation_heatmap(df: pd.DataFrame, numeric_cols: Optional[List[str]] = None) -> go.Figure:
    """
    Create a Heatmap visualization of Pearson correlations for numeric features.
    
    This function computes the correlation matrix for numeric columns and displays
    it as an interactive heatmap, useful for identifying feature relationships.
    
    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    numeric_cols : List[str], optional
        List of numeric column names to include. If None, all numeric columns
        are used (default=None).
    
    Returns
    -------
    go.Figure
        A Plotly heatmap figure object.
    
    Raises
    ------
    ValueError
        If no numeric columns are found or provided.
    
    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'age': [25, 30, 35],
    ...     'salary': [50000, 60000, 70000],
    ...     'years': [1, 3, 5]
    ... })
    >>> fig = plot_correlation_heatmap(df)
    >>> fig.show()
    """
    # If no columns specified, use all numeric columns
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        raise ValueError("No numeric columns found in DataFrame")
    
    # Compute correlation matrix
    corr_matrix = df[numeric_cols].corr()
    
    # Create heatmap using plotly
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.values.round(2),
        texttemplate='%{text:.2f}',
        textfont={"size": 10},
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title='Correlation Heatmap',
        xaxis_title='Features',
        yaxis_title='Features',
        height=600,
        width=700,
        hovermode='closest'
    )
    
    return fig


def plot_geo_map(
    df: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    color_col: Optional[str] = None
) -> go.Figure:
    """
    Create a Geographic scatter plot visualization for location-based data.
    
    This function creates an interactive scatter plot with latitude and longitude,
    useful for visualizing spatial distributions and patterns in location-based data.
    
    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing latitude and longitude columns.
    lat_col : str
        The name of the latitude column.
    lon_col : str
        The name of the longitude column.
    color_col : str, optional
        The name of a column to use for color coding points. If None, all points
        are the same color (default=None).
    
    Returns
    -------
    go.Figure
        A Plotly scatter figure object.
    
    Raises
    ------
    KeyError
        If latitude or longitude columns are not found.
    ValueError
        If latitude/longitude data is invalid.
    
    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'latitude': [40.7128, 34.0522],
    ...     'longitude': [-74.0060, -118.2437],
    ...     'city': ['NYC', 'LA']
    ... })
    >>> fig = plot_geo_map(df, 'latitude', 'longitude', 'city')
    >>> fig.show()
    """
    if lat_col not in df.columns:
        raise KeyError(f"Column '{lat_col}' not found in DataFrame")
    if lon_col not in df.columns:
        raise KeyError(f"Column '{lon_col}' not found in DataFrame")
    
    # Remove NaN values for lat/lon
    df_clean = df[[lat_col, lon_col]].copy()
    if color_col and color_col in df.columns:
        df_clean[color_col] = df[color_col]
    
    df_clean = df_clean.dropna(subset=[lat_col, lon_col])
    
    if len(df_clean) == 0:
        raise ValueError(f"No valid data found for '{lat_col}' and '{lon_col}'")
    
    # Validate latitude/longitude ranges
    if not (df_clean[lat_col].between(-90, 90).all() and 
            df_clean[lon_col].between(-180, 180).all()):
        raise ValueError("Invalid latitude/longitude values (must be within valid ranges)")
    
    # Create scatter plot
    if color_col and color_col in df_clean.columns:
        fig = px.scatter(
            df_clean,
            x=lon_col,
            y=lat_col,
            color=color_col,
            title=f'Geographic Distribution: {lat_col} vs {lon_col}',
            labels={lat_col: f'{lat_col} (Latitude)', 
                   lon_col: f'{lon_col} (Longitude)'},
            hover_data=[lat_col, lon_col, color_col],
            color_continuous_scale='Viridis'
        )
    else:
        fig = px.scatter(
            df_clean,
            x=lon_col,
            y=lat_col,
            title=f'Geographic Distribution: {lat_col} vs {lon_col}',
            labels={lat_col: f'{lat_col} (Latitude)', 
                   lon_col: f'{lon_col} (Longitude)'},
            hover_data=[lat_col, lon_col],
            color_discrete_sequence=['steelblue']
        )
    
    fig.update_layout(
        height=600,
        hovermode='closest',
        xaxis_title=f'{lon_col} (Longitude)',
        yaxis_title=f'{lat_col} (Latitude)'
    )
    
    return fig


# ============================================================================
# ORCHESTRATOR FUNCTION
# ============================================================================

def generate_auto_eda_report(
    df: pd.DataFrame,
    target_col: str,
    max_features: int = 5
) -> Dict[str, Any]:
    """
    Generate a comprehensive automated EDA report with end-to-end analysis.
    
    This orchestrator function performs full exploratory data analysis by:
    1. Analyzing dataset structure and problem type
    2. Generating univariate plots for top features
    3. Creating bivariate plots showing feature-target relationships
    4. Producing multivariate visualizations (correlation, geo)
    
    All operations include robust error handling to skip failed plots gracefully
    without crashing the entire report generation.
    
    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame to analyze.
    target_col : str
        The name of the target column.
    max_features : int, optional
        Maximum number of features to plot in univariate and bivariate sections
        (default=5). Applies separately to numeric and categorical features.
    
    Returns
    -------
    Dict[str, Any]
        A structured dictionary containing:
        - 'structure' (dict): Output from analyze_dataset_structure()
          with keys: 'problem_type', 'feature_types'
        - 'univariate_plots' (dict): {column_name: plotly_fig} for numeric
          and categorical features
        - 'bivariate_plots' (dict): {column_name: plotly_fig} showing
          feature vs target relationships
        - 'summary_plots' (dict): {plot_type: plotly_fig} containing
          'correlation_heatmap' and optional 'geo_map'
        - 'report_metadata' (dict): Metadata about report generation including
          'features_processed', 'features_skipped', 'errors'
    
    Raises
    ------
    KeyError
        If the target_col is not found in the DataFrame.
    ValueError
        If the DataFrame is empty.
    
    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'age': [25, 30, 35, 40],
    ...     'salary': [50000, 60000, 70000, 80000],
    ...     'target': [0, 1, 0, 1]
    ... })
    >>> report = generate_auto_eda_report(df, 'target', max_features=2)
    >>> print(report['structure']['problem_type'])
    'BINARY'
    >>> fig = report['univariate_plots']['age']
    >>> fig.show()
    """
    
    # Validate inputs
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame")
    
    # Initialize report structure
    report = {
        'structure': {},
        'univariate_plots': {},
        'bivariate_plots': {},
        'summary_plots': {},
        'report_metadata': {
            'features_processed': 0,
            'features_skipped': 0,
            'errors': []
        }
    }
    
    # Step 1: Analyze dataset structure
    try:
        report['structure'] = analyze_dataset_structure(df, target_col)
    except Exception as e:
        report['report_metadata']['errors'].append(
            f"Error analyzing dataset structure: {str(e)}"
        )
        return report
    
    feature_types = report['structure']['feature_types']
    problem_type = report['structure']['problem_type']
    
    # Prepare feature lists (exclude target)
    numeric_features = [f for f in feature_types['numeric'] if f != target_col]
    categorical_features = [f for f in feature_types['categorical'] if f != target_col]
    geo_features = feature_types['geo']
    
    # Limit to max_features
    numeric_features_subset = numeric_features[:max_features]
    categorical_features_subset = categorical_features[:max_features]
    
    # Step 2: Generate univariate plots
    all_feature_subsets = {
        'numeric': numeric_features_subset,
        'categorical': categorical_features_subset
    }
    
    for feature_type, features in all_feature_subsets.items():
        for col in features:
            try:
                if feature_type == 'numeric':
                    fig = plot_univariate_numeric(df, col)
                else:  # categorical
                    fig = plot_univariate_categorical(df, col)
                
                report['univariate_plots'][col] = fig
                report['report_metadata']['features_processed'] += 1
                
            except Exception as e:
                report['report_metadata']['features_skipped'] += 1
                report['report_metadata']['errors'].append(
                    f"Univariate plot for '{col}' ({feature_type}): {str(e)}"
                )
    
    # Step 3: Generate bivariate plots (feature vs target)
    for col in numeric_features_subset:
        try:
            fig = plot_bivariate_vs_target(
                df,
                feature_col=col,
                target_col=target_col,
                problem_type=problem_type,
                feature_type='numeric'
            )
            report['bivariate_plots'][col] = fig
            report['report_metadata']['features_processed'] += 1
            
        except Exception as e:
            report['report_metadata']['features_skipped'] += 1
            report['report_metadata']['errors'].append(
                f"Bivariate plot for '{col}' (numeric vs target): {str(e)}"
            )
    
    for col in categorical_features_subset:
        try:
            fig = plot_bivariate_vs_target(
                df,
                feature_col=col,
                target_col=target_col,
                problem_type=problem_type,
                feature_type='categorical'
            )
            report['bivariate_plots'][col] = fig
            report['report_metadata']['features_processed'] += 1
            
        except Exception as e:
            report['report_metadata']['features_skipped'] += 1
            report['report_metadata']['errors'].append(
                f"Bivariate plot for '{col}' (categorical vs target): {str(e)}"
            )
    
    # Step 4: Generate multivariate plots
    
    # 4a: Correlation heatmap (if numeric features exist)
    if numeric_features_subset:
        try:
            fig = plot_correlation_heatmap(df, numeric_features_subset)
            report['summary_plots']['correlation_heatmap'] = fig
            
        except Exception as e:
            report['report_metadata']['errors'].append(
                f"Correlation heatmap generation: {str(e)}"
            )
    
    # 4b: Geographic map (if geo features exist)
    if len(geo_features) >= 2:
        # Find latitude and longitude columns independently
        lat_col = None
        lon_col = None
        
        for col in geo_features:
            col_lower = col.lower()
            if 'lat' in col_lower and lat_col is None:
                lat_col = col
            if 'lon' in col_lower and lon_col is None:
                lon_col = col
        
        # Only create geo map if both latitude and longitude were found
        if lat_col and lon_col:
            # Try to find a suitable color column (target or first numeric feature)
            color_col = target_col if target_col in df.columns else None
            if not color_col and numeric_features_subset:
                color_col = numeric_features_subset[0]
            
            try:
                fig = plot_geo_map(df, lat_col, lon_col, color_col=color_col)
                report['summary_plots']['geo_map'] = fig
                
            except Exception as e:
                report['report_metadata']['errors'].append(
                    f"Geographic map generation ({lat_col}, {lon_col}): {str(e)}"
                )
    
    return report


def print_eda_report_summary(report: Dict[str, Any]) -> None:
    """
    Print a human-readable summary of the EDA report.
    
    This utility function displays key information about the generated report
    including dataset structure, number of plots generated, and any errors.
    
    Parameters
    ----------
    report : Dict[str, Any]
        The report dictionary returned by generate_auto_eda_report().
    
    Examples
    --------
    >>> report = generate_auto_eda_report(df, 'target')
    >>> print_eda_report_summary(report)
    """
    print("\n" + "=" * 70)
    print("AUTOMATED EDA REPORT SUMMARY")
    print("=" * 70)
    
    # Dataset structure
    structure = report.get('structure', {})
    print(f"\nDataset Structure:")
    print(f"  Problem Type: {structure.get('problem_type', 'Unknown')}")
    
    feature_types = structure.get('feature_types', {})
    print(f"  Feature Types:")
    for ftype, features in feature_types.items():
        if features:
            print(f"    {ftype}: {len(features)} features")
    
    # Plots generated
    print(f"\nPlots Generated:")
    univariate_count = len(report.get('univariate_plots', {}))
    bivariate_count = len(report.get('bivariate_plots', {}))
    summary_count = len(report.get('summary_plots', {}))
    
    print(f"  Univariate Plots: {univariate_count}")
    print(f"  Bivariate Plots: {bivariate_count}")
    print(f"  Summary Plots: {summary_count}")
    
    # Metadata
    metadata = report.get('report_metadata', {})
    print(f"\nReport Metadata:")
    print(f"  Features Processed: {metadata.get('features_processed', 0)}")
    print(f"  Features Skipped: {metadata.get('features_skipped', 0)}")
    
    errors = metadata.get('errors', [])
    if errors:
        print(f"  Errors ({len(errors)}):")
        for error in errors:
            print(f"    - {error}")
    else:
        print(f"  Errors: None")
    
    print("\n" + "=" * 70)
