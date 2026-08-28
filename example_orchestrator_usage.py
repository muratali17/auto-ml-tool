"""
Example: Using the Orchestrator Function for Automated EDA

This script demonstrates how to use generate_auto_eda_report() to perform
comprehensive exploratory data analysis in just a few lines of code.
"""

import pandas as pd
import numpy as np
from eda_engine import generate_auto_eda_report, print_eda_report_summary


def example_1_simple_usage():
    """Example 1: Basic usage with a simple dataset."""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Simple Binary Classification Dataset")
    print("=" * 70)
    
    # Create sample dataset
    df = pd.DataFrame({
        'age': np.random.normal(35, 10, 200),
        'income': np.random.normal(60000, 20000, 200),
        'credit_score': np.random.normal(700, 100, 200),
        'employment_type': np.random.choice(['Full-time', 'Part-time', 'Self-employed'], 200),
        'approved': np.random.choice([0, 1], 200)
    })
    
    # Generate complete EDA report
    report = generate_auto_eda_report(df, target_col='approved', max_features=3)
    
    # Print summary
    print_eda_report_summary(report)
    
    # Access specific plots
    print("\nAccessing individual plots:")
    print(f"  Univariate plots available: {list(report['univariate_plots'].keys())}")
    print(f"  Bivariate plots available: {list(report['bivariate_plots'].keys())}")
    print(f"  Summary plots available: {list(report['summary_plots'].keys())}")
    
    # You can access individual plots like this:
    if 'age' in report['univariate_plots']:
        age_plot = report['univariate_plots']['age']
        print(f"\n  Age plot title: {age_plot.layout.title.text}")
        # age_plot.show()  # Uncomment to display


def example_2_regression_with_many_features():
    """Example 2: Regression problem with multiple features."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Regression Problem (House Pricing)")
    print("=" * 70)
    
    # Create realistic housing dataset
    n_samples = 300
    df = pd.DataFrame({
        'square_feet': np.random.uniform(800, 5000, n_samples),
        'bedrooms': np.random.choice([1, 2, 3, 4, 5], n_samples),
        'bathrooms': np.random.choice([1, 1.5, 2, 2.5, 3], n_samples),
        'age_years': np.random.uniform(0, 100, n_samples),
        'garage_spaces': np.random.choice([0, 1, 2, 3], n_samples),
        'neighborhood': np.random.choice(['Downtown', 'Suburbs', 'Rural'], n_samples),
        'price': np.random.uniform(200000, 1500000, n_samples)
    })
    
    # Generate report with max_features=4
    report = generate_auto_eda_report(df, target_col='price', max_features=4)
    
    print_eda_report_summary(report)
    
    # Access structure information
    structure = report['structure']
    print(f"\nDataset Structure:")
    print(f"  Problem Type: {structure['problem_type']}")
    print(f"  Numeric Features: {structure['feature_types']['numeric']}")
    print(f"  Categorical Features: {structure['feature_types']['categorical']}")


def example_3_multiclass_with_many_features():
    """Example 3: Multiclass classification with feature limit."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Multiclass Classification (Iris Dataset)")
    print("=" * 70)
    
    # Create iris-like dataset
    n_samples = 150
    df = pd.DataFrame({
        'sepal_length': np.random.uniform(4.3, 7.9, n_samples),
        'sepal_width': np.random.uniform(2.0, 4.4, n_samples),
        'petal_length': np.random.uniform(1.0, 6.9, n_samples),
        'petal_width': np.random.uniform(0.1, 2.5, n_samples),
        'species': np.random.choice(['setosa', 'versicolor', 'virginica'], n_samples)
    })
    
    # Generate report with max_features=2 (limiting to 2 features per category)
    report = generate_auto_eda_report(df, target_col='species', max_features=2)
    
    print_eda_report_summary(report)
    
    # Check for errors during processing
    if report['report_metadata']['errors']:
        print("\nErrors encountered:")
        for error in report['report_metadata']['errors']:
            print(f"  - {error}")


def example_4_with_geographic_data():
    """Example 4: Dataset with geographic features."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Geographic Data Analysis")
    print("=" * 70)
    
    # Create dataset with geographic coordinates
    cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix']
    lats = [40.7128, 34.0522, 41.8781, 29.7604, 33.4484]
    lons = [-74.0060, -118.2437, -87.6298, -95.3698, -112.0742]
    
    n_samples = 100
    df = pd.DataFrame({
        'latitude': np.repeat(lats, n_samples // len(lats)),
        'longitude': np.repeat(lons, n_samples // len(lons)),
        'population': np.random.randint(500000, 8000000, n_samples),
        'crime_rate': np.random.uniform(0, 1000, n_samples),
        'safety_score': np.random.randint(0, 100, n_samples)
    })
    
    # Generate report
    report = generate_auto_eda_report(df, target_col='safety_score', max_features=2)
    
    print_eda_report_summary(report)
    
    # Check if geographic map was generated
    if 'geo_map' in report['summary_plots']:
        print("\n✓ Geographic map was automatically generated!")
        geo_fig = report['summary_plots']['geo_map']
        print(f"  Map title: {geo_fig.layout.title.text}")


def example_5_accessing_report_data():
    """Example 5: Accessing and working with report data."""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Working with Report Data")
    print("=" * 70)
    
    # Create dataset
    df = pd.DataFrame({
        'feature_1': np.random.normal(0, 1, 150),
        'feature_2': np.random.normal(5, 2, 150),
        'feature_3': np.random.choice(['A', 'B', 'C'], 150),
        'target': np.random.choice([0, 1], 150)
    })
    
    # Generate report
    report = generate_auto_eda_report(df, target_col='target', max_features=2)
    
    # Access different parts of the report
    print("\n1. Dataset Structure:")
    structure = report['structure']
    print(f"   Problem Type: {structure['problem_type']}")
    
    print("\n2. Univariate Plots:")
    for col_name, fig in report['univariate_plots'].items():
        print(f"   - {col_name}: {fig.layout.title.text}")
    
    print("\n3. Bivariate Plots (Feature vs Target):")
    for col_name, fig in report['bivariate_plots'].items():
        print(f"   - {col_name}: {fig.layout.title.text}")
    
    print("\n4. Summary/Multivariate Plots:")
    for plot_type, fig in report['summary_plots'].items():
        print(f"   - {plot_type}: {fig.layout.title.text}")
    
    print("\n5. Processing Metadata:")
    metadata = report['report_metadata']
    print(f"   Features Processed: {metadata['features_processed']}")
    print(f"   Features Skipped: {metadata['features_skipped']}")
    print(f"   Total Errors: {len(metadata['errors'])}")


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("AUTOMATED EDA ORCHESTRATOR - USAGE EXAMPLES")
    print("=" * 70)
    
    # Run all examples
    example_1_simple_usage()
    example_2_regression_with_many_features()
    example_3_multiclass_with_many_features()
    example_4_with_geographic_data()
    example_5_accessing_report_data()
    
    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)
    print("\nTo display plots, uncomment the .show() calls in the examples.")
    print("Individual figures can be accessed from the report dictionary:")
    print("  report['univariate_plots']['column_name']")
    print("  report['bivariate_plots']['column_name']")
    print("  report['summary_plots']['plot_type']")
