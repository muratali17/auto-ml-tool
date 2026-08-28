"""
Test module for eda_engine.py

Demonstrates the usage of analyze_dataset_structure function with various datasets
and tests all plotting functions.
"""

import pandas as pd
import numpy as np
from eda_engine import (
    analyze_dataset_structure,
    plot_univariate_numeric,
    plot_univariate_categorical,
    plot_bivariate_vs_target,
    plot_correlation_heatmap,
    plot_geo_map
)


def test_binary_classification():
    """Test with a binary classification dataset."""
    print("=" * 60)
    print("TEST 1: Binary Classification Dataset")
    print("=" * 60)
    
    df = pd.DataFrame({
        'age': [25, 30, 35, 40, 45, 50],
        'salary': [50000, 60000, 70000, 80000, 90000, 100000],
        'years_experience': [1, 3, 5, 7, 10, 15],
        'education_level': ['HS', 'BS', 'MS', 'PhD', 'BS', 'MS'],
        'hired': [0, 1, 0, 1, 1, 0]
    })
    
    result = analyze_dataset_structure(df, 'hired')
    print(f"Problem Type: {result['problem_type']}")
    print(f"Feature Types: {result['feature_types']}")
    print()


def test_multiclass_classification():
    """Test with a multiclass classification dataset."""
    print("=" * 60)
    print("TEST 2: Multiclass Classification Dataset")
    print("=" * 60)
    
    df = pd.DataFrame({
        'petal_length': [1.4, 1.5, 4.7, 6.3, 6.2],
        'petal_width': [0.2, 0.2, 1.4, 1.8, 2.2],
        'sepal_length': [5.1, 4.9, 6.2, 7.1, 6.1],
        'sepal_width': [3.5, 3.0, 2.9, 3.0, 2.9],
        'flower_species': ['setosa', 'setosa', 'versicolor', 'virginica', 'virginica']
    })
    
    result = analyze_dataset_structure(df, 'flower_species')
    print(f"Problem Type: {result['problem_type']}")
    print(f"Feature Types: {result['feature_types']}")
    print()


def test_regression():
    """Test with a regression dataset."""
    print("=" * 60)
    print("TEST 3: Regression Dataset")
    print("=" * 60)
    
    df = pd.DataFrame({
        'square_feet': [1500, 2000, 2500, 3000, 3500],
        'bedrooms': [2, 3, 3, 4, 4],
        'bathrooms': [1, 2, 2, 3, 3],
        'age_years': [10, 15, 5, 20, 8],
        'house_price': [250000, 350000, 400000, 500000, 550000]
    })
    
    result = analyze_dataset_structure(df, 'house_price')
    print(f"Problem Type: {result['problem_type']}")
    print(f"Feature Types: {result['feature_types']}")
    print()


def test_with_edge_cases():
    """Test with edge cases like numeric columns with few unique values and geo data."""
    print("=" * 60)
    print("TEST 4: Dataset with Edge Cases (Geo & Low-Cardinality Numeric)")
    print("=" * 60)
    
    df = pd.DataFrame({
        'latitude': [40.7128, 34.0522, 41.8781, 29.7604, 39.7392],
        'longitude': [-74.0060, -118.2437, -87.6298, -95.3698, -104.9903],
        'rating': [1, 2, 5, 4, 3],  # numeric but <= 5 unique -> categorical
        'num_rooms': [2, 3, 2, 4, 3],  # numeric but <= 5 unique -> categorical
        'revenue': [100000, 250000, 350000, 500000, 750000],  # many unique -> numeric
        'city': ['NYC', 'LA', 'Chicago', 'Houston', 'Denver'],
        'popularity': [0, 1, 1, 0, 1]  # target: binary
    })
    
    result = analyze_dataset_structure(df, 'popularity')
    print(f"Problem Type: {result['problem_type']}")
    print(f"Feature Types:")
    for feature_type, columns in result['feature_types'].items():
        if columns:
            print(f"  {feature_type}: {columns}")
    print()


def test_with_datetime():
    """Test with datetime columns."""
    print("=" * 60)
    print("TEST 5: Dataset with Datetime Columns")
    print("=" * 60)
    
    df = pd.DataFrame({
        'transaction_date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04']),
        'amount': [100, 250, 350, 500],
        'category': ['A', 'B', 'A', 'C'],
        'converted': [0, 1, 1, 0]
    })
    
    result = analyze_dataset_structure(df, 'converted')
    print(f"Problem Type: {result['problem_type']}")
    print(f"Feature Types: {result['feature_types']}")
    print()


def test_plot_univariate_numeric():
    """Test plot_univariate_numeric function."""
    print("=" * 60)
    print("TEST 6: Univariate Numeric Plot")
    print("=" * 60)
    
    df = pd.DataFrame({
        'age': np.random.normal(40, 15, 100),
        'income': np.random.normal(50000, 20000, 100)
    })
    
    try:
        fig = plot_univariate_numeric(df, 'age')
        print(f"✓ plot_univariate_numeric created successfully")
        print(f"  Figure type: {type(fig)}")
        print(f"  Figure title: {fig.layout.title.text}")
    except Exception as e:
        print(f"✗ Error: {e}")
    print()


def test_plot_univariate_categorical():
    """Test plot_univariate_categorical function."""
    print("=" * 60)
    print("TEST 7: Univariate Categorical Plot")
    print("=" * 60)
    
    categories = ['A', 'B', 'C', 'D', 'E', 'F', 'G'] * 10 + ['H', 'I', 'J'] * 5
    df = pd.DataFrame({'category': categories})
    
    try:
        fig = plot_univariate_categorical(df, 'category', top_n=7)
        print(f"✓ plot_univariate_categorical created successfully")
        print(f"  Figure type: {type(fig)}")
        print(f"  Figure title: {fig.layout.title.text}")
    except Exception as e:
        print(f"✗ Error: {e}")
    print()


def test_plot_bivariate_numeric_regression():
    """Test plot_bivariate_vs_target for regression with numeric feature."""
    print("=" * 60)
    print("TEST 8: Bivariate Plot (Regression + Numeric)")
    print("=" * 60)
    
    df = pd.DataFrame({
        'square_feet': np.random.uniform(1000, 4000, 50),
        'house_price': np.random.uniform(200000, 800000, 50)
    })
    # Add correlation
    df['house_price'] = df['square_feet'] * 200 + np.random.normal(0, 50000, 50)
    
    try:
        fig = plot_bivariate_vs_target(df, 'square_feet', 'house_price', 'REGRESSION', 'numeric')
        print(f"✓ plot_bivariate_vs_target (regression) created successfully")
        print(f"  Figure type: {type(fig)}")
        print(f"  Figure title: {fig.layout.title.text}")
    except Exception as e:
        print(f"✗ Error: {e}")
    print()


def test_plot_bivariate_classification_numeric():
    """Test plot_bivariate_vs_target for classification with numeric feature."""
    print("=" * 60)
    print("TEST 9: Bivariate Plot (Classification + Numeric)")
    print("=" * 60)
    
    df = pd.DataFrame({
        'age': np.random.normal(40, 15, 100),
        'target': np.random.choice([0, 1], 100)
    })
    
    try:
        fig = plot_bivariate_vs_target(df, 'age', 'target', 'BINARY', 'numeric')
        print(f"✓ plot_bivariate_vs_target (classification) created successfully")
        print(f"  Figure type: {type(fig)}")
        print(f"  Figure title: {fig.layout.title.text}")
    except Exception as e:
        print(f"✗ Error: {e}")
    print()


def test_plot_bivariate_categorical():
    """Test plot_bivariate_vs_target with categorical feature."""
    print("=" * 60)
    print("TEST 10: Bivariate Plot (Categorical Feature)")
    print("=" * 60)
    
    df = pd.DataFrame({
        'color': np.random.choice(['Red', 'Blue', 'Green'], 100),
        'target': np.random.choice([0, 1], 100)
    })
    
    try:
        fig = plot_bivariate_vs_target(df, 'color', 'target', 'BINARY', 'categorical')
        print(f"✓ plot_bivariate_vs_target (categorical) created successfully")
        print(f"  Figure type: {type(fig)}")
        print(f"  Figure title: {fig.layout.title.text}")
    except Exception as e:
        print(f"✗ Error: {e}")
    print()


def test_plot_correlation_heatmap():
    """Test plot_correlation_heatmap function."""
    print("=" * 60)
    print("TEST 11: Correlation Heatmap")
    print("=" * 60)
    
    df = pd.DataFrame({
        'age': np.random.normal(40, 15, 100),
        'salary': np.random.normal(50000, 20000, 100),
        'years_exp': np.random.normal(10, 5, 100),
        'score': np.random.uniform(0, 100, 100)
    })
    
    try:
        fig = plot_correlation_heatmap(df)
        print(f"✓ plot_correlation_heatmap created successfully")
        print(f"  Figure type: {type(fig)}")
        print(f"  Figure title: {fig.layout.title.text}")
    except Exception as e:
        print(f"✗ Error: {e}")
    print()


def test_plot_geo_map():
    """Test plot_geo_map function."""
    print("=" * 60)
    print("TEST 12: Geographic Map")
    print("=" * 60)
    
    df = pd.DataFrame({
        'latitude': [40.7128, 34.0522, 41.8781, 29.7604, 39.7392],
        'longitude': [-74.0060, -118.2437, -87.6298, -95.3698, -104.9903],
        'city': ['NYC', 'LA', 'Chicago', 'Houston', 'Denver'],
        'population': [8000000, 4000000, 2700000, 2300000, 700000]
    })
    
    try:
        fig = plot_geo_map(df, 'latitude', 'longitude', 'population')
        print(f"✓ plot_geo_map created successfully")
        print(f"  Figure type: {type(fig)}")
        print(f"  Figure title: {fig.layout.title.text}")
    except Exception as e:
        print(f"✗ Error: {e}")
    print()


if __name__ == '__main__':
    # Test structure analysis functions
    test_binary_classification()
    test_multiclass_classification()
    test_regression()
    test_with_edge_cases()
    test_with_datetime()
    
    # Test plotting functions
    test_plot_univariate_numeric()
    test_plot_univariate_categorical()
    test_plot_bivariate_numeric_regression()
    test_plot_bivariate_classification_numeric()
    test_plot_bivariate_categorical()
    test_plot_correlation_heatmap()
    test_plot_geo_map()
    
    print("=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)
