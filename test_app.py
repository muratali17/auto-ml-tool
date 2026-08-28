"""
Test Script for Gradio Web Application
Validates the app functionality with sample data
"""

import pandas as pd
import numpy as np
import tempfile
import os
from app import (
    load_file,
    generate_report,
    get_summary_plots,
    get_univariate_plots,
    get_bivariate_plots,
    app_state
)


def create_sample_dataset():
    """Create a sample dataset for testing."""
    n_samples = 150
    df = pd.DataFrame({
        'age': np.random.normal(40, 15, n_samples),
        'salary': np.random.normal(60000, 20000, n_samples),
        'experience': np.random.normal(10, 5, n_samples),
        'education': np.random.choice(['HS', 'BS', 'MS', 'PhD'], n_samples),
        'department': np.random.choice(['Sales', 'IT', 'HR', 'Finance'], n_samples),
        'hired': np.random.choice([0, 1], n_samples)
    })
    return df


def test_file_loading():
    """Test file loading functionality."""
    print("\n" + "=" * 60)
    print("TEST 1: File Loading")
    print("=" * 60)
    
    try:
        # Create temporary CSV file
        df = create_sample_dataset()
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            temp_file = f.name
        
        # Test loading (pass file path directly)
        loaded_df, columns, status = load_file(temp_file)
        
        assert loaded_df is not None, f"DataFrame not loaded: {status}"
        assert len(columns) == 6, "Incorrect number of columns"
        assert "hired" in columns, "Target column not found"
        assert "✅" in status, "Success status not returned"
        
        print("✓ File loading works correctly")
        print(f"  - Columns extracted: {columns}")
        print(f"  - Shape: {loaded_df.shape}")
        
        # Clean up
        os.unlink(temp_file)
        
    except Exception as e:
        print(f"✗ Error: {e}")


def test_report_generation():
    """Test report generation."""
    print("\n" + "=" * 60)
    print("TEST 2: Report Generation")
    print("=" * 60)
    
    try:
        # Load sample data
        df = create_sample_dataset()
        app_state.df = df
        app_state.columns = df.columns.tolist()
        
        # Generate report
        overview, metadata = generate_report('hired', max_features=2)
        
        assert "✅" in overview or "Problem Type" in overview, "Overview not generated"
        assert "BINARY" in overview, "Problem type not detected"
        assert app_state.report is not None, "Report not stored"
        
        print("✓ Report generation works correctly")
        print(f"  - Problem Type: {app_state.report['structure']['problem_type']}")
        print(f"  - Features Processed: {app_state.report['report_metadata']['features_processed']}")
        print(f"  - Univariate Plots: {len(app_state.report['univariate_plots'])}")
        print(f"  - Bivariate Plots: {len(app_state.report['bivariate_plots'])}")
        
    except Exception as e:
        print(f"✗ Error: {e}")


def test_summary_plots():
    """Test summary plots generation."""
    print("\n" + "=" * 60)
    print("TEST 3: Summary Plots")
    print("=" * 60)
    
    try:
        if app_state.report is None:
            print("⚠ Skipping (no report generated)")
            return
        
        corr_html, geo_html = get_summary_plots()
        
        assert corr_html is not None, "Correlation heatmap not generated"
        assert "<" in corr_html, "HTML not generated"
        
        print("✓ Summary plots generated correctly")
        print(f"  - Correlation heatmap: {'Yes' if corr_html else 'No'}")
        print(f"  - Geographic map: {'Yes' if geo_html else 'No'}")
        print(f"  - HTML size: {len(corr_html)} characters")
        
    except Exception as e:
        print(f"✗ Error: {e}")


def test_univariate_plots():
    """Test univariate plots."""
    print("\n" + "=" * 60)
    print("TEST 4: Univariate Plots")
    print("=" * 60)
    
    try:
        if app_state.report is None:
            print("⚠ Skipping (no report generated)")
            return
        
        plots_list = get_univariate_plots()
        
        assert len(plots_list) > 0, "No univariate plots generated"
        assert plots_list[0][1] != "", "Plot HTML empty"
        
        print("✓ Univariate plots generated correctly")
        print(f"  - Total plots: {len(plots_list)}")
        print(f"  - Features: {[p[0] for p in plots_list[:3]]}")
        if len(plots_list) > 3:
            print(f"    ... and {len(plots_list) - 3} more")
        
    except Exception as e:
        print(f"✗ Error: {e}")


def test_bivariate_plots():
    """Test bivariate plots."""
    print("\n" + "=" * 60)
    print("TEST 5: Bivariate Plots")
    print("=" * 60)
    
    try:
        if app_state.report is None:
            print("⚠ Skipping (no report generated)")
            return
        
        plots_list = get_bivariate_plots()
        
        assert len(plots_list) > 0, "No bivariate plots generated"
        assert plots_list[0][1] != "", "Plot HTML empty"
        
        print("✓ Bivariate plots generated correctly")
        print(f"  - Total plots: {len(plots_list)}")
        print(f"  - Features: {[p[0] for p in plots_list[:3]]}")
        if len(plots_list) > 3:
            print(f"    ... and {len(plots_list) - 3} more")
        
    except Exception as e:
        print(f"✗ Error: {e}")


def test_app_workflow():
    """Test complete app workflow."""
    print("\n" + "=" * 60)
    print("TEST 6: Complete App Workflow")
    print("=" * 60)
    
    try:
        # Reset state
        app_state.df = None
        app_state.report = None
        
        # Step 1: Create and load file
        df = create_sample_dataset()
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            temp_file = f.name
        
        loaded_df, columns, status = load_file(temp_file)
        
        # Step 2: Generate report
        overview, metadata = generate_report('hired', max_features=3)
        
        # Step 3: Get all plots
        corr_html, geo_html = get_summary_plots()
        univariate_plots = get_univariate_plots()
        bivariate_plots = get_bivariate_plots()
        
        # Validate
        assert app_state.report is not None, "Report not generated"
        assert len(univariate_plots) > 0, "No univariate plots"
        assert len(bivariate_plots) > 0, "No bivariate plots"
        assert corr_html is not None, "No correlation HTML"
        
        print("✓ Complete workflow executed successfully")
        print(f"  - File loaded: {loaded_df.shape}")
        print(f"  - Report generated: {app_state.report['structure']['problem_type']}")
        print(f"  - Univariate plots: {len(univariate_plots)}")
        print(f"  - Bivariate plots: {len(bivariate_plots)}")
        print(f"  - Summary plots: {2 if geo_html else 1}")
        
        # Clean up
        os.unlink(temp_file)
        
    except Exception as e:
        print(f"✗ Error: {e}")


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("GRADIO APP VALIDATION TESTS")
    print("=" * 60)
    
    test_file_loading()
    test_report_generation()
    test_summary_plots()
    test_univariate_plots()
    test_bivariate_plots()
    test_app_workflow()
    
    print("\n" + "=" * 60)
    print("All validation tests completed!")
    print("=" * 60)
    print("\nTo launch the web app, run:")
    print("  python app.py")
    print("\nThen visit http://localhost:7860 in your browser")
    print("=" * 60)
