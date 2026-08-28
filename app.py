"""
Gradio Web Application for Automated EDA
Provides an interactive dashboard for exploratory data analysis using eda_engine
"""

import gradio as gr
import pandas as pd
import numpy as np
from eda_engine import (
    generate_auto_eda_report,
    print_eda_report_summary
)
import plotly.graph_objects as go
from typing import Optional, Tuple, List


# Global state to store current report and data
class AppState:
    def __init__(self):
        self.df = None
        self.report = None
        self.columns = []
        self.numeric_columns = []
        self.categorical_columns = []


app_state = AppState()


def load_file(file_obj) -> Tuple[Optional[pd.DataFrame], List[str], str]:
    if file_obj is None:
        return None, [], "No file uploaded"
    
    try:
        file_path = file_obj if isinstance(file_obj, str) else (file_obj.name if hasattr(file_obj, 'name') else str(file_obj))
        
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        else:
            return None, [], "❌ Unsupported file format. Please upload CSV or Excel."
        
        if df.empty:
            return None, [], "❌ Uploaded file is empty"
        
        columns = df.columns.tolist()
        app_state.df = df
        app_state.columns = columns
        app_state.numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        app_state.categorical_columns = df.select_dtypes(exclude=[np.number]).columns.tolist()
        
        status = f"✅ File loaded successfully!\n\n**Dataset Info:**\n- Shape: {df.shape}\n- Columns: {len(columns)}"
        
        return df, columns, status
    
    except Exception as e:
        return None, [], f"❌ Error loading file: {str(e)}"


def generate_report(target_col: str, max_features: int) -> Tuple[str, str]:
    if app_state.df is None:
        return "❌ No file uploaded", ""
    
    if not target_col:
        return "❌ Please select a target column", ""
    
    try:
        app_state.report = generate_auto_eda_report(
            app_state.df,
            target_col=target_col,
            max_features=max_features
        )
        
        structure = app_state.report['structure']
        metadata = app_state.report['report_metadata']
        
        overview_md = f"""
# 📊 EDA Report Overview

## Dataset Structure
- **Problem Type**: `{structure['problem_type']}`
- **Total Rows**: {app_state.df.shape[0]:,}
- **Total Columns**: {app_state.df.shape[1]}
- **Target Column**: `{target_col}`

## Feature Breakdown
"""
        feature_types = structure['feature_types']
        for ftype, columns in feature_types.items():
            if columns:
                overview_md += f"- **{ftype.capitalize()}**: {len(columns)} columns\n"
        
        overview_md += f"""

## Report Generation Stats
- **Univariate Plots**: {len(app_state.report['univariate_plots'])}
- **Bivariate Plots**: {len(app_state.report['bivariate_plots'])}
- **Summary Plots**: {len(app_state.report['summary_plots'])}
- **Features Processed**: {metadata['features_processed']}
- **Features Skipped**: {metadata['features_skipped']}

## Error Summary
"""
        if metadata['errors']:
            overview_md += f"⚠️ **{len(metadata['errors'])} errors encountered:**\n"
            for error in metadata['errors'][:5]:
                overview_md += f"- {error[:100]}...\n"
        else:
            overview_md += "✅ **No errors - report generated successfully!**\n"
        
        detailed_md = f"## Detailed Metadata\n\n### Feature Types Distribution\n"
        for ftype, columns in feature_types.items():
            if columns:
                detailed_md += f"\n**{ftype.upper()}** ({len(columns)}):\n"
                for col in columns[:10]:
                    detailed_md += f"- `{col}`\n"
                if len(columns) > 10:
                    detailed_md += f"- ... and {len(columns) - 10} more\n"
        
        return overview_md, detailed_md
    
    except Exception as e:
        return f"❌ Error generating report: {str(e)}", ""


def get_summary_plots() -> Tuple[Optional[go.Figure], Optional[go.Figure]]:
    if app_state.report is None:
        return None, None
    summary_plots = app_state.report.get('summary_plots', {})
    return summary_plots.get('correlation_heatmap'), summary_plots.get('geo_map')


def create_interface():
    with gr.Blocks(title="Auto-ML EDA Engine") as demo:
        
        gr.HTML("""
        <div style='text-align: center; padding: 20px;'>
            <h1>🔍 Auto-ML EDA Engine</h1>
            <p style='color: #666; font-size: 16px;'>
                Automated Exploratory Data Analysis with Interactive Visualizations
            </p>
        </div>
        """)
        
        with gr.Group():
            gr.Markdown("### 📁 Data Upload & Configuration")
            with gr.Row():
                file_upload = gr.File(
                    label="Upload Dataset (CSV/Excel)",
                    file_types=[".csv", ".xlsx", ".xls"],
                    file_count="single"
                )
                with gr.Column(scale=2):
                    upload_status = gr.Markdown("Upload a CSV or Excel file to get started")
            
            with gr.Row():
                target_dropdown = gr.Dropdown(label="Target Column", choices=[], interactive=True)
                max_features_slider = gr.Slider(minimum=1, maximum=15, value=5, step=1, label="Max Features per Category", interactive=True)
            
            generate_btn = gr.Button("🚀 Generate EDA Report", variant="primary", size="lg")
        
        with gr.Tabs():
            with gr.Tab("📊 Overview & Summary"):
                with gr.Row():
                    overview_output = gr.Markdown("Select a file and target column to generate report")
                    metadata_output = gr.Markdown("")
            
            with gr.Tab("📈 Summary Plots"):
                with gr.Group():
                    gr.Markdown("#### Correlation Heatmap")
                    correlation_output = gr.Plot()
                
                with gr.Group():
                    gr.Markdown("#### Geographic Map (if available)")
                    geo_output = gr.Plot()
            
            with gr.Tab("📉 Univariate Analysis"):
                univariate_selector = gr.Dropdown(label="Select Feature to Display", interactive=True)
                univariate_output = gr.Plot()
            
            with gr.Tab("🔗 Bivariate Analysis (vs Target)"):
                bivariate_selector = gr.Dropdown(label="Select Feature to Display", interactive=True)
                bivariate_output = gr.Plot()
        
        # Callbacks
        def on_file_upload(file_obj):
            df, columns, status = load_file(file_obj)
            return gr.update(choices=columns if columns else []), status
        
        file_upload.change(fn=on_file_upload, inputs=file_upload, outputs=[target_dropdown, upload_status])
        
        def on_generate_report(target_col, max_features):
            if app_state.df is None or not target_col:
                return (
                    "❌ Please upload a file and select a target column", "",
                    None, None,
                    gr.update(choices=[], value=None), None,
                    gr.update(choices=[], value=None), None
                )
            
            overview, metadata = generate_report(target_col, max_features)
            corr_fig, geo_fig = get_summary_plots()
            
            univariate_plots = app_state.report.get('univariate_plots', {})
            univariate_choices = list(univariate_plots.keys())
            first_uni_fig = univariate_plots.get(univariate_choices[0]) if univariate_choices else None
            first_uni_choice = univariate_choices[0] if univariate_choices else None
            
            bivariate_plots = app_state.report.get('bivariate_plots', {})
            bivariate_choices = list(bivariate_plots.keys())
            first_bi_fig = bivariate_plots.get(bivariate_choices[0]) if bivariate_choices else None
            first_bi_choice = bivariate_choices[0] if bivariate_choices else None
            
            return (
                overview, metadata,
                corr_fig, geo_fig,
                gr.update(choices=univariate_choices, value=first_uni_choice), first_uni_fig,
                gr.update(choices=bivariate_choices, value=first_bi_choice), first_bi_fig
            )
        
        generate_btn.click(
            fn=on_generate_report,
            inputs=[target_dropdown, max_features_slider],
            outputs=[
                overview_output, metadata_output,
                correlation_output, geo_output,
                univariate_selector, univariate_output,
                bivariate_selector, bivariate_output
            ]
        )
        
        def update_univariate_display(feature_name):
            if app_state.report is None or not feature_name:
                return None
            return app_state.report.get('univariate_plots', {}).get(feature_name)
        
        univariate_selector.change(fn=update_univariate_display, inputs=univariate_selector, outputs=univariate_output)
        
        def update_bivariate_display(feature_name):
            if app_state.report is None or not feature_name:
                return None
            return app_state.report.get('bivariate_plots', {}).get(feature_name)
        
        bivariate_selector.change(fn=update_bivariate_display, inputs=bivariate_selector, outputs=bivariate_output)
    
    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True, theme=gr.themes.Soft())