import sys

import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx
from streamlit.web.cli import main as streamlit_main

if __name__ == "__main__" and get_script_run_ctx(suppress_warning=True) is None:
    sys.argv = ["streamlit", "run", __file__, *sys.argv[1:]]
    raise SystemExit(streamlit_main())

from assets.fonts_and_css import inject_custom_css

from steps.step0_task_select import select_task_type
from steps.step1_overview import show_overview
from steps.step2_quality import show_data_quality
from steps.step3_target import select_target_and_detect
from steps.step4_preprocessing import show_preprocessing_plan
from steps.step5_visualization import show_visualizations
from steps.step6_training import train_and_compare_models
from steps.step7_prediction import show_prediction_interface
from steps.step8_confusion_matrix import show_confusion_matrix
from steps.step9_feature_importance import show_feature_importance
from steps.step_tuning import show_tuning_section
from steps.step_clustering import run_clustering_pipeline
from steps.step_dim_reduction import run_dim_reduction_pipeline

st.set_page_config(
    page_title="AutoML Studio",
    layout="wide",
    page_icon="🧠",
    initial_sidebar_state="collapsed"
)

# Inject modern dark theme CSS & Google Fonts
inject_custom_css()

# Header Section
col_head1, col_head2 = st.columns([4, 1])
with col_head1:
    st.title("🧠 AutoML Studio")
    st.caption("Upload any tabular CSV or Excel (.xlsx / .xls) dataset — automatically profile it, clean it, visualize it, train & compare ML models, tune parameters, and serve real-time predictions.")


with col_head2:
    st.markdown(
        """
        <div style="text-align: right; padding-top: 10px;">
            <span style="background: rgba(124, 92, 255, 0.15); border: 1px solid #7C5CFF; color: #A78BFA; padding: 6px 12px; border-radius: 20px; font-weight: 600; font-size: 0.8rem;">
                ⚡ AutoML Engine
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )


st.divider()

# Step 0 — Task Selection
with st.container(border=True):
    task_type = select_task_type()

# Step 1 — Overview & Upload
with st.container(border=True):
    df = show_overview()

if df is not None:
    # Step 2 — Data Quality Audit
    with st.container(border=True):
        show_data_quality(df)

    if task_type in ("Classification", "Regression"):
        # Step 3 — Target Selection & Auto-Detection
        with st.container(border=True):
            target_col, problem_type = select_target_and_detect(df, task_type)

        if target_col is not None:
            # Step 4 — Preprocessing Plan
            with st.container(border=True):
                show_preprocessing_plan(df, target_col)

            # Step 5 — Visualizations
            with st.container(border=True):
                show_visualizations(df)

            # Step 6 — Train & Compare Models
            with st.container(border=True):
                best_model, results_df = train_and_compare_models(df, target_col, problem_type)

            # Step 6.1 — Confusion Matrix (Classification only)
            if problem_type == "Classification":
                with st.container(border=True):
                    show_confusion_matrix(best_model)

            # Step 6.2 — Feature Importance
            with st.container(border=True):
                show_feature_importance(best_model)

            # Step 6.3 — Hyperparameter Tuning
            with st.container(border=True):
                show_tuning_section(best_model)

            # Step 7 — Prediction Suite
            with st.container(border=True):
                show_prediction_interface(best_model, df, target_col)

    elif task_type == "Clustering":
        # Step 3 — Feature Selection
        with st.container(border=True):
            select_target_and_detect(df, task_type)

        # Step 4 — Preprocessing Plan
        with st.container(border=True):
            show_preprocessing_plan(df, target_col=None)

        # Step 5 — Visualizations
        with st.container(border=True):
            show_visualizations(df)

        # Step 8 — Clustering Execution
        with st.container(border=True):
            run_clustering_pipeline(df)

    elif task_type == "Dimensionality Reduction":
        # Step 3 — Feature Selection
        with st.container(border=True):
            select_target_and_detect(df, task_type)

        # Step 4 — Preprocessing Plan
        with st.container(border=True):
            show_preprocessing_plan(df, target_col=None)

        # Step 9 — Dim Reduction Execution
        with st.container(border=True):
            run_dim_reduction_pipeline(df)
