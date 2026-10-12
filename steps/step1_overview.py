import os
import pandas as pd
import streamlit as st
from typing import Optional
from utils.data_utils import load_uploaded_file

def show_overview() -> Optional[pd.DataFrame]:
    """Step 1: Dataset upload or sample dataset loading, showing dataset overview."""
    st.subheader("Step 1 — Upload & Preview Dataset", anchor=False)

    col1, col2 = st.columns([3, 2])

    df = None

    with col1:
        uploaded_file = st.file_uploader(
            "Upload your dataset (CSV or Excel format):",
            type=["csv", "xlsx", "xls"],
            help="Select a tabular CSV or Excel (.xlsx, .xls) file up to 200MB."
        )

    with col2:
        st.markdown("**Or load a sample dataset:**")
        sample_choice = st.selectbox(
            "Quick Demo Datasets:",
            ["-- Select Sample Dataset --", "Customer Churn (Classification)", "Housing Prices (Regression)", "Customer Segments (Clustering)"]
        )

    if uploaded_file is not None:
        try:
            df = load_uploaded_file(uploaded_file)
            st.session_state["uploaded_filename"] = uploaded_file.name
        except Exception as e:
            st.error(f"Error reading dataset file: {e}")
            return None
    elif sample_choice != "-- Select Sample Dataset --":
        sample_path_map = {
            "Customer Churn (Classification)": "sample_data/churn_sample.csv",
            "Housing Prices (Regression)": "sample_data/housing_sample.csv",
            "Customer Segments (Clustering)": "sample_data/customer_segments_sample.csv"
        }
        target_path = sample_path_map.get(sample_choice)
        if target_path and os.path.exists(target_path):
            df = load_uploaded_file(target_path)
            st.session_state["uploaded_filename"] = os.path.basename(target_path)

    if df is None:
        st.info("👆 Upload a CSV or Excel file, or pick a sample dataset above to begin the AutoML pipeline.")
        return None


    # Summary KPI Tiles
    n_rows, n_cols = df.shape
    memory_mb = round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("Total Rows", f"{n_rows:,}")
    with kpi2:
        st.metric("Total Columns", f"{n_cols}")
    with kpi3:
        st.metric("Memory Footprint", f"{memory_mb} MB")
    with kpi4:
        st.metric("Numeric Features", len(df.select_dtypes(include=['number']).columns))

    st.markdown("### Dataset Preview (`df.head()`)")
    st.dataframe(df.head(10), use_container_width=True)

    return df
