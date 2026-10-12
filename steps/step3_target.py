import pandas as pd
import streamlit as st
from typing import Tuple, Optional, List
from utils.data_utils import detect_problem_type

def select_target_and_detect(df: pd.DataFrame, task_type: str) -> Tuple[Optional[str], str]:
    """Step 3: Target column selection and problem type auto-detection (for Supervised tasks), or Feature selection (for Unsupervised)."""
    st.subheader("Step 3 — Target & Feature Selection", anchor=False)

    if task_type in ("Clustering", "Dimensionality Reduction"):
        st.markdown(f"**Unsupervised Learning ({task_type})**: No target column required.")
        numeric_cols = list(df.select_dtypes(include=['number']).columns)
        all_cols = list(df.columns)

        selected_features = st.multiselect(
            "Select feature columns to include in unsupervised modeling:",
            options=all_cols,
            default=numeric_cols if numeric_cols else all_cols[:5],
            help="Numeric columns work best for distance-based clustering and component reduction algorithms."
        )

        if not selected_features:
            st.warning("⚠️ Please select at least 2 feature columns.")
            return None, task_type

        st.session_state["selected_features"] = selected_features
        return None, task_type

    # Supervised Learning (Regression or Classification)
    columns = list(df.columns)

    # Suggest default target column (e.g. 'target', 'churn', 'price', 'label', or last column)
    default_idx = len(columns) - 1
    for i, col in enumerate(columns):
        if col.lower() in ["target", "churn", "price", "label", "outcome", "class", "status", "sales"]:
            default_idx = i
            break

    target_col = st.selectbox(
        "Select the Target Column (variable to predict):",
        options=columns,
        index=default_idx
    )

    detected_type, reasoning = detect_problem_type(df, target_col)

    st.markdown(
        f"""
        <div class="banner-info">
            <strong>🤖 Auto-Detected Problem Type: {detected_type}</strong><br/>
            <span>{reasoning}</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Allow user override if needed
    override_type = st.radio(
        "Confirm or override detected problem type:",
        ["Classification", "Regression"],
        index=0 if detected_type == "Classification" else 1,
        horizontal=True
    )

    st.session_state["target_col"] = target_col
    st.session_state["problem_type"] = override_type

    return target_col, override_type
