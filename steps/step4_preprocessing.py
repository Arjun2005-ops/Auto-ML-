import pandas as pd
import streamlit as st
from typing import Optional
from utils.data_utils import get_preprocessing_plan_summary

def show_preprocessing_plan(df: pd.DataFrame, target_col: Optional[str] = None):

    """Step 4: Detailed preprocessing plan tables (Imputation, Scaling, Encoding) with explicit fill values."""
    st.subheader("Step 4 — Preprocessing Plan & Feature Pipeline", anchor=False)

    num_df, cat_df = get_preprocessing_plan_summary(df, target_col)

    tab1, tab2 = st.tabs(["🔢 Numeric Features Plan", "🏷️ Categorical Features Plan"])

    with tab1:
        if not num_df.empty:
            st.dataframe(num_df, use_container_width=True)
            st.caption(
                "💡 **Numeric Strategy**: Missing values are imputed using column **medians** (robust against outliers). "
                "All numeric features are subsequently scaled using **StandardScaler** ($Z = (x - \\mu) / \\sigma$) to ensure unit variance across distance metrics and gradient algorithms."
            )
        else:
            st.info("No numeric features found in dataset.")

    with tab2:
        if not cat_df.empty:
            st.dataframe(cat_df, use_container_width=True)
            st.caption(
                "💡 **Categorical Strategy**: Missing values are imputed using the **most frequent category (mode)**. "
                "Features are transformed into binary indicators via **One-Hot Encoding** (with `handle_unknown='ignore'`). "
                "Features with >25 unique categories trigger high-cardinality flags."
            )
        else:
            st.info("No categorical features found in dataset.")

    st.markdown(
        """
        <div class="banner-success">
            ✅ <strong>Preprocessing pipeline built successfully.</strong> Ready for automated model training and evaluation.
        </div>
        """,
        unsafe_allow_html=True
    )

