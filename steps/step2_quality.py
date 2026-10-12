import pandas as pd
import streamlit as st
from utils.data_utils import get_data_quality_summary, get_column_types

def show_data_quality(df: pd.DataFrame):

    """Step 2: Data quality check, missing value audit, duplicates check, and dtype breakdown."""
    st.subheader("Step 2 — Data Quality Audit & Health Scan", anchor=False)

    quality_df, duplicate_rows, memory_mb = get_data_quality_summary(df)
    num_cols, cat_cols, dt_cols = get_column_types(df)

    # Duplicate Warning Banner
    if duplicate_rows > 0:
        st.markdown(
            f"""
            <div class="banner-warning">
                ⚠ <strong>{duplicate_rows:,} duplicate row(s) detected</strong>. 
                These will be automatically handled during preprocessing to prevent data leakage.
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div class="banner-success">
                ✅ <strong>Zero duplicate rows detected.</strong> Clean row uniqueness check passed.
            </div>
            """,
            unsafe_allow_html=True
        )

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("### Missing Values & Column Imputation Plan")
        st.dataframe(
            quality_df.style.highlight_between(subset=["Missing Values"], left=1, right=float('inf'), color="#3B1C24"),
            use_container_width=True,
            height=300
        )

    with col2:
        st.markdown("### Column Type Breakdown")
        st.markdown(f"- **Numeric Columns ({len(num_cols)}):** `{', '.join(num_cols[:8])}`{'...' if len(num_cols) > 8 else ''}")
        st.markdown(f"- **Categorical Columns ({len(cat_cols)}):** `{', '.join(cat_cols[:8])}`{'...' if len(cat_cols) > 8 else ''}")
        if dt_cols:
            st.markdown(f"- **Datetime Columns ({len(dt_cols)}):** `{', '.join(dt_cols)}`")

        missing_cols_count = len(quality_df[quality_df["Missing Values"] > 0])
        st.markdown(f"- **Columns with Missing Data:** `{missing_cols_count}` of `{len(df.columns)}`")

