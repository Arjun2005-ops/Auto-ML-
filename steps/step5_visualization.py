import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from utils.data_utils import get_column_types

def show_visualizations(df: pd.DataFrame):
    """Step 5: Interactive Plotly data visualizations (Correlation heatmap, Distribution histograms/boxplots, Categorical frequency bar charts)."""
    st.subheader("Step 5 — Exploratory Data Visualizations", anchor=False)

    num_cols, cat_cols, _ = get_column_types(df)

    tab1, tab2, tab3 = st.tabs([
        "🔥 Correlation Heatmap", 
        "📊 Distribution & Boxplots", 
        "🏷️ Categorical Frequencies"
    ])

    with tab1:
        if len(num_cols) >= 2:
            corr = df[num_cols].corr()
            fig = px.imshow(
                corr,
                text_auto=".2f",
                color_continuous_scale="Purples",
                title="Numeric Feature Correlation Heatmap (Pearson r)",
                aspect="auto"
            )
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#151922",
                plot_bgcolor="#151922",
                font=dict(family="Inter, sans-serif", color="#E6E6EB")
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("💡 **Correlation Insight**: Strong positive/negative correlations (close to +1.0 or -1.0) indicate potential multicollinearity between features.")
        else:
            st.info("At least 2 numeric features are required to compute a correlation matrix.")

    with tab2:
        if num_cols:
            selected_num = st.selectbox("Select numeric feature to inspect distribution:", options=num_cols)
            col_a, col_b = st.columns(2)

            with col_a:
                fig_hist = px.histogram(
                    df,
                    x=selected_num,
                    nbins=30,
                    title=f"Histogram — {selected_num}",
                    color_discrete_sequence=["#7C5CFF"]
                )
                fig_hist.update_layout(template="plotly_dark", paper_bgcolor="#151922", plot_bgcolor="#151922")
                st.plotly_chart(fig_hist, use_container_width=True)

            with col_b:
                fig_box = px.box(
                    df,
                    y=selected_num,
                    title=f"Boxplot — {selected_num} (Outlier Scan)",
                    color_discrete_sequence=["#22D3B8"]
                )
                fig_box.update_layout(template="plotly_dark", paper_bgcolor="#151922", plot_bgcolor="#151922")
                st.plotly_chart(fig_box, use_container_width=True)

            st.caption("💡 **Distribution & Outliers**: Histograms highlight skewness/bimodality. Boxplot points extending beyond whiskers flag statistical outliers.")
        else:
            st.info("No numeric columns available for distribution analysis.")

    with tab3:
        if cat_cols:
            selected_cat = st.selectbox("Select categorical feature to inspect class frequencies:", options=cat_cols)
            val_counts = df[selected_cat].value_counts().reset_index()
            val_counts.columns = [selected_cat, "Count"]

            fig_bar = px.bar(
                val_counts,
                x=selected_cat,
                y="Count",
                title=f"Category Distribution — {selected_cat}",
                color="Count",
                color_continuous_scale="Viridis"
            )
            fig_bar.update_layout(template="plotly_dark", paper_bgcolor="#151922", plot_bgcolor="#151922")
            st.plotly_chart(fig_bar, use_container_width=True)

            # Rare category alert
            total_rows = len(df)
            rare_cats = val_counts[val_counts["Count"] / total_rows < 0.02][selected_cat].tolist()
            if rare_cats:
                st.markdown(
                    f"""
                    <div class="banner-warning">
                        ⚠️ <strong>Rare categories detected (&lt; 2% frequency):</strong> <code>{', '.join(map(str, rare_cats[:5]))}</code>.
                        Rare levels may increase variance during One-Hot encoding.
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.info("No categorical columns found in dataset.")
