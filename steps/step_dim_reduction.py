import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from utils.data_utils import prepare_pipeline_data, get_column_types

def run_dim_reduction_pipeline(df: pd.DataFrame):

    """Step 9 (Dimensionality Reduction): PCA, t-SNE, UMAP 2D/3D visualization and component CSV export."""
    st.subheader("Step 9 — Dimensionality Reduction & Manifold Projection", anchor=False)

    selected_features = st.session_state.get("selected_features", list(df.columns))
    df_features = df[selected_features].copy()

    # Preprocess
    prep_data = prepare_pipeline_data(df_features, target_col=None, problem_type="Dimensionality Reduction")
    X_processed = prep_data["X_processed"]

    _, cat_cols, _ = get_column_types(df)

    col1, col2, col3 = st.columns(3)
    with col1:
        algo = st.selectbox("Select Reduction Algorithm:", ["PCA", "t-SNE", "UMAP"])
    with col2:
        dims = st.radio("Select Components Count:", [2, 3], horizontal=True)
    with col3:
        color_col = st.selectbox("Color points by column (optional):", ["None"] + list(df.columns))

    color_series = df[color_col] if color_col != "None" else None

    with st.spinner(f"Computing {dims}D {algo} embedding..."):
        if algo == "PCA":
            model = PCA(n_components=dims, random_state=42)
            coords = model.fit_transform(X_processed)
            explained_var = model.explained_variance_ratio_

            st.markdown(f"**Explained Variance Ratio:** `{', '.join([f'PC{i+1}: {v*100:.1f}%' for i, v in enumerate(explained_var)])}` (Total: `{np.sum(explained_var)*100:.1f}%`)")
        elif algo == "t-SNE":
            model = TSNE(n_components=dims, random_state=42, init='pca', learning_rate='auto')
            coords = model.fit_transform(X_processed)
            st.info("💡 **t-SNE Note**: t-SNE optimizes local neighborhood structures via KL-divergence minimization.")
        else:
            try:
                import umap
                model = umap.UMAP(n_components=dims, random_state=42)
                coords = model.fit_transform(X_processed)
                st.info("💡 **UMAP Note**: UMAP preserves both local and global manifold relationships efficiently.")
            except Exception:
                st.warning("`umap-learn` library unavailable. Falling back to PCA.")
                model = PCA(n_components=dims, random_state=42)
                coords = model.fit_transform(X_processed)

    # Build dataframe for Plotly
    comp_names = [f"Component_{i+1}" for i in range(dims)]
    comp_df = pd.DataFrame(coords, columns=comp_names)

    if color_series is not None:
        comp_df[color_col] = color_series.values

    # Render Plotly Scatter
    if dims == 2:
        fig = px.scatter(
            comp_df,
            x=comp_names[0],
            y=comp_names[1],
            color=color_col if color_col != "None" else None,
            title=f"2D {algo} Manifold Projection",
            color_discrete_sequence=px.colors.qualitative.Bold
        )
    else:
        fig = px.scatter_3d(
            comp_df,
            x=comp_names[0],
            y=comp_names[1],
            z=comp_names[2],
            color=color_col if color_col != "None" else None,
            title=f"3D {algo} Manifold Projection",
            color_discrete_sequence=px.colors.qualitative.Bold
        )

    fig.update_layout(template="plotly_dark", paper_bgcolor="#151922", plot_bgcolor="#151922", height=650)
    st.plotly_chart(fig, use_container_width=True)

    # Component CSV Export
    csv_bytes = comp_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"📥 Download Reduced {algo} Components (CSV)",
        data=csv_bytes,
        file_name=f"automl_{algo.lower()}_{dims}d_components.csv",
        mime="text/csv"
    )

