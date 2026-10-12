import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

from utils.data_utils import prepare_pipeline_data

def run_clustering_pipeline(df: pd.DataFrame):

    """Step 8 (Clustering): Unsupervised pipeline with Elbow curve, Silhouette analysis, multi-algorithm execution, and PCA cluster visualization."""
    st.subheader("Step 8 — Unsupervised Clustering & Segmentation", anchor=False)

    selected_features = st.session_state.get("selected_features", list(df.columns))
    df_cluster = df[selected_features].copy()

    # Preprocess feature subset
    prep_data = prepare_pipeline_data(df_cluster, target_col=None, problem_type="Clustering")
    X_processed = prep_data["X_processed"]
    feature_names = prep_data["feature_names"]

    tab1, tab2, tab3 = st.tabs(["📉 Optimal K Analysis (Elbow & Silhouette)", "🧩 Cluster Fitting & Scatter", "📋 Cluster Profiles"])

    # Compute Elbow & Silhouette metrics
    k_range = list(range(2, 9))
    inertias = []
    silhouettes = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_processed)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_processed, labels))

    with tab1:
        c1, c2 = st.columns(2)

        with c1:
            fig_elbow = px.line(
                x=k_range,
                y=inertias,
                markers=True,
                title="Elbow Method — Inertia vs Number of Clusters (K)",
                labels=dict(x="Number of Clusters (K)", y="Inertia (Sum of Squared Distances)")
            )
            fig_elbow.update_layout(template="plotly_dark", paper_bgcolor="#151922", plot_bgcolor="#151922")
            st.plotly_chart(fig_elbow, use_container_width=True)

        with c2:
            fig_sil = px.line(
                x=k_range,
                y=silhouettes,
                markers=True,
                title="Silhouette Score vs Number of Clusters (K)",
                labels=dict(x="Number of Clusters (K)", y="Mean Silhouette Score")
            )
            fig_sil.update_layout(template="plotly_dark", paper_bgcolor="#151922", plot_bgcolor="#151922")
            st.plotly_chart(fig_sil, use_container_width=True)

        best_k_sil = k_range[np.argmax(silhouettes)]
        st.info(f"💡 **Optimal K Recommendation**: Highest silhouette score is achieved at **K = {best_k_sil}** (Score: {max(silhouettes):.4f}).")

    with tab2:
        col_algo, col_k = st.columns(2)
        with col_algo:
            algo_choice = st.selectbox("Select Clustering Algorithm:", ["KMeans", "DBSCAN", "Agglomerative Clustering", "Gaussian Mixture Model"])
        with col_k:
            chosen_k = st.slider("Select Cluster Count (K):", min_value=2, max_value=8, value=int(best_k_sil))

        # Fit chosen model
        if algo_choice == "KMeans":
            model = KMeans(n_clusters=chosen_k, random_state=42, n_init=10)
            cluster_labels = model.fit_predict(X_processed)
        elif algo_choice == "DBSCAN":
            model = DBSCAN(eps=0.5, min_samples=5)
            cluster_labels = model.fit_predict(X_processed)
        elif algo_choice == "Agglomerative Clustering":
            model = AgglomerativeClustering(n_clusters=chosen_k)
            cluster_labels = model.fit_predict(X_processed)
        else:
            model = GaussianMixture(n_components=chosen_k, random_state=42)
            cluster_labels = model.fit_predict(X_processed)

        df_result = df.copy()
        df_result["Cluster"] = [f"Cluster {c}" if c != -1 else "Noise (-1)" for c in cluster_labels]
        st.session_state["cluster_df"] = df_result

        # Reduce to 2D using PCA for scatter plot
        pca = PCA(n_components=2, random_state=42)
        pcs = pca.fit_transform(X_processed)
        pca_df = pd.DataFrame(pcs, columns=["PC1", "PC2"])
        pca_df["Cluster"] = df_result["Cluster"]

        var_explained = np.sum(pca.explained_variance_ratio_) * 100

        fig_scatter = px.scatter(
            pca_df,
            x="PC1",
            y="PC2",
            color="Cluster",
            title=f"PCA 2D Cluster Projection — {algo_choice} (Total Explained Variance: {var_explained:.1f}%)",
            color_discrete_sequence=px.colors.qualitative.Vivid
        )
        fig_scatter.update_layout(template="plotly_dark", paper_bgcolor="#151922", plot_bgcolor="#151922")
        st.plotly_chart(fig_scatter, use_container_width=True)

    with tab3:
        st.markdown("### Cluster Profiles (Feature Means per Cluster)")
        num_cols = list(df_cluster.select_dtypes(include=['number']).columns)
        if num_cols:
            profile_df = df_result.groupby("Cluster")[num_cols].mean().round(3)
            st.dataframe(profile_df, use_container_width=True)
        else:
            st.info("No numeric features available for cluster profile aggregation.")

