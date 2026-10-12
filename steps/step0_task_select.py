import streamlit as st

def select_task_type() -> str:
    """Step 0: Allows user to choose pipeline task type."""
    st.subheader("Step 0 — Select Pipeline Task Type", anchor=False)

    task_options = [
        "Classification",
        "Regression",
        "Clustering",
        "Dimensionality Reduction"
    ]

    task_descriptions = {
        "Classification": "Predict discrete categories/classes (e.g. Churn vs No Churn, Spam vs Normal). Requires a target column.",
        "Regression": "Predict continuous numeric quantities (e.g. House Prices, Sales Revenue). Requires a target column.",
        "Clustering": "Group similar rows without pre-labeled targets (Unsupervised - KMeans, DBSCAN, GMM). No target column required.",
        "Dimensionality Reduction": "Compress high-dimensional features into 2D/3D visual maps (Unsupervised - PCA, t-SNE, UMAP). No target column required."
    }

    selected_task = st.radio(
        "Choose the machine learning objective for your dataset:",
        task_options,
        index=0,
        horizontal=True,
        key="radio_task_type"
    )

    st.markdown(
        f"""
        <div class="banner-info">
            <strong>🎯 Selected Pipeline: {selected_task}</strong><br/>
            <span style="font-size: 0.9rem; opacity: 0.9;">{task_descriptions[selected_task]}</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.session_state["task_type"] = selected_task
    return selected_task
