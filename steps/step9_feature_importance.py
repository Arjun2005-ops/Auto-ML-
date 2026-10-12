from typing import Any
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from sklearn.inspection import permutation_importance

def show_feature_importance(model: Any):
    """Step 6.2 / Step 9: Feature Importance horizontal bar chart using native tree importances or permutation importance."""
    st.subheader("Step 6.2 — Feature Importance Analysis", anchor=False)

    prep_data = st.session_state.get("pipeline_prep_data")
    best_model_obj = st.session_state.get("best_model_obj")
    best_model_name = st.session_state.get("best_model_name", "Model")

    if prep_data is None or best_model_obj is None:
        st.info("Train a model to display feature importance.")
        return

    feature_names = prep_data["feature_names"]
    X_test = prep_data["X_test"]
    y_test = prep_data["y_test"]

    importances = None

    # Check tree/boosting model native feature importances
    if hasattr(best_model_obj, "feature_importances_"):
        importances = best_model_obj.feature_importances_
    # Check linear coefficients
    elif hasattr(best_model_obj, "coef_"):
        importances = np.abs(best_model_obj.coef_).ravel()
        if len(importances) != len(feature_names):
            importances = np.mean(np.abs(best_model_obj.coef_), axis=0)

    # Fallback to Permutation Importance
    if importances is None or len(importances) != len(feature_names):
        with st.spinner("Computing Permutation Feature Importance..."):
            perm_res = permutation_importance(best_model_obj, X_test, y_test, n_repeats=5, random_state=42)
            importances = perm_res.importances_mean

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=True).tail(15)

    fig = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title=f"Top Feature Importances — {best_model_name}",
        color="Importance",
        color_continuous_scale="Purples"
    )
    fig.update_layout(template="plotly_dark", paper_bgcolor="#151922", plot_bgcolor="#151922")
    st.plotly_chart(fig, use_container_width=True)

    st.caption("💡 **Interpretation**: Taller/longer bars indicate features that contribute most heavily to model decisions and predictive performance.")
