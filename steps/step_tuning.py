from typing import Any
import json
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import RandomizedSearchCV, KFold, StratifiedKFold
from utils.model_registry import get_tuning_grid

def show_tuning_section(model: Any):
    """Step 6.3: Hyperparameter tuning using RandomizedSearchCV, displaying search grid, side-by-side metric comparison, and relative improvement badge."""
    st.subheader("Step 6.3 — Automated Hyperparameter Tuning", anchor=False)

    best_model_name = st.session_state.get("best_model_name")
    best_model_obj = st.session_state.get("best_model_obj")
    prep_data = st.session_state.get("pipeline_prep_data")
    problem_type = st.session_state.get("problem_type", "Classification")
    results_df = st.session_state.get("results_df")

    if best_model_obj is None or prep_data is None:
        st.info("Train models in Step 6 first to enable tuning.")
        return

    param_grid = get_tuning_grid(best_model_name)

    if not param_grid:
        st.info(f"Hyperparameter tuning grid is not pre-configured for {best_model_name}.")
        return

    st.markdown(f"Tuning **{best_model_name}** using RandomizedSearchCV across the following hyperparameter space:")

    st.json(param_grid)

    if st.button(f"⚡ Tune Hyperparameters for {best_model_name}", type="primary"):
        X_train = prep_data["X_train"]
        y_train = prep_data["y_train"]

        if problem_type == "Classification":
            cv_strat = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scoring = "f1_weighted"
        else:
            cv_strat = KFold(n_splits=5, shuffle=True, random_state=42)
            scoring = "r2"

        with st.spinner("Executing RandomizedSearchCV across 10 parameter combinations (5-fold CV)..."):
            search = RandomizedSearchCV(
                estimator=best_model_obj,
                param_distributions=param_grid,
                n_iter=10,
                cv=cv_strat,
                scoring=scoring,
                random_state=42,
                n_jobs=-1
            )
            search.fit(X_train, y_train)

        default_cv_score = results_df.iloc[0]["CV Mean"]
        tuned_cv_score = search.best_score_
        pct_improvement = ((tuned_cv_score - default_cv_score) / abs(default_cv_score)) * 100 if default_cv_score != 0 else 0.0

        st.markdown("### Tuning Comparison Results")
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Default 5-Fold CV Score", f"{default_cv_score:.4f}")
        with m2:
            st.metric("Tuned 5-Fold CV Score", f"{tuned_cv_score:.4f}")
        with m3:
            st.metric("Relative Improvement", f"{pct_improvement:+.2f}%", delta=f"{pct_improvement:+.2f}%")

        st.markdown("### Best Hyperparameter Combination Found")
        st.json(search.best_params_)

        # Update best model in session state with tuned instance
        st.session_state["best_model_obj"] = search.best_estimator_
        st.markdown(
            """
            <div class="banner-success">
                ✅ <strong>Model updated with tuned hyperparameters.</strong> Downstream prediction interface will now utilize the optimized estimator.
            </div>
            """,
            unsafe_allow_html=True
        )
