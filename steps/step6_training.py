import numpy as np
import pandas as pd
import streamlit as st
from typing import Tuple, Dict, Any

from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    r2_score, mean_squared_error, mean_absolute_error
)

from utils.data_utils import prepare_pipeline_data
from utils.model_registry import get_regression_models, get_classification_models

def train_and_compare_models(df: pd.DataFrame, target_col: str, problem_type: str) -> Tuple[Any, pd.DataFrame]:

    """
    Step 6: Preprocesses data, trains all registered models with 5-fold cross-validation,
    evaluates on test set, displays comparative metrics leaderboard, and highlights best model.
    """
    st.subheader("Step 6 — Train & Benchmark Machine Learning Models", anchor=False)

    # Prepare data pipeline
    prep_data = prepare_pipeline_data(df, target_col=target_col, problem_type=problem_type)
    X_train, X_test = prep_data["X_train"], prep_data["X_test"]
    y_train, y_test = prep_data["y_train"], prep_data["y_test"]

    st.session_state["pipeline_prep_data"] = prep_data

    # Retrieve models
    if problem_type == "Classification":
        models_dict = get_classification_models()
        cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scoring = 'f1_weighted'
    else:
        models_dict = get_regression_models()
        cv_strategy = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scoring = 'r2'

    st.markdown(f"Evaluating **{len(models_dict)} machine learning models** using **5-fold Cross-Validation**:")

    results_list = []
    trained_model_objs = {}

    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, (name, model) in enumerate(models_dict.items()):
        status_text.text(f"Training & evaluating {name} ({idx+1}/{len(models_dict)})...")
        try:
            # 5-fold CV on train set
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv_strategy, scoring=cv_scoring)
            cv_mean = np.mean(cv_scores)
            cv_std = np.std(cv_scores)

            # Fit on full training set and predict test set
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            trained_model_objs[name] = model

            if problem_type == "Classification":
                acc = accuracy_score(y_test, preds)
                f1 = f1_score(y_test, preds, average='weighted', zero_division=0)
                prec = precision_score(y_test, preds, average='weighted', zero_division=0)
                rec = recall_score(y_test, preds, average='weighted', zero_division=0)

                results_list.append({
                    "Model Name": name,
                    "Accuracy": round(acc, 4),
                    "F1 Score": round(f1, 4),
                    "Precision": round(prec, 4),
                    "Recall": round(rec, 4),
                    "CV Mean": round(cv_mean, 4),
                    "CV Std": round(cv_std, 4),
                    "Status": "Completed"
                })
            else:
                r2 = r2_score(y_test, preds)
                rmse = np.sqrt(mean_squared_error(y_test, preds))
                mae = mean_absolute_error(y_test, preds)

                results_list.append({
                    "Model Name": name,
                    "R² Score": round(r2, 4),
                    "RMSE": round(rmse, 4),
                    "MAE": round(mae, 4),
                    "CV Mean": round(cv_mean, 4),
                    "CV Std": round(cv_std, 4),
                    "Status": "Completed"
                })

        except Exception as e:
            results_list.append({
                "Model Name": name,
                "Status": f"Failed: {str(e)[:30]}"
            })

        progress_bar.progress((idx + 1) / len(models_dict))

    status_text.empty()
    progress_bar.empty()

    results_df = pd.DataFrame(results_list)

    # Sort results by main primary metric
    sort_key = "F1 Score" if problem_type == "Classification" else "R² Score"
    results_df = results_df.sort_values(by=sort_key, ascending=False).reset_index(drop=True)

    st.markdown("### Model Comparison Leaderboard")
    st.dataframe(results_df, use_container_width=True)

    # Identify best model
    best_model_name = results_df.iloc[0]["Model Name"]
    best_model_obj = trained_model_objs[best_model_name]
    best_score = results_df.iloc[0][sort_key]
    best_cv_mean = results_df.iloc[0]["CV Mean"]

    st.session_state["best_model_name"] = best_model_name
    st.session_state["best_model_obj"] = best_model_obj
    st.session_state["results_df"] = results_df
    st.session_state["y_test_preds"] = best_model_obj.predict(X_test)

    # Green Best Model Banner
    st.markdown(
        f"""
        <div class="banner-success">
            🏆 <strong>Best Model Selected: {best_model_name} ({sort_key} = {best_score:.4f})</strong><br/>
            This model achieved a 5-fold cross-validation average score of <strong>{best_cv_mean:.4f}</strong>, 
            confirming its generalizability across multiple folds of the training data.
        </div>
        """,
        unsafe_allow_html=True
    )

    return best_model_obj, results_df

