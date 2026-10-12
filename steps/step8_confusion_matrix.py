import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.metrics import confusion_matrix, classification_report

def show_confusion_matrix(model: str):
    """Step 6.1 / Step 8: Classification analytics showing Plotly Confusion Matrix and Classification Report."""
    st.subheader("Step 6.1 — Confusion Matrix & Classification Report", anchor=False)

    prep_data = st.session_state.get("pipeline_prep_data")
    best_model_obj = st.session_state.get("best_model_obj")

    if prep_data is None or best_model_obj is None:
        st.info("Train a classification model in Step 6 to inspect confusion matrix.")
        return

    X_test = prep_data["X_test"]
    y_test = prep_data["y_test"]
    target_encoder = prep_data.get("target_encoder")

    preds = best_model_obj.predict(X_test)
    cm = confusion_matrix(y_test, preds)

    # Decode class labels if target_encoder exists
    if target_encoder is not None:
        class_names = [str(c) for c in target_encoder.classes_]
    else:
        unique_labels = np.unique(np.concatenate([y_test, preds]))
        class_names = [f"Class {c}" for c in unique_labels]

    col1, col2 = st.columns([1, 1])

    with col1:
        fig_cm = px.imshow(
            cm,
            x=class_names,
            y=class_names,
            text_auto=True,
            color_continuous_scale="Purples",
            title=f"Confusion Matrix — {st.session_state.get('best_model_name', 'Best Model')}",
            labels=dict(x="Predicted Class", y="Actual True Class")
        )
        fig_cm.update_layout(template="plotly_dark", paper_bgcolor="#151922", plot_bgcolor="#151922")
        st.plotly_chart(fig_cm, use_container_width=True)

    with col2:
        st.markdown("### Per-Class Evaluation Metrics")
        report_dict = classification_report(y_test, preds, target_names=class_names, output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report_dict).transpose().round(4)
        st.dataframe(report_df, use_container_width=True)

    st.caption("💡 **Confusion Matrix Guide**: Diagonal values represent correct predictions (True Positives / True Negatives). Off-diagonal cells indicate misclassifications.")
