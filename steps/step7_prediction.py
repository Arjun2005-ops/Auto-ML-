import io
from typing import Any
import pandas as pd
import numpy as np
import streamlit as st
from utils.data_utils import get_column_types, load_uploaded_file

def show_prediction_interface(best_model: Any, df: pd.DataFrame, target_col: str):
    """Step 7: Real-time interactive prediction suite featuring single-instance manual inputs and batch CSV/Excel prediction."""
    st.subheader("Step 7 — Serve Real-Time & Batch Predictions", anchor=False)

    prep_data = st.session_state.get("pipeline_prep_data")
    best_model_obj = st.session_state.get("best_model_obj")
    problem_type = st.session_state.get("problem_type", "Classification")

    if prep_data is None or best_model_obj is None:
        st.info("Train a model first to make predictions.")
        return

    preprocessor = prep_data["preprocessor"]
    target_encoder = prep_data.get("target_encoder")
    feature_df = df.drop(columns=[target_col]) if target_col in df.columns else df.copy()
    num_cols, cat_cols, _ = get_column_types(feature_df)

    tab1, tab2 = st.tabs(["🎛️ Manual Single Input", "📁 Batch File Prediction (CSV / Excel)"])

    with tab1:
        st.markdown("### Enter Feature Values")
        input_data = {}

        # Render 3 columns for input widgets
        cols = st.columns(3)
        all_features = num_cols + cat_cols

        for idx, feature in enumerate(all_features):
            col_target = cols[idx % 3]
            with col_target:
                if feature in num_cols:
                    series = feature_df[feature].dropna()
                    min_val = float(series.min()) if not series.empty else 0.0
                    max_val = float(series.max()) if not series.empty else 100.0
                    median_val = float(series.median()) if not series.empty else 0.0

                    input_data[feature] = st.number_input(
                        f"{feature} (numeric)",
                        value=median_val,
                        min_value=min_val - (abs(min_val) * 2),
                        max_value=max_val + (abs(max_val) * 2),
                        key=f"manual_{feature}"
                    )
                else:
                    unique_cats = list(feature_df[feature].dropna().unique())
                    if not unique_cats:
                        unique_cats = ["Unknown"]
                    input_data[feature] = st.selectbox(
                        f"{feature} (categorical)",
                        options=unique_cats,
                        key=f"manual_{feature}"
                    )

        if st.button("🔮 Predict Single Instance", type="primary"):
            single_df = pd.DataFrame([input_data])
            try:
                processed_single = preprocessor.transform(single_df)
                raw_pred = best_model_obj.predict(processed_single)[0]

                if problem_type == "Classification":
                    if target_encoder is not None:
                        pred_label = target_encoder.inverse_transform([int(raw_pred)])[0]
                    else:
                        pred_label = raw_pred

                    # Probability estimate if available
                    proba_text = ""
                    if hasattr(best_model_obj, "predict_proba"):
                        probas = best_model_obj.predict_proba(processed_single)[0]
                        max_proba = np.max(probas) * 100
                        proba_text = f" (Confidence: {max_proba:.1f}%)"

                    st.markdown(
                        f"""
                        <div class="banner-success" style="font-size: 1.25rem;">
                            🎯 <strong>Predicted Class: {pred_label}</strong>{proba_text}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="banner-success" style="font-size: 1.25rem;">
                            📈 <strong>Predicted Value: {raw_pred:,.4f}</strong>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            except Exception as e:
                st.error(f"Error during prediction transform: {e}")

    with tab2:
        st.markdown("### Upload Batch File (CSV or Excel)")
        batch_file = st.file_uploader("Upload CSV or Excel file containing identical feature schema:", type=["csv", "xlsx", "xls"], key="batch_uploader")

        if batch_file is not None:
            try:
                batch_df = load_uploaded_file(batch_file)
                st.dataframe(batch_df.head(), use_container_width=True)

                if st.button("🚀 Run Batch Prediction", type="primary"):
                    processed_batch = preprocessor.transform(batch_df)
                    raw_preds = best_model_obj.predict(processed_batch)

                    if problem_type == "Classification" and target_encoder is not None:
                        final_preds = target_encoder.inverse_transform(raw_preds.astype(int))
                    else:
                        final_preds = raw_preds

                    output_df = batch_df.copy()
                    output_df[f"Predicted_{target_col}"] = final_preds

                    st.markdown("### Prediction Results Preview")
                    st.dataframe(output_df.head(10), use_container_width=True)

                    dl_col1, dl_col2 = st.columns(2)

                    with dl_col1:
                        csv_bytes = output_df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="📥 Download Predictions CSV",
                            data=csv_bytes,
                            file_name=f"automl_predictions_{target_col}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )

                    with dl_col2:
                        buffer = io.BytesIO()
                        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                            output_df.to_excel(writer, index=False, sheet_name='Predictions')
                        excel_bytes = buffer.getvalue()
                        st.download_button(
                            label="📊 Download Predictions Excel (.xlsx)",
                            data=excel_bytes,
                            file_name=f"automl_predictions_{target_col}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )
            except Exception as e:
                st.error(f"Failed to run batch prediction: {e}")

