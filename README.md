# 🧠 AutoML Studio

> **No-Code Interactive Machine Learning & Analytics Platform**

AutoML Studio is an end-to-end Streamlit web platform that automatically profiles, cleans, visualizes, trains, tunes, and deploys machine learning models on any tabular CSV or Excel (.xlsx / .xls) dataset.

---

## ✨ Features

- **4 Pipeline Objectives**:
  1. **Classification**: Predict discrete categories (Logistic Regression, Decision Trees, Random Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost, KNN, SVM, Naive Bayes).
  2. **Regression**: Predict continuous numerical values (Linear Regression, Ridge, Lasso, Decision Trees, Random Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost, KNN, SVR).
  3. **Clustering**: Group unlabeled data points (KMeans, DBSCAN, Agglomerative Clustering, Gaussian Mixture Models).
  4. **Dimensionality Reduction**: Manifold compression and 2D/3D interactive visual maps (PCA, t-SNE, UMAP).

- **Data Quality & Health Audit**: Missing value counts, column dtype categorization, duplicate row detection, memory footprint KPIs.
- **Preprocessing Pipeline**: Automatic median imputation + StandardScaler for numeric features; mode imputation + OneHotEncoder for categorical features.
- **Exploratory Data Visualizations**: Plotly interactive correlation heatmaps, histogram/boxplot distributions, rare category alerts.
- **Automated Benchmark & 5-Fold Cross-Validation**: Leaderboard comparison sorted by primary metric (F1/R²), highlighting best generalizable model.
- **Hyperparameter Optimization**: RandomizedSearchCV grid search comparing default vs tuned 5-fold CV metrics with relative improvement badges.
- **Real-Time Inference Suite**: Interactive single-instance feature form input and downloadable batch CSV / Excel predictions.
- **xAI Grok AI Insights**: Embedded plain-English analytical recommendations for data quality, model performance, and prediction decisions.

---

## 🛠️ Supported Pipelines & Model Matrix

| Pipeline | Model Families | Primary Evaluation Metrics | Key Visualizations |
| :--- | :--- | :--- | :--- |
| **Classification** | Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost, KNN, SVM, Naive Bayes | F1 Score, Accuracy, Precision, Recall, 5-Fold CV Mean | Confusion Matrix Heatmap, Classification Report, Feature Importance |
| **Regression** | Linear, Ridge, Lasso, Decision Tree, Random Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost, KNN, SVR | R² Score, RMSE, MAE, 5-Fold CV Mean | Feature Importance, Target vs Predicted Scatter |
| **Clustering** | KMeans, DBSCAN, Agglomerative, Gaussian Mixture Model | Silhouette Score, Inertia (Elbow Method) | Elbow Curve, Silhouette Chart, PCA 2D Cluster Scatter, Cluster Profile Means |
| **Dimensionality Reduction** | PCA, t-SNE, UMAP | Explained Variance Ratio, KL-Divergence | 2D/3D Component Scatter Plots, Component Matrix Export |

---

## 🚀 Getting Started

### 1. Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/your-username/automl-studio.git
cd automl-studio
pip install -r requirements.txt
```

### 2. Run Application

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## 📁 Project Structure

```
automl-studio/
├── .streamlit/
│   ├── config.toml            # Dark mode theme & server options
│   └── secrets.toml           # GROK_API_KEY (gitignored)
├── assets/
│   └── fonts_and_css.py       # Google Fonts & modern dark styling
├── sample_data/               # Housing, Churn & Customer Segments demo CSVs
├── steps/
│   ├── step0_task_select.py   # Regression / Classification / Clustering / Dim Red selector
│   ├── step1_overview.py      # Dataset upload & shape preview
│   ├── step2_quality.py       # Missing values & duplicate scan
│   ├── step3_target.py        # Target selection & problem type auto-detection
│   ├── step4_preprocessing.py # Imputation, scaling & encoding plan
│   ├── step5_visualization.py # Plotly correlation heatmaps & histograms
│   ├── step6_training.py      # 5-fold CV model training & leaderboard
│   ├── step7_prediction.py    # Single input form & Batch CSV prediction
│   ├── step8_confusion_matrix.py # Confusion matrix heatmap & report
│   ├── step9_feature_importance.py # Feature importance bar plots
│   ├── step_tuning.py         # RandomizedSearchCV hyperparameter tuning
│   ├── step_clustering.py     # Elbow, Silhouette & cluster profiling
│   ├── step_dim_reduction.py  # PCA / t-SNE / UMAP 2D/3D scatter
│   └── step_ai_insights.py    # Embedded Grok AI insights widget
├── utils/
│   ├── data_utils.py          # Data scanning & preprocessing pipeline builder
│   ├── model_registry.py      # Task-specific model dictionaries & tuning grids
│   └── grok_client.py         # xAI Grok API OpenAI-compatible wrapper
├── app.py                     # Main application orchestrator
├── requirements.txt           # Python dependency requirements
└── README.md
```

---

## 📄 License

MIT License. Built for data science teams and ML enthusiasts.
