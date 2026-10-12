import numpy as np
from typing import Dict, Any, Tuple

# Scikit-learn models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC
from sklearn.naive_bayes import GaussianNB

# Clustering algorithms
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture

# Dimensionality Reduction algorithms
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Optional advanced models wrapped with safety check
def get_regression_models() -> Dict[str, Any]:
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=0.1),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "K-Nearest Neighbors": KNeighborsRegressor(n_neighbors=5),
        "Support Vector Regressor": SVR(C=1.0),
    }

    try:
        import xgboost as xgb
        models["XGBoost"] = xgb.XGBRegressor(random_state=42, verbosity=0)
    except Exception:
        pass

    try:
        import lightgbm as lgb
        models["LightGBM"] = lgb.LGBMRegressor(random_state=42, verbose=-1)
    except Exception:
        pass

    try:
        import catboost as cb
        models["CatBoost"] = cb.CatBoostRegressor(random_state=42, verbose=0)
    except Exception:
        pass

    return models

def get_classification_models() -> Dict[str, Any]:
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Support Vector Machine": SVC(probability=True, random_state=42),
        "Naive Bayes": GaussianNB(),
    }

    try:
        import xgboost as xgb
        models["XGBoost"] = xgb.XGBClassifier(random_state=42, eval_metric='logloss', verbosity=0)
    except Exception:
        pass

    try:
        import lightgbm as lgb
        models["LightGBM"] = lgb.LGBMClassifier(random_state=42, verbose=-1)
    except Exception:
        pass

    try:
        import catboost as cb
        models["CatBoost"] = cb.CatBoostClassifier(random_state=42, verbose=0)
    except Exception:
        pass

    return models

def get_clustering_models(n_clusters: int = 3) -> Dict[str, Any]:
    return {
        "KMeans": KMeans(n_clusters=n_clusters, random_state=42, n_init=10),
        "DBSCAN": DBSCAN(eps=0.5, min_samples=5),
        "Agglomerative Clustering": AgglomerativeClustering(n_clusters=n_clusters),
        "Gaussian Mixture Model": GaussianMixture(n_components=n_clusters, random_state=42)
    }

def get_dim_reduction_models(n_components: int = 2) -> Dict[str, Any]:
    models = {
        "PCA": PCA(n_components=n_components, random_state=42),
        "t-SNE": TSNE(n_components=n_components, random_state=42, init='pca', learning_rate='auto')
    }

    try:
        import umap
        models["UMAP"] = umap.UMAP(n_components=n_components, random_state=42)
    except Exception:
        pass

    return models

def get_tuning_grid(model_name: str) -> Dict[str, Any]:
    """Returns hyperparameter grid for RandomizedSearchCV based on model family."""
    if "Random Forest" in model_name:
        return {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        }
    elif "Gradient Boosting" in model_name:
        return {
            "n_estimators": [50, 100, 150],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "max_depth": [3, 5, 8],
            "subsample": [0.7, 0.8, 1.0]
        }
    elif "XGBoost" in model_name:
        return {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "max_depth": [3, 6, 9],
            "colsample_bytree": [0.6, 0.8, 1.0]
        }
    elif "LightGBM" in model_name:
        return {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.05, 0.1],
            "num_leaves": [15, 31, 63],
            "max_depth": [-1, 10, 20]
        }
    elif "CatBoost" in model_name:
        return {
            "iterations": [50, 100, 200],
            "learning_rate": [0.01, 0.05, 0.1],
            "depth": [4, 6, 8]
        }
    elif "Logistic" in model_name or "Ridge" in model_name or "Lasso" in model_name:
        return {
            "C": [0.01, 0.1, 1.0, 10.0] if "Logistic" in model_name else [0.01, 0.1, 1.0, 10.0],
        }
    elif "Decision Tree" in model_name:
        return {
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        }
    elif "K-Nearest" in model_name:
        return {
            "n_neighbors": [3, 5, 7, 11, 15],
            "weights": ["uniform", "distance"]
        }
    elif "Support Vector" in model_name or "SVM" in model_name:
        return {
            "C": [0.1, 1.0, 10.0],
            "kernel": ["rbf", "linear"]
        }
    else:
        return {}
