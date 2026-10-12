import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_uploaded_file(file_input: Any) -> pd.DataFrame:
    """
    Reads a CSV or Excel (.xlsx, .xls) file or Streamlit UploadedFile object into a pandas DataFrame.
    """
    if isinstance(file_input, str):
        filename = file_input.lower()
    else:
        filename = getattr(file_input, "name", "").lower()

    if filename.endswith((".xlsx", ".xls")):
        return pd.read_excel(file_input)
    else:
        return pd.read_csv(file_input)

def get_column_types(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:

    """Scans DataFrame columns and categorizes into Numeric, Categorical, and Datetime lists."""
    numeric_cols = []
    categorical_cols = []
    datetime_cols = []

    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            datetime_cols.append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    return numeric_cols, categorical_cols, datetime_cols

def get_data_quality_summary(df: pd.DataFrame) -> Tuple[pd.DataFrame, int, float]:
    """Computes missing value stats, duplicate row count, and total memory usage in MB."""
    total_rows = len(df)
    missing_count = df.isnull().sum()
    missing_pct = (missing_count / total_rows) * 100
    dtypes = df.dtypes.astype(str)

    quality_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": dtypes,
        "Missing Values": missing_count,
        "Missing %": missing_pct.round(2),
    }).reset_index(drop=True)

    # Add fill strategy indicator column
    fill_strategies = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            fill_strategies.append("Median Imputation")
        else:
            fill_strategies.append("Mode (Most Frequent)")
    quality_df["Suggested Imputation"] = fill_strategies

    duplicate_rows = int(df.duplicated().sum())
    memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)

    return quality_df, duplicate_rows, round(memory_mb, 2)

def detect_problem_type(df: pd.DataFrame, target_col: str) -> Tuple[str, str]:
    """
    Auto-detects whether a problem is Classification or Regression.
    - If target has a low number of unique values (e.g. <= 20 or ratio <= 0.05) -> Classification
    - Else -> Regression
    """
    if target_col not in df.columns:
        return "Classification", "Target column not found."

    series = df[target_col].dropna()
    unique_count = series.nunique()
    total_rows = len(series)
    ratio = unique_count / total_rows if total_rows > 0 else 0.0

    is_string_or_obj = pd.api.types.is_object_dtype(series) or pd.api.types.is_categorical_dtype(series) or pd.api.types.is_bool_dtype(series)

    if is_string_or_obj or unique_count <= 15 or (unique_count <= 30 and ratio <= 0.05):
        problem_type = "Classification"
        reasoning = f"'{target_col}' has {unique_count} unique value(s), which is treated as a Classification problem (predicting discrete classes/categories)."
    else:
        problem_type = "Regression"
        reasoning = f"'{target_col}' has {unique_count} unique numerical values across {total_rows} rows, which is treated as a Regression problem (predicting a continuous numeric value)."

    return problem_type, reasoning

def get_preprocessing_plan_summary(df: pd.DataFrame, target_col: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generates detailed summary tables showing exact statistics for numeric and categorical columns."""
    feature_df = df.drop(columns=[target_col]) if target_col and target_col in df.columns else df.copy()
    num_cols, cat_cols, _ = get_column_types(feature_df)

    # Numeric summary table
    num_summary_rows = []
    for col in num_cols:
        series = feature_df[col]
        num_summary_rows.append({
            "Column": col,
            "Missing Values": series.isnull().sum(),
            "Fill Value (Median)": round(series.median(), 4) if not series.dropna().empty else "N/A",
            "Mean (Before Scaling)": round(series.mean(), 4) if not series.dropna().empty else "N/A",
            "Std Dev (Before Scaling)": round(series.std(), 4) if not series.dropna().empty else "N/A",
            "Scaling Method": "StandardScaler (Z-score)"
        })
    num_df = pd.DataFrame(num_summary_rows) if num_summary_rows else pd.DataFrame(columns=["Column", "Missing Values", "Fill Value (Median)", "Mean (Before Scaling)", "Std Dev (Before Scaling)", "Scaling Method"])

    # Categorical summary table
    cat_summary_rows = []
    for col in cat_cols:
        series = feature_df[col]
        mode_val = series.mode().iloc[0] if not series.mode().empty else "N/A"
        unique_cnt = series.nunique()
        cat_summary_rows.append({
            "Column": col,
            "Missing Values": series.isnull().sum(),
            "Fill Value (Most Frequent)": str(mode_val),
            "Unique Categories": unique_cnt,
            "Encoding Method": "One-Hot Encoding" if unique_cnt <= 25 else "One-Hot Encoding (High Cardinality Warning)"
        })
    cat_df = pd.DataFrame(cat_summary_rows) if cat_summary_rows else pd.DataFrame(columns=["Column", "Missing Values", "Fill Value (Most Frequent)", "Unique Categories", "Encoding Method"])

    return num_df, cat_df

def prepare_pipeline_data(
    df: pd.DataFrame, 
    target_col: Optional[str] = None, 
    problem_type: str = "Classification", 
    test_size: float = 0.2, 
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Constructs scikit-learn preprocessing pipeline, transforms features, and splits data into train/test sets.
    Returns dictionary with transformed arrays, feature names, preprocessor, and target encoding map if applicable.
    """
    df_clean = df.copy()

    if target_col and target_col in df_clean.columns:
        X = df_clean.drop(columns=[target_col])
        y = df_clean[target_col]

        # Handle missing values in target
        valid_idx = y.notnull()
        X = X[valid_idx]
        y = y[valid_idx]

        # Encode target if classification with string labels
        target_encoder = None
        if problem_type == "Classification" and (pd.api.types.is_object_dtype(y) or pd.api.types.is_categorical_dtype(y)):
            from sklearn.preprocessing import LabelEncoder
            target_encoder = LabelEncoder()
            y = target_encoder.fit_transform(y)
    else:
        X = df_clean
        y = None
        target_encoder = None

    num_cols, cat_cols, _ = get_column_types(X)

    # Construct Pipelines
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]) if num_cols else None

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ]) if cat_cols else None

    transformers = []
    if num_cols:
        transformers.append(('num', num_pipeline, num_cols))
    if cat_cols:
        transformers.append(('cat', cat_pipeline, cat_cols))

    preprocessor = ColumnTransformer(transformers=transformers)
    X_processed = preprocessor.fit_transform(X)

    # Extract feature names after One-Hot Encoding
    feature_names = []
    if num_cols:
        feature_names.extend(num_cols)
    if cat_cols:
        cat_encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
        encoded_cat_names = cat_encoder.get_feature_names_out(cat_cols)
        feature_names.extend(encoded_cat_names)

    # Train / Test split if target is present
    if y is not None:
        stratify_arr = y if (problem_type == "Classification" and len(np.unique(y)) > 1) else None
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y, test_size=test_size, random_state=random_state, stratify=stratify_arr
            )
        except Exception:
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y, test_size=test_size, random_state=random_state
            )
    else:
        X_train, X_test, y_train, y_test = X_processed, X_processed, None, None

    return {
        "X_raw": X,
        "y_raw": y,
        "X_processed": X_processed,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "target_encoder": target_encoder
    }
