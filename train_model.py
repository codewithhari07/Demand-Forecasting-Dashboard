from __future__ import annotations

import gc
import pickle
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from project_utils import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    add_lag_features,
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "sales_data.csv"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
MODEL_PATH = MODEL_DIR / "best_demand_model.pkl"
CLASSIFIER_PATH = MODEL_DIR / "high_demand_classifier.pkl"


def make_pipeline(model, scale_numeric: bool = False) -> Pipeline:
    """Build preprocessing + estimator so the complete pipeline can be pickled."""
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(steps=numeric_steps)

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def main() -> None:
    start = time.time()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Put your CSV in data/sales_data.csv."
        )

    df = pd.read_csv(DATA_PATH)
    required = {
        "Date", "Store ID", "Product ID", "Category", "Region",
        "Inventory Level", "Units Sold", "Units Ordered", "Price", "Discount",
        "Weather Condition", "Promotion", "Competitor Pricing", "Seasonality",
        "Epidemic", "Demand",
    }
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"CSV is missing required columns: {sorted(missing_cols)}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", TARGET_COLUMN]).copy()
    df = df.drop_duplicates(subset=["Date", "Store ID", "Product ID"])
    # Keep category types consistent between training and future dashboard inputs.
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype(str)
    df = df.sort_values(["Store ID", "Product ID", "Date"]).reset_index(drop=True)

    print(f"Loaded data: {len(df):,} rows and {len(df.columns)} columns")
    print(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    print(f"Missing values before feature engineering: {int(df.isna().sum().sum())}")

    # Only past Demand values become lag features. Same-day Units Sold, inventory,
    # and orders are intentionally excluded to reduce target leakage.
    featured = add_lag_features(df)
    featured = featured.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN]).copy()
    featured = featured.sort_values("Date").reset_index(drop=True)

    unique_dates = np.sort(featured["Date"].unique())
    if len(unique_dates) < 10:
        raise ValueError("Not enough unique dates to make a time-based split.")

    split_index = int(len(unique_dates) * 0.80)
    split_index = min(max(split_index, 1), len(unique_dates) - 1)
    cutoff_date = pd.Timestamp(unique_dates[split_index])
    train_df = featured[featured["Date"] < cutoff_date].copy()
    test_df = featured[featured["Date"] >= cutoff_date].copy()

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN].astype(float)
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN].astype(float)

    print(f"Training rows: {len(train_df):,}")
    print(f"Validation rows: {len(test_df):,}")
    print(f"Validation starts: {cutoff_date.date()}")
    print("\nRegression: comparing LightGBM and Random Forest by MAE...\n")

    # Two regressors forecast numeric demand. The lower-MAE model becomes the
    # saved forecasting model.
    regression_models = [
        (
            "LightGBM Regressor",
            lambda: lgb.LGBMRegressor(
                objective="regression",
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=31,
                max_depth=-1,
                min_child_samples=30,
                subsample=0.85,
                subsample_freq=1,
                colsample_bytree=0.85,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=2,
                verbosity=-1,
            ),
        ),
        (
            "Random Forest Regressor",
            lambda: RandomForestRegressor(
                n_estimators=60,
                max_depth=16,
                min_samples_leaf=3,
                max_features=0.9,
                random_state=42,
                n_jobs=2,
            ),
        ),
    ]

    regression_results = []
    best_regression_pipeline = None
    best_regression_name = None
    best_mae = float("inf")
    best_validation_predictions = None

    for name, factory in regression_models:
        pipeline = make_pipeline(factory())
        model_start = time.time()
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        mae = float(mean_absolute_error(y_test, predictions))
        rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
        r2 = float(r2_score(y_test, predictions))
        regression_results.append(
            {
                "model": name,
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
                "validation_rows": int(len(test_df)),
            }
        )
        print(
            f"{name}: MAE={mae:.3f} | RMSE={rmse:.3f} | R2={r2:.4f} "
            f"| fit+predict={time.time() - model_start:.1f}s"
        )

        if mae < best_mae:
            if best_regression_pipeline is not None:
                del best_regression_pipeline
                gc.collect()
            best_mae = mae
            best_regression_name = name
            best_regression_pipeline = pipeline
            best_validation_predictions = predictions.copy()
        else:
            del pipeline
        del predictions
        gc.collect()

    regression_results = sorted(regression_results, key=lambda row: row["MAE"])

    # Logistic Regression is a classifier, not a numeric demand regressor. Here
    # it predicts whether demand is in the high-demand group (>= training 75th pct).
    high_demand_threshold = float(y_train.quantile(0.75))
    y_train_high = (y_train >= high_demand_threshold).astype(int)
    y_test_high = (y_test >= high_demand_threshold).astype(int)

    print(
        "\nClassification: training Logistic Regression for high demand "
        f"(Demand >= {high_demand_threshold:.2f}, the training 75th percentile)...\n"
    )
    classifier_pipeline = make_pipeline(
        LogisticRegression(
            max_iter=1200,
            class_weight="balanced",
            solver="lbfgs",
            random_state=42,
        ),
        scale_numeric=True,
    )
    classifier_start = time.time()
    classifier_pipeline.fit(X_train, y_train_high)
    class_predictions = classifier_pipeline.predict(X_test)
    class_probabilities = classifier_pipeline.predict_proba(X_test)[:, 1]

    classifier_metrics = {
        "model": "Logistic Regression Classifier",
        "positive_class_definition": f"Demand >= {high_demand_threshold:.2f}",
        "high_demand_threshold": high_demand_threshold,
        "accuracy": float(accuracy_score(y_test_high, class_predictions)),
        "precision": float(precision_score(y_test_high, class_predictions, zero_division=0)),
        "recall": float(recall_score(y_test_high, class_predictions, zero_division=0)),
        "f1": float(f1_score(y_test_high, class_predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test_high, class_probabilities)),
        "validation_rows": int(len(test_df)),
    }
    print(
        "Logistic Regression: "
        f"Accuracy={classifier_metrics['accuracy']:.3f} | "
        f"Precision={classifier_metrics['precision']:.3f} | "
        f"Recall={classifier_metrics['recall']:.3f} | "
        f"F1={classifier_metrics['f1']:.3f} | "
        f"ROC-AUC={classifier_metrics['roc_auc']:.3f} "
        f"| fit+predict={time.time() - classifier_start:.1f}s"
    )

    # Preserve holdout scores, then refit selected models on all eligible data.
    final_regressor = clone(best_regression_pipeline.named_steps["model"])
    del best_regression_pipeline
    gc.collect()
    final_regression_pipeline = make_pipeline(final_regressor)
    final_regression_pipeline.fit(featured[FEATURE_COLUMNS], featured[TARGET_COLUMN].astype(float))

    final_classifier = make_pipeline(
        LogisticRegression(
            max_iter=1200,
            class_weight="balanced",
            solver="lbfgs",
            random_state=42,
        ),
        scale_numeric=True,
    )
    y_all_high = (featured[TARGET_COLUMN].astype(float) >= high_demand_threshold).astype(int)
    final_classifier.fit(featured[FEATURE_COLUMNS], y_all_high)

    model_bundle = {
        "model": final_regression_pipeline,
        "best_model_name": best_regression_name,
        "metrics": regression_results,
        "feature_columns": FEATURE_COLUMNS,
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "target_column": TARGET_COLUMN,
        "trained_through_date": str(df["Date"].max().date()),
        "validation_start_date": str(cutoff_date.date()),
        "selection_metric": "MAE (lower is better)",
    }
    classifier_bundle = {
        "model": final_classifier,
        "classifier_name": "Logistic Regression Classifier",
        "metrics": classifier_metrics,
        "high_demand_threshold": high_demand_threshold,
        "positive_class_definition": classifier_metrics["positive_class_definition"],
        "trained_through_date": str(df["Date"].max().date()),
        "validation_start_date": str(cutoff_date.date()),
        "feature_columns": FEATURE_COLUMNS,
    }

    with open(MODEL_PATH, "wb") as file:
        pickle.dump(model_bundle, file, protocol=pickle.HIGHEST_PROTOCOL)
    with open(CLASSIFIER_PATH, "wb") as file:
        pickle.dump(classifier_bundle, file, protocol=pickle.HIGHEST_PROTOCOL)

    pd.DataFrame(regression_results).to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    pd.DataFrame([classifier_metrics]).to_csv(OUTPUT_DIR / "classification_metrics.csv", index=False)

    validation_output = test_df[["Date", "Store ID", "Product ID", TARGET_COLUMN]].copy()
    validation_output["Predicted Demand"] = best_validation_predictions
    validation_output["Absolute Error"] = (
        validation_output[TARGET_COLUMN] - validation_output["Predicted Demand"]
    ).abs()
    validation_output.to_csv(OUTPUT_DIR / "validation_predictions.csv", index=False)

    classification_output = test_df[["Date", "Store ID", "Product ID", TARGET_COLUMN]].copy()
    classification_output["Actual High Demand"] = y_test_high.to_numpy()
    classification_output["Predicted High Demand"] = class_predictions.astype(int)
    classification_output["High Demand Probability"] = class_probabilities
    classification_output.to_csv(
        OUTPUT_DIR / "classification_validation_predictions.csv", index=False
    )

    print("\nRegression model comparison (sorted by MAE):")
    print(pd.DataFrame(regression_results).to_string(index=False))
    print("\nLogistic Regression classification metrics:")
    print(pd.DataFrame([classifier_metrics]).to_string(index=False))
    print(f"\nBest demand forecasting model: {best_regression_name}")
    print(f"Saved regression pickle: {MODEL_PATH}")
    print(f"Saved classifier pickle: {CLASSIFIER_PATH}")
    print(f"Saved metrics to: {OUTPUT_DIR}")
    print(f"Total runtime: {time.time() - start:.1f}s")
    print(
        "\nNote: regression validation is a time-based one-day-ahead evaluation. "
        "The logistic classifier labels demand >= the training-set 75th percentile "
        "as high demand. Multi-day recursive forecasts can have larger errors."
    )


if __name__ == "__main__":
    main()
