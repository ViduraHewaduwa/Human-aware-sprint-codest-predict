#!/usr/bin/env python3
"""Train leakage-safe Random Forest models for Agile sprint cost forecasting."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib_cache")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


TARGET_COLUMN = "actual_sprint_cost"
TEAM_COLUMN = "team_id"
DEFAULT_DATA_PATH = Path("Dataset/final_agile_dataset.csv")
DEFAULT_OUTPUT_DIR = Path("artifacts/sprint_cost_forecasting")
PLOT_STYLE = "seaborn-v0_8-whitegrid"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    name: str
    raw_features: list[str]
    engineered_features: list[str]
    outlier_features: list[str]


@dataclass
class EvaluationMetrics:
    r2: float
    rmse: float
    mae: float
    mape: float


MODEL_SPECS: dict[str, ModelSpec] = {
    "initial": ModelSpec(
        key="initial",
        name="Initial Sprint Cost Estimation",
        raw_features=[
            "team_id",
            "team_size",
            "total_tasks",
            "planned_story_points",
            "historical_velocity_avg",
            "absence_days_total",
            "skill_match_ratio",
        ],
        engineered_features=[
            "points_per_task",
            "absence_per_developer",
            "tasks_per_developer",
            "planned_points_per_developer",
            "planned_vs_velocity",
            "velocity_per_developer",
        ],
        outlier_features=[
            "planned_story_points",
            "historical_velocity_avg",
        ],
    ),
    "mid": ModelSpec(
        key="mid",
        name="Mid-Sprint Cost Update",
        raw_features=[
            "team_id",
            "team_size",
            "total_tasks",
            "planned_story_points",
            "historical_velocity_avg",
            "absence_days_total",
            "skill_match_ratio",
            "completed_story_points",
            "overtime_hours_total",
            "total_time_spent",
        ],
        engineered_features=[
            "points_per_task",
            "absence_per_developer",
            "tasks_per_developer",
            "planned_points_per_developer",
            "planned_vs_velocity",
            "velocity_per_developer",
            "completion_ratio",
            "points_completed_per_developer",
            "overtime_per_developer",
            "overtime_per_completed_point",
            "time_spent_per_completed_point",
        ],
        outlier_features=[
            "planned_story_points",
            "historical_velocity_avg",
        ],
    ),
}


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    numerator_values = numerator.astype(float).to_numpy()
    denominator_values = denominator.astype(float).to_numpy()
    result = np.divide(
        numerator_values,
        denominator_values,
        out=np.full_like(numerator_values, np.nan, dtype=float),
        where=denominator_values != 0,
    )
    return pd.Series(result, index=numerator.index, dtype=float)


class AgileFeatureBuilder(BaseEstimator, TransformerMixin):
    """Create leakage-safe derived features from allowed columns."""

    def __init__(self, model_key: str):
        self.model_key = model_key

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "AgileFeatureBuilder":
        self.feature_names_in_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        spec = MODEL_SPECS[self.model_key]
        data = X.copy()
        data["points_per_task"] = safe_divide(
            data["planned_story_points"], data["total_tasks"]
        )
        data["absence_per_developer"] = safe_divide(
            data["absence_days_total"], data["team_size"]
        )
        data["tasks_per_developer"] = safe_divide(
            data["total_tasks"], data["team_size"]
        )
        data["planned_points_per_developer"] = safe_divide(
            data["planned_story_points"], data["team_size"]
        )
        data["planned_vs_velocity"] = safe_divide(
            data["planned_story_points"], data["historical_velocity_avg"]
        )
        data["velocity_per_developer"] = safe_divide(
            data["historical_velocity_avg"], data["team_size"]
        )

        if self.model_key == "mid":
            data["completion_ratio"] = safe_divide(
                data["completed_story_points"], data["planned_story_points"]
            )
            data["points_completed_per_developer"] = safe_divide(
                data["completed_story_points"], data["team_size"]
            )
            data["overtime_per_developer"] = safe_divide(
                data["overtime_hours_total"], data["team_size"]
            )
            data["overtime_per_completed_point"] = safe_divide(
                data["overtime_hours_total"], data["completed_story_points"]
            )
            data["time_spent_per_completed_point"] = safe_divide(
                data["total_time_spent"], data["completed_story_points"]
            )

        ordered_columns = list(dict.fromkeys(spec.raw_features + spec.engineered_features))
        return data[ordered_columns].replace([np.inf, -np.inf], np.nan)

    def get_feature_names_out(self, input_features: list[str] | None = None) -> np.ndarray:
        base_features = input_features if input_features is not None else self.feature_names_in_
        spec = MODEL_SPECS[self.model_key]
        return np.array(list(dict.fromkeys(list(base_features) + spec.engineered_features)))


class FeatureOutlierCapper(BaseEstimator, TransformerMixin):
    """Cap extreme numeric outliers using training-only IQR bounds."""

    def __init__(self, columns: list[str], iqr_multiplier: float = 3.0):
        self.columns = columns
        self.iqr_multiplier = iqr_multiplier

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "FeatureOutlierCapper":
        self.bounds_: dict[str, dict[str, float]] = {}
        for column in self.columns:
            if column not in X.columns:
                continue
            series = pd.to_numeric(X[column], errors="coerce").dropna()
            if series.empty:
                continue
            q1 = float(series.quantile(0.25))
            q3 = float(series.quantile(0.75))
            iqr = q3 - q1
            lower = max(0.0, q1 - self.iqr_multiplier * iqr)
            upper = q3 + self.iqr_multiplier * iqr if iqr > 0 else q3
            self.bounds_[column] = {
                "lower": lower,
                "upper": float(upper),
                "q1": q1,
                "q3": q3,
            }
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        data = X.copy()
        for column, bounds in getattr(self, "bounds_", {}).items():
            if column in data.columns:
                data[column] = data[column].clip(bounds["lower"], bounds["upper"])
        return data


class LogTargetRegressor(BaseEstimator, RegressorMixin):
    """Log-transform the cost target, then invert predictions back to cost space."""

    def __init__(self, regressor: RandomForestRegressor):
        self.regressor = regressor

    def fit(self, X, y):
        y_series = pd.Series(y, dtype=float)
        self.regressor_ = clone(self.regressor)
        self.regressor_.fit(X, np.log1p(y_series))
        return self

    def predict(self, X):
        log_predictions = self.regressor_.predict(X)
        return np.expm1(log_predictions)

    @property
    def feature_importances_(self):
        return self.regressor_.feature_importances_


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train or use leakage-safe Random Forest sprint cost forecasting models."
    )
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train", help="Train both sprint cost models.")
    train_parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=f"Training CSV path. Default: {DEFAULT_DATA_PATH}",
    )
    train_parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Artifact directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    train_parser.add_argument(
        "--test-group-fraction",
        type=float,
        default=0.2,
        help="Fraction of teams held out for testing. Default: 0.2",
    )
    train_parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Maximum GroupKFold splits for cross-validation. Default: 5",
    )
    train_parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used for reproducibility. Default: 42",
    )

    predict_parser = subparsers.add_parser("predict", help="Predict sprint cost from a CSV.")
    predict_parser.add_argument(
        "--model-key",
        choices=sorted(MODEL_SPECS),
        required=True,
        help="Which trained model to use.",
    )
    predict_parser.add_argument(
        "--input-path",
        type=Path,
        required=True,
        help="Input CSV containing the required feature columns.",
    )
    predict_parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("predictions.csv"),
        help="CSV path for predictions. Default: predictions.csv",
    )
    predict_parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Artifact directory containing saved models. Default: {DEFAULT_OUTPUT_DIR}",
    )

    args = parser.parse_args()
    if args.command is None:
        args.command = "train"
        args.data_path = DEFAULT_DATA_PATH
        args.output_dir = DEFAULT_OUTPUT_DIR
        args.test_group_fraction = 0.2
        args.cv_folds = 5
        args.random_state = 42
    return args


def load_dataset(data_path: Path, require_target: bool = True) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    data = pd.read_csv(data_path)
    if require_target and TARGET_COLUMN not in data.columns:
        raise ValueError(f"Expected target column '{TARGET_COLUMN}' in {data_path}")
    return data


def validate_columns(df: pd.DataFrame) -> None:
    required_columns = {
        TARGET_COLUMN,
        TEAM_COLUMN,
        "team_size",
        "total_tasks",
        "planned_story_points",
        "historical_velocity_avg",
        "absence_days_total",
        "skill_match_ratio",
        "completed_story_points",
        "overtime_hours_total",
        "total_time_spent",
    }
    missing = sorted(required_columns.difference(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")


def clean_training_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    report: dict[str, object] = {
        "input_rows": int(len(df)),
        "missing_values_by_column": {
            column: int(count)
            for column, count in df.isna().sum().items()
            if int(count) > 0
        },
        "historical_velocity_eq_200_count": int((df["historical_velocity_avg"] == 200).sum()),
    }

    invalid_mask = (df["team_size"] <= 0) | (df["total_tasks"] <= 0)
    invalid_mask |= df[TARGET_COLUMN].isna()
    cleaned = df.loc[~invalid_mask].copy()
    cleaned[TEAM_COLUMN] = cleaned[TEAM_COLUMN].fillna("UNKNOWN_TEAM").astype(str)

    report["removed_invalid_rows"] = int(invalid_mask.sum())
    report["remaining_rows"] = int(len(cleaned))
    report["removed_rows_breakdown"] = {
        "team_size_le_zero": int((df["team_size"] <= 0).sum()),
        "total_tasks_le_zero": int((df["total_tasks"] <= 0).sum()),
        "missing_target": int(df[TARGET_COLUMN].isna().sum()),
    }
    return cleaned, report


def split_train_test_by_team(
    df: pd.DataFrame, test_group_fraction: float, random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    group_count = df[TEAM_COLUMN].nunique()
    if group_count < 2:
        raise ValueError("Need at least two teams to create a group-based holdout split.")

    splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_group_fraction, random_state=random_state
    )
    train_index, test_index = next(splitter.split(df, groups=df[TEAM_COLUMN]))
    train_df = df.iloc[train_index].copy()
    test_df = df.iloc[test_index].copy()

    split_report = {
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_team_count": int(train_df[TEAM_COLUMN].nunique()),
        "test_team_count": int(test_df[TEAM_COLUMN].nunique()),
        "held_out_teams": sorted(test_df[TEAM_COLUMN].unique().tolist()),
    }
    return train_df, test_df, split_report


def split_train_test_by_time(
    df: pd.DataFrame,
    test_fraction: float,
    time_column: str = "sprint_number",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    if time_column not in df.columns:
        raise ValueError(f"Time-based split requires column '{time_column}'.")

    train_frames: list[pd.DataFrame] = []
    test_frames: list[pd.DataFrame] = []

    for _, team_df in df.groupby(TEAM_COLUMN, sort=False):
        ordered = team_df.sort_values([time_column, "sprint_id"], kind="mergesort")
        if len(ordered) < 2:
            train_frames.append(ordered)
            continue

        split_index = int(np.floor(len(ordered) * (1 - test_fraction)))
        split_index = min(max(split_index, 1), len(ordered) - 1)
        train_frames.append(ordered.iloc[:split_index])
        test_frames.append(ordered.iloc[split_index:])

    train_df = pd.concat(train_frames).sort_values([time_column, TEAM_COLUMN, "sprint_id"])
    test_df = pd.concat(test_frames).sort_values([time_column, TEAM_COLUMN, "sprint_id"])
    split_report = {
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_team_count": int(train_df[TEAM_COLUMN].nunique()),
        "test_team_count": int(test_df[TEAM_COLUMN].nunique()),
        "shared_team_count": int(
            len(set(train_df[TEAM_COLUMN].unique()).intersection(test_df[TEAM_COLUMN].unique()))
        ),
        "train_time_range": [
            float(train_df[time_column].min()),
            float(train_df[time_column].max()),
        ],
        "test_time_range": [
            float(test_df[time_column].min()),
            float(test_df[time_column].max()),
        ],
    }
    return train_df, test_df, split_report


def build_preprocessor(feature_columns: list[str]) -> ColumnTransformer:
    categorical_features = [TEAM_COLUMN] if TEAM_COLUMN in feature_columns else []
    numeric_features = [column for column in feature_columns if column not in categorical_features]
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                numeric_features,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", make_one_hot_encoder()),
                    ]
                ),
                categorical_features,
            ),
        ]
    )


def build_training_pipeline(model_key: str, random_state: int) -> Pipeline:
    spec = MODEL_SPECS[model_key]
    preprocessor = build_preprocessor(spec.raw_features + spec.engineered_features)
    regressor = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
    )
    return Pipeline(
        steps=[
            ("feature_builder", AgileFeatureBuilder(model_key=model_key)),
            ("feature_capper", FeatureOutlierCapper(columns=spec.outlier_features)),
            ("preprocessor", preprocessor),
            ("model", LogTargetRegressor(regressor=regressor)),
        ]
    )


def safe_mape(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    true_values = np.asarray(y_true, dtype=float)
    predicted_values = np.asarray(y_pred, dtype=float)
    denominator = np.clip(np.abs(true_values), 1.0, None)
    return float(np.mean(np.abs(true_values - predicted_values) / denominator) * 100)


def rmse_score(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def compute_metrics(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> EvaluationMetrics:
    return EvaluationMetrics(
        r2=float(r2_score(y_true, y_pred)),
        rmse=rmse_score(y_true, y_pred),
        mae=float(mean_absolute_error(y_true, y_pred)),
        mape=safe_mape(y_true, y_pred),
    )


def cross_validate_model(
    pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series, groups: pd.Series, cv_folds: int
) -> dict[str, float]:
    unique_groups = int(groups.nunique())
    split_count = min(cv_folds, unique_groups)
    if split_count < 2:
        raise ValueError("Need at least two training teams for GroupKFold cross-validation.")

    cv = GroupKFold(n_splits=split_count)
    scoring = {
        "r2": "r2",
        "rmse": make_scorer(rmse_score, greater_is_better=False),
        "mae": make_scorer(mean_absolute_error, greater_is_better=False),
        "mape": make_scorer(safe_mape, greater_is_better=False),
    }
    scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        groups=groups,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
        return_train_score=False,
    )
    return {
        "folds": split_count,
        "cv_r2_mean": float(scores["test_r2"].mean()),
        "cv_r2_std": float(scores["test_r2"].std()),
        "cv_rmse_mean": float(-scores["test_rmse"].mean()),
        "cv_rmse_std": float(scores["test_rmse"].std()),
        "cv_mae_mean": float(-scores["test_mae"].mean()),
        "cv_mape_mean": float(-scores["test_mape"].mean()),
    }


def assign_size_band(
    values: pd.Series, low_cutoff: float, high_cutoff: float
) -> pd.Series:
    return pd.Series(
        np.where(
            values <= low_cutoff,
            "small",
            np.where(values >= high_cutoff, "large", "medium"),
        ),
        index=values.index,
    )


def summarize_group_metrics(df: pd.DataFrame, group_column: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group_name, group_df in df.groupby(group_column, dropna=False):
        actual = group_df[TARGET_COLUMN]
        predicted = group_df["predicted_sprint_cost"]
        rmse_value = rmse_score(actual, predicted)
        mae_value = float(mean_absolute_error(actual, predicted))
        mape_value = safe_mape(actual, predicted)
        r2_value: float | None
        if len(group_df) < 2:
            r2_value = None
        else:
            r2_value = float(r2_score(actual, predicted))
        rows.append(
            {
                group_column: group_name,
                "count": int(len(group_df)),
                "r2": r2_value,
                "rmse": rmse_value,
                "mae": mae_value,
                "mape": mape_value,
            }
        )
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    return summary.sort_values(["mae", "rmse"], ascending=[False, False]).reset_index(drop=True)


def sanitize_feature_names(feature_names: np.ndarray) -> list[str]:
    clean_names: list[str] = []
    for name in feature_names.tolist():
        updated = name.replace("numeric__", "")
        updated = updated.replace("categorical__", "")
        updated = updated.replace("team_id_", "team_id=")
        clean_names.append(updated)
    return clean_names


def extract_feature_importance(pipeline: Pipeline) -> pd.DataFrame:
    importances = pipeline.named_steps["model"].feature_importances_
    feature_names = sanitize_feature_names(
        pipeline.named_steps["preprocessor"].get_feature_names_out()
    )
    return (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def save_feature_importance_plot(feature_importance: pd.DataFrame, output_path: Path) -> None:
    top_features = feature_importance.head(15).iloc[::-1]
    plt.figure(figsize=(10, 6))
    plt.barh(top_features["feature"], top_features["importance"], color="#2a6f97")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_actual_vs_predicted_plot(df: pd.DataFrame, output_path: Path, title: str) -> None:
    plt.figure(figsize=(7, 7))
    plt.scatter(
        df[TARGET_COLUMN],
        df["predicted_sprint_cost"],
        alpha=0.55,
        color="#468faf",
        edgecolor="none",
    )
    max_axis = float(max(df[TARGET_COLUMN].max(), df["predicted_sprint_cost"].max()))
    plt.plot([0, max_axis], [0, max_axis], linestyle="--", color="#d00000", linewidth=1.5)
    plt.xlabel("Actual Sprint Cost")
    plt.ylabel("Predicted Sprint Cost")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_residual_distribution_plot(df: pd.DataFrame, output_path: Path) -> None:
    residuals = df["residual"]
    plt.figure(figsize=(8, 5))
    plt.hist(residuals, bins=35, color="#90be6d", edgecolor="white")
    plt.axvline(0, color="#d00000", linestyle="--", linewidth=1.5)
    plt.xlabel("Residual (Actual - Predicted)")
    plt.ylabel("Frequency")
    plt.title("Residual Distribution")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_error_distribution_plot(df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.hist(df["absolute_error"], bins=35, color="#f4a261", edgecolor="white")
    plt.xlabel("Absolute Error")
    plt.ylabel("Frequency")
    plt.title("Error Distribution")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_model_artifacts(
    model_dir: Path,
    model_key: str,
    pipeline: Pipeline,
    feature_importance: pd.DataFrame,
    group_holdout_predictions: pd.DataFrame,
    time_holdout_predictions: pd.DataFrame,
    metrics_payload: dict[str, object],
) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)

    (model_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2))
    group_holdout_predictions.to_csv(model_dir / "holdout_predictions.csv", index=False)
    group_holdout_predictions.to_csv(model_dir / "group_holdout_predictions.csv", index=False)
    time_holdout_predictions.to_csv(model_dir / "time_holdout_predictions.csv", index=False)
    feature_importance.to_csv(model_dir / "feature_importance.csv", index=False)

    joblib.dump(pipeline, model_dir / f"{model_key}_random_forest_model.joblib")
    joblib.dump(pipeline[:-1], model_dir / f"{model_key}_preprocessing_pipeline.joblib")

    feature_metadata = {
        "model_key": model_key,
        "raw_features": MODEL_SPECS[model_key].raw_features,
        "engineered_features": MODEL_SPECS[model_key].engineered_features,
        "transformed_features": feature_importance["feature"].tolist(),
    }
    (model_dir / "feature_list.json").write_text(json.dumps(feature_metadata, indent=2))

    save_feature_importance_plot(feature_importance, model_dir / "feature_importance.png")
    save_actual_vs_predicted_plot(
        group_holdout_predictions,
        model_dir / "actual_vs_predicted.png",
        f"{MODEL_SPECS[model_key].name}: Team Holdout Actual vs Predicted",
    )
    save_residual_distribution_plot(
        group_holdout_predictions, model_dir / "residual_distribution.png"
    )
    save_error_distribution_plot(group_holdout_predictions, model_dir / "error_distribution.png")
    save_actual_vs_predicted_plot(
        time_holdout_predictions,
        model_dir / "time_actual_vs_predicted.png",
        f"{MODEL_SPECS[model_key].name}: Time Split Actual vs Predicted",
    )
    save_residual_distribution_plot(
        time_holdout_predictions, model_dir / "time_residual_distribution.png"
    )
    save_error_distribution_plot(
        time_holdout_predictions, model_dir / "time_error_distribution.png"
    )


def build_prediction_frame(test_df: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
    prediction_frame = test_df[
        [TEAM_COLUMN, "sprint_id", "planned_story_points", "total_tasks", TARGET_COLUMN]
    ].copy()
    prediction_frame["predicted_sprint_cost"] = predictions
    prediction_frame["residual"] = (
        prediction_frame[TARGET_COLUMN] - prediction_frame["predicted_sprint_cost"]
    )
    prediction_frame["absolute_error"] = prediction_frame["residual"].abs()
    prediction_frame["absolute_percentage_error"] = (
        prediction_frame["absolute_error"]
        / np.clip(prediction_frame[TARGET_COLUMN].abs(), 1.0, None)
        * 100
    )
    return prediction_frame


def enrich_prediction_frame_with_size_bands(
    prediction_frame: pd.DataFrame,
    reference_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float]]:
    small_cutoff = float(reference_df["planned_story_points"].quantile(0.33))
    large_cutoff = float(reference_df["planned_story_points"].quantile(0.67))
    binary_cutoff = float(reference_df["planned_story_points"].median())
    enriched = prediction_frame.copy()
    enriched["sprint_size_band"] = assign_size_band(
        enriched["planned_story_points"], small_cutoff, large_cutoff
    )
    enriched["small_vs_large"] = np.where(
        enriched["planned_story_points"] <= binary_cutoff, "small", "large"
    )
    thresholds = {
        "small_upper_bound": small_cutoff,
        "large_lower_bound": large_cutoff,
        "small_vs_large_cutoff": binary_cutoff,
    }
    return enriched, thresholds


def evaluate_split(
    model_key: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    random_state: int,
    cv_folds: int | None,
    scenario_name: str,
) -> dict[str, object]:
    spec = MODEL_SPECS[model_key]
    pipeline = build_training_pipeline(model_key=model_key, random_state=random_state)

    X_train = train_df[spec.raw_features].copy()
    y_train = train_df[TARGET_COLUMN].astype(float)
    X_test = test_df[spec.raw_features].copy()
    y_test = test_df[TARGET_COLUMN].astype(float)

    cv_results = None
    if cv_folds is not None:
        cv_results = cross_validate_model(
            pipeline=pipeline,
            X_train=X_train,
            y_train=y_train,
            groups=train_df[TEAM_COLUMN],
            cv_folds=cv_folds,
        )

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    holdout_metrics = compute_metrics(y_test, predictions)

    holdout_predictions = build_prediction_frame(test_df, predictions)
    holdout_predictions, thresholds = enrich_prediction_frame_with_size_bands(
        holdout_predictions, train_df
    )

    team_metrics = summarize_group_metrics(holdout_predictions, TEAM_COLUMN)
    size_metrics = summarize_group_metrics(holdout_predictions, "sprint_size_band")
    small_large_metrics = summarize_group_metrics(holdout_predictions, "small_vs_large")

    return {
        "scenario_name": scenario_name,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "cross_validation": cv_results,
        "holdout_metrics": asdict(holdout_metrics),
        "team_error_summary": team_metrics.to_dict(orient="records"),
        "sprint_size_error_summary": size_metrics.to_dict(orient="records"),
        "small_vs_large_error_summary": small_large_metrics.to_dict(orient="records"),
        "size_band_thresholds": thresholds,
        "predictions": holdout_predictions,
    }


def train_models(
    data_path: Path,
    output_dir: Path,
    test_group_fraction: float,
    cv_folds: int,
    random_state: int,
) -> None:
    try:
        plt.style.use(PLOT_STYLE)
    except OSError:
        pass

    df = load_dataset(data_path, require_target=True)
    validate_columns(df)
    cleaned_df, cleaning_report = clean_training_rows(df)
    group_train_df, group_test_df, group_split_report = split_train_test_by_team(
        cleaned_df,
        test_group_fraction=test_group_fraction,
        random_state=random_state,
    )
    time_train_df, time_test_df, time_split_report = split_train_test_by_time(
        cleaned_df,
        test_fraction=test_group_fraction,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    model_summaries = []
    for model_key in ("initial", "mid"):
        group_evaluation = evaluate_split(
            model_key=model_key,
            train_df=group_train_df,
            test_df=group_test_df,
            random_state=random_state,
            cv_folds=cv_folds,
            scenario_name="group_holdout",
        )
        time_evaluation = evaluate_split(
            model_key=model_key,
            train_df=time_train_df,
            test_df=time_test_df,
            random_state=random_state,
            cv_folds=None,
            scenario_name="time_based",
        )

        spec = MODEL_SPECS[model_key]
        final_pipeline = build_training_pipeline(model_key=model_key, random_state=random_state)
        final_pipeline.fit(
            cleaned_df[spec.raw_features], cleaned_df[TARGET_COLUMN].astype(float)
        )
        feature_importance = extract_feature_importance(final_pipeline)
        model_dir = output_dir / f"{model_key}_model"
        feature_capper = final_pipeline.named_steps["feature_capper"]
        metrics_payload = {
            "model_key": model_key,
            "model_name": spec.name,
            "saved_model_training_rows": int(len(cleaned_df)),
            "raw_features": spec.raw_features,
            "engineered_features": spec.engineered_features,
            "evaluations": {
                "group_holdout": {
                    "training_rows": group_evaluation["train_rows"],
                    "holdout_rows": group_evaluation["test_rows"],
                    "cross_validation": group_evaluation["cross_validation"],
                    "holdout_metrics": group_evaluation["holdout_metrics"],
                    "team_error_summary": group_evaluation["team_error_summary"],
                    "sprint_size_error_summary": group_evaluation["sprint_size_error_summary"],
                    "small_vs_large_error_summary": group_evaluation[
                        "small_vs_large_error_summary"
                    ],
                    "size_band_thresholds": group_evaluation["size_band_thresholds"],
                },
                "time_based": {
                    "training_rows": time_evaluation["train_rows"],
                    "holdout_rows": time_evaluation["test_rows"],
                    "cross_validation": time_evaluation["cross_validation"],
                    "holdout_metrics": time_evaluation["holdout_metrics"],
                    "team_error_summary": time_evaluation["team_error_summary"],
                    "sprint_size_error_summary": time_evaluation["sprint_size_error_summary"],
                    "small_vs_large_error_summary": time_evaluation[
                        "small_vs_large_error_summary"
                    ],
                    "size_band_thresholds": time_evaluation["size_band_thresholds"],
                },
            },
            "feature_outlier_caps": getattr(feature_capper, "bounds_", {}),
            "target_transform": "log1p_only_no_target_clipping",
        }
        save_model_artifacts(
            model_dir=model_dir,
            model_key=model_key,
            pipeline=final_pipeline,
            feature_importance=feature_importance,
            group_holdout_predictions=group_evaluation["predictions"],
            time_holdout_predictions=time_evaluation["predictions"],
            metrics_payload=metrics_payload,
        )
        model_summaries.append(
            {
                "model_key": model_key,
                "model_name": spec.name,
                "artifact_dir": str(model_dir),
                "group_holdout": group_evaluation["holdout_metrics"],
                "time_based": time_evaluation["holdout_metrics"],
            }
        )

    summary_payload = {
        "data_path": str(data_path),
        "output_dir": str(output_dir),
        "random_state": random_state,
        "test_group_fraction": test_group_fraction,
        "cleaning_report": cleaning_report,
        "group_split_report": group_split_report,
        "time_split_report": time_split_report,
        "models": model_summaries,
    }
    (output_dir / "training_summary.json").write_text(json.dumps(summary_payload, indent=2))

    print(f"Training complete. Artifacts saved to: {output_dir.resolve()}")
    print(f"Held-out teams: {', '.join(group_split_report['held_out_teams'])}")
    for summary in model_summaries:
        print(
            f"{summary['model_name']} -> "
            f"group holdout R^2: {summary['group_holdout']['r2']:.4f}, "
            f"time split R^2: {summary['time_based']['r2']:.4f}"
        )


def predict_with_model(
    model_key: str, input_path: Path, output_path: Path, artifact_dir: Path
) -> None:
    model_path = artifact_dir / f"{model_key}_model" / f"{model_key}_random_forest_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Saved model not found: {model_path}")

    model = joblib.load(model_path)
    input_df = load_dataset(input_path, require_target=False)
    required_features = list(getattr(model, "feature_names_in_", []))
    missing = [feature for feature in required_features if feature not in input_df.columns]
    if missing:
        raise ValueError(
            "Prediction input is missing required columns: " + ", ".join(sorted(missing))
        )

    predictions = model.predict(input_df[required_features])
    result = input_df.copy()
    result["predicted_sprint_cost"] = predictions
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    print(f"Predictions saved to: {output_path.resolve()}")
    print(result.head(5).to_string(index=False))


def main() -> None:
    args = parse_args()
    if args.command == "predict":
        predict_with_model(
            model_key=args.model_key,
            input_path=args.input_path,
            output_path=args.output_path,
            artifact_dir=args.artifact_dir,
        )
        return

    train_models(
        data_path=args.data_path,
        output_dir=args.output_dir,
        test_group_fraction=args.test_group_fraction,
        cv_folds=args.cv_folds,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
    except KeyboardInterrupt:
        raise SystemExit("\nCancelled by user.") from None
