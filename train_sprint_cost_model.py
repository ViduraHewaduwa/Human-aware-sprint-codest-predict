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
            "planned_points_team_interaction",
            "task_team_interaction",
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
            "planned_points_team_interaction",
            "task_team_interaction",
            "completion_ratio",
            "completion_gap",
            "progress_vs_expected",
            "remaining_points_per_developer",
            "points_completed_per_developer",
            "overtime_per_developer",
            "overtime_per_completed_point",
            "time_spent_per_developer",
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
        data["planned_points_team_interaction"] = (
            data["planned_story_points"].astype(float) * data["team_size"].astype(float)
        )
        data["task_team_interaction"] = (
            data["total_tasks"].astype(float) * data["team_size"].astype(float)
        )

        if self.model_key == "mid":
            data["completion_ratio"] = safe_divide(
                data["completed_story_points"], data["planned_story_points"]
            )
            data["completion_gap"] = (
                data["planned_story_points"].astype(float)
                - data["completed_story_points"].astype(float)
            )
            data["progress_vs_expected"] = safe_divide(
                data["completed_story_points"], data["historical_velocity_avg"]
            )
            data["remaining_points_per_developer"] = safe_divide(
                data["completion_gap"], data["team_size"]
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
            data["time_spent_per_developer"] = safe_divide(
                data["total_time_spent"], data["team_size"]
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

    def fit(self, X, y, sample_weight=None):
        y_series = pd.Series(y, dtype=float)
        self.regressor_ = clone(self.regressor)
        fit_kwargs: dict[str, object] = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = np.asarray(sample_weight, dtype=float)
        self.regressor_.fit(X, np.log1p(y_series), **fit_kwargs)
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


def build_random_forest(random_state: int) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
    )


def build_training_sample_weights(y: pd.Series) -> pd.Series:
    y_series = pd.Series(y, dtype=float)
    q80 = float(y_series.quantile(0.80))
    q95 = float(y_series.quantile(0.95))
    weights = np.ones(len(y_series), dtype=float)
    weights[y_series >= q80] = 2.0
    weights[y_series >= q95] = 4.0
    return pd.Series(weights, index=y_series.index, dtype=float)


class SingleModelSprintCostForecaster(BaseEstimator, RegressorMixin):
    """Single weighted Random Forest forecaster over all sprint rows."""

    def __init__(self, model_key: str, random_state: int = 42, use_sample_weights: bool = True):
        self.model_key = model_key
        self.random_state = random_state
        self.use_sample_weights = use_sample_weights

    def fit(self, X: pd.DataFrame, y) -> "SingleModelSprintCostForecaster":
        spec = MODEL_SPECS[self.model_key]
        self.feature_names_in_ = np.array(spec.raw_features, dtype=object)
        X_df = X.loc[:, spec.raw_features].copy()
        y_series = pd.Series(y, index=X_df.index, dtype=float)

        self.feature_builder_ = AgileFeatureBuilder(model_key=self.model_key)
        built = self.feature_builder_.fit_transform(X_df)
        self.feature_capper_ = FeatureOutlierCapper(columns=spec.outlier_features)
        capped = self.feature_capper_.fit_transform(built)
        self.preprocessor_ = build_preprocessor(spec.raw_features + spec.engineered_features)
        transformed = self.preprocessor_.fit_transform(capped)

        self.model_ = LogTargetRegressor(regressor=build_random_forest(self.random_state))
        sample_weights = (
            build_training_sample_weights(y_series) if self.use_sample_weights else None
        )
        self.model_.fit(transformed, y_series, sample_weight=sample_weights)
        self.sample_weight_summary_ = (
            {
                "q80_cost": float(y_series.quantile(0.80)),
                "q95_cost": float(y_series.quantile(0.95)),
                "max_weight": float(sample_weights.max()),
            }
            if sample_weights is not None
            else None
        )
        return self

    def _prepare_features(self, X: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        spec = MODEL_SPECS[self.model_key]
        X_df = X.loc[:, spec.raw_features].copy()
        built = self.feature_builder_.transform(X_df)
        capped = self.feature_capper_.transform(built)
        transformed = self.preprocessor_.transform(capped)
        return capped, transformed

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        _, transformed = self._prepare_features(X)
        return self.model_.predict(transformed)

    @property
    def feature_importances_(self) -> np.ndarray:
        return self.model_.feature_importances_

    def get_feature_importance_table(self) -> pd.DataFrame:
        feature_names = sanitize_feature_names(self.preprocessor_.get_feature_names_out())
        return (
            pd.DataFrame(
                {"feature": feature_names, "importance": self.feature_importances_}
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )


class SegmentedSprintCostForecaster(BaseEstimator, RegressorMixin):
    """Route large-complexity sprints to a dedicated weighted Random Forest."""

    def __init__(
        self,
        model_key: str,
        random_state: int = 42,
        large_quantile: float = 0.90,
        use_sample_weights: bool = True,
        min_large_rows: int = 40,
    ):
        self.model_key = model_key
        self.random_state = random_state
        self.large_quantile = large_quantile
        self.use_sample_weights = use_sample_weights
        self.min_large_rows = min_large_rows

    def fit(self, X: pd.DataFrame, y) -> "SegmentedSprintCostForecaster":
        spec = MODEL_SPECS[self.model_key]
        self.feature_names_in_ = np.array(spec.raw_features, dtype=object)
        X_df = X.loc[:, spec.raw_features].copy()
        y_series = pd.Series(y, index=X_df.index, dtype=float)

        self.feature_builder_ = AgileFeatureBuilder(model_key=self.model_key)
        built = self.feature_builder_.fit_transform(X_df)
        self.feature_capper_ = FeatureOutlierCapper(columns=spec.outlier_features)
        capped = self.feature_capper_.fit_transform(built)
        self.preprocessor_ = build_preprocessor(spec.raw_features + spec.engineered_features)
        transformed = self.preprocessor_.fit_transform(capped)

        self.routing_thresholds_ = {
            "planned_story_points": float(capped["planned_story_points"].quantile(self.large_quantile)),
            "total_tasks": float(capped["total_tasks"].quantile(self.large_quantile)),
            "planned_vs_velocity": float(capped["planned_vs_velocity"].quantile(self.large_quantile)),
        }
        large_mask = self._build_large_mask(capped)
        if int(large_mask.sum()) < self.min_large_rows:
            rank_source = capped["planned_story_points"].fillna(0.0) + capped["total_tasks"].fillna(0.0)
            top_count = min(len(rank_source), max(self.min_large_rows, int(len(rank_source) * 0.1)))
            selected_index = rank_source.nlargest(top_count).index
            large_mask = capped.index.isin(selected_index)

        large_mask = pd.Series(large_mask, index=capped.index, dtype=bool)
        normal_mask = ~large_mask
        sample_weights = (
            build_training_sample_weights(y_series) if self.use_sample_weights else None
        )

        self.normal_model_ = LogTargetRegressor(regressor=build_random_forest(self.random_state))
        self.normal_model_.fit(
            transformed[normal_mask.to_numpy()],
            y_series.loc[normal_mask],
            sample_weight=sample_weights.loc[normal_mask] if sample_weights is not None else None,
        )

        if int(large_mask.sum()) >= self.min_large_rows:
            self.large_model_ = LogTargetRegressor(regressor=build_random_forest(self.random_state))
            self.large_model_.fit(
                transformed[large_mask.to_numpy()],
                y_series.loc[large_mask],
                sample_weight=sample_weights.loc[large_mask] if sample_weights is not None else None,
            )
            self.has_large_model_ = True
        else:
            self.large_model_ = self.normal_model_
            self.has_large_model_ = False

        self.segment_summary_ = {
            "large_quantile": self.large_quantile,
            "min_large_rows": self.min_large_rows,
            "normal_rows": int(normal_mask.sum()),
            "large_rows": int(large_mask.sum()),
            "routing_thresholds": self.routing_thresholds_,
            "sample_weight_summary": (
                {
                    "q80_cost": float(y_series.quantile(0.80)),
                    "q95_cost": float(y_series.quantile(0.95)),
                }
                if sample_weights is not None
                else None
            ),
        }
        return self

    def _build_large_mask(self, capped_features: pd.DataFrame) -> pd.Series:
        return (
            (capped_features["planned_story_points"] >= self.routing_thresholds_["planned_story_points"])
            | (capped_features["total_tasks"] >= self.routing_thresholds_["total_tasks"])
            | (
                capped_features["planned_vs_velocity"].fillna(0.0)
                >= self.routing_thresholds_["planned_vs_velocity"]
            )
        )

    def _prepare_features(self, X: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        spec = MODEL_SPECS[self.model_key]
        X_df = X.loc[:, spec.raw_features].copy()
        built = self.feature_builder_.transform(X_df)
        capped = self.feature_capper_.transform(built)
        transformed = self.preprocessor_.transform(capped)
        return capped, transformed

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        capped, transformed = self._prepare_features(X)
        large_mask = self._build_large_mask(capped).to_numpy()
        predictions = np.zeros(len(capped), dtype=float)
        if (~large_mask).any():
            predictions[~large_mask] = self.normal_model_.predict(transformed[~large_mask])
        if large_mask.any():
            predictions[large_mask] = self.large_model_.predict(transformed[large_mask])
        return predictions

    @property
    def feature_importances_(self) -> np.ndarray:
        normal_rows = max(1, self.segment_summary_["normal_rows"])
        large_rows = max(1, self.segment_summary_["large_rows"])
        total_rows = normal_rows + large_rows
        return (
            self.normal_model_.feature_importances_ * (normal_rows / total_rows)
            + self.large_model_.feature_importances_ * (large_rows / total_rows)
        )

    def get_feature_importance_table(self) -> pd.DataFrame:
        feature_names = sanitize_feature_names(self.preprocessor_.get_feature_names_out())
        return (
            pd.DataFrame(
                {"feature": feature_names, "importance": self.feature_importances_}
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )


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


def build_baseline_estimator(model_key: str, random_state: int) -> SingleModelSprintCostForecaster:
    return SingleModelSprintCostForecaster(
        model_key=model_key,
        random_state=random_state,
        use_sample_weights=True,
    )


def build_segmented_estimator(
    model_key: str, random_state: int
) -> SegmentedSprintCostForecaster:
    return SegmentedSprintCostForecaster(
        model_key=model_key,
        random_state=random_state,
        large_quantile=0.90,
        use_sample_weights=True,
        min_large_rows=40,
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
    estimator, X_train: pd.DataFrame, y_train: pd.Series, groups: pd.Series, cv_folds: int
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
        estimator,
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


def extract_feature_importance(model) -> pd.DataFrame:
    if hasattr(model, "get_feature_importance_table"):
        return model.get_feature_importance_table()

    importances = model.named_steps["model"].feature_importances_
    feature_names = sanitize_feature_names(
        model.named_steps["preprocessor"].get_feature_names_out()
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
    model,
    feature_importance: pd.DataFrame,
    scenario_frames: dict[str, pd.DataFrame],
    metrics_payload: dict[str, object],
) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)

    (model_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2))
    for frame_name, frame in scenario_frames.items():
        frame.to_csv(model_dir / f"{frame_name}.csv", index=False)
    scenario_frames["group_holdout_segmented_predictions"].to_csv(
        model_dir / "holdout_predictions.csv", index=False
    )
    scenario_frames["group_holdout_segmented_predictions"].to_csv(
        model_dir / "group_holdout_predictions.csv", index=False
    )
    scenario_frames["time_based_segmented_predictions"].to_csv(
        model_dir / "time_holdout_predictions.csv", index=False
    )
    feature_importance.to_csv(model_dir / "feature_importance.csv", index=False)

    joblib.dump(model, model_dir / f"{model_key}_random_forest_model.joblib")
    joblib.dump(
        {
            "feature_builder": getattr(model, "feature_builder_", None),
            "feature_capper": getattr(model, "feature_capper_", None),
            "preprocessor": getattr(model, "preprocessor_", None),
        },
        model_dir / f"{model_key}_preprocessing_pipeline.joblib",
    )

    feature_metadata = {
        "model_key": model_key,
        "raw_features": list(getattr(model, "feature_names_in_", MODEL_SPECS[model_key].raw_features)),
        "engineered_features": MODEL_SPECS[model_key].engineered_features,
        "transformed_features": feature_importance["feature"].tolist(),
    }
    (model_dir / "feature_list.json").write_text(json.dumps(feature_metadata, indent=2))

    save_feature_importance_plot(feature_importance, model_dir / "feature_importance.png")
    save_actual_vs_predicted_plot(
        scenario_frames["group_holdout_segmented_predictions"],
        model_dir / "actual_vs_predicted.png",
        f"{MODEL_SPECS[model_key].name}: Team Holdout Actual vs Predicted",
    )
    save_residual_distribution_plot(
        scenario_frames["group_holdout_segmented_predictions"],
        model_dir / "residual_distribution.png",
    )
    save_error_distribution_plot(
        scenario_frames["group_holdout_segmented_predictions"],
        model_dir / "error_distribution.png",
    )
    save_actual_vs_predicted_plot(
        scenario_frames["time_based_segmented_predictions"],
        model_dir / "time_actual_vs_predicted.png",
        f"{MODEL_SPECS[model_key].name}: Time Split Actual vs Predicted",
    )
    save_residual_distribution_plot(
        scenario_frames["time_based_segmented_predictions"],
        model_dir / "time_residual_distribution.png",
    )
    save_error_distribution_plot(
        scenario_frames["time_based_segmented_predictions"],
        model_dir / "time_error_distribution.png",
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


def compute_tail_metrics(
    prediction_frame: pd.DataFrame, tail_threshold: float
) -> dict[str, object]:
    tail_frame = prediction_frame.loc[prediction_frame[TARGET_COLUMN] >= tail_threshold].copy()
    if tail_frame.empty:
        return {
            "tail_threshold": tail_threshold,
            "tail_row_count": 0,
            "metrics": None,
        }
    tail_metrics = None
    if len(tail_frame) >= 2:
        tail_metrics = asdict(
            compute_metrics(tail_frame[TARGET_COLUMN], tail_frame["predicted_sprint_cost"])
        )
    return {
        "tail_threshold": tail_threshold,
        "tail_row_count": int(len(tail_frame)),
        "metrics": tail_metrics,
    }


def compute_metric_delta(
    baseline_metrics: dict[str, float], segmented_metrics: dict[str, float]
) -> dict[str, float]:
    return {
        "r2_delta": float(segmented_metrics["r2"] - baseline_metrics["r2"]),
        "rmse_delta": float(segmented_metrics["rmse"] - baseline_metrics["rmse"]),
        "mae_delta": float(segmented_metrics["mae"] - baseline_metrics["mae"]),
        "mape_delta": float(segmented_metrics["mape"] - baseline_metrics["mape"]),
    }


def evaluate_estimator(
    estimator,
    model_key: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cv_folds: int | None,
    evaluation_label: str,
) -> dict[str, object]:
    spec = MODEL_SPECS[model_key]
    X_train = train_df[spec.raw_features].copy()
    y_train = train_df[TARGET_COLUMN].astype(float)
    X_test = test_df[spec.raw_features].copy()
    y_test = test_df[TARGET_COLUMN].astype(float)

    cv_results = None
    if cv_folds is not None:
        cv_results = cross_validate_model(
            estimator=estimator,
            X_train=X_train,
            y_train=y_train,
            groups=train_df[TEAM_COLUMN],
            cv_folds=cv_folds,
        )

    estimator.fit(X_train, y_train)
    predictions = estimator.predict(X_test)
    holdout_metrics = compute_metrics(y_test, predictions)
    holdout_predictions = build_prediction_frame(test_df, predictions)
    holdout_predictions, thresholds = enrich_prediction_frame_with_size_bands(
        holdout_predictions, train_df
    )

    team_metrics = summarize_group_metrics(holdout_predictions, TEAM_COLUMN)
    size_metrics = summarize_group_metrics(holdout_predictions, "sprint_size_band")
    small_large_metrics = summarize_group_metrics(holdout_predictions, "small_vs_large")
    tail_threshold = float(y_train.quantile(0.90))
    tail_summary = compute_tail_metrics(holdout_predictions, tail_threshold)

    return {
        "evaluation_label": evaluation_label,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "cross_validation": cv_results,
        "holdout_metrics": asdict(holdout_metrics),
        "team_error_summary": team_metrics.to_dict(orient="records"),
        "sprint_size_error_summary": size_metrics.to_dict(orient="records"),
        "small_vs_large_error_summary": small_large_metrics.to_dict(orient="records"),
        "size_band_thresholds": thresholds,
        "tail_summary": tail_summary,
        "predictions": holdout_predictions,
        "estimator": estimator,
    }


def evaluate_split(
    model_key: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    random_state: int,
    cv_folds: int | None,
    scenario_name: str,
) -> dict[str, object]:
    baseline_evaluation = evaluate_estimator(
        estimator=build_baseline_estimator(model_key=model_key, random_state=random_state),
        model_key=model_key,
        train_df=train_df,
        test_df=test_df,
        cv_folds=cv_folds,
        evaluation_label="baseline",
    )
    segmented_evaluation = evaluate_estimator(
        estimator=build_segmented_estimator(model_key=model_key, random_state=random_state),
        model_key=model_key,
        train_df=train_df,
        test_df=test_df,
        cv_folds=cv_folds,
        evaluation_label="segmented",
    )

    comparison_frame = baseline_evaluation["predictions"].merge(
        segmented_evaluation["predictions"][
            [
                TEAM_COLUMN,
                "sprint_id",
                TARGET_COLUMN,
                "predicted_sprint_cost",
                "absolute_error",
                "absolute_percentage_error",
            ]
        ].rename(
            columns={
                "predicted_sprint_cost": "segmented_predicted_sprint_cost",
                "absolute_error": "segmented_absolute_error",
                "absolute_percentage_error": "segmented_absolute_percentage_error",
            }
        ),
        on=[TEAM_COLUMN, "sprint_id", TARGET_COLUMN],
        how="inner",
    )
    comparison_frame = comparison_frame.rename(
        columns={
            "predicted_sprint_cost": "baseline_predicted_sprint_cost",
            "absolute_error": "baseline_absolute_error",
            "absolute_percentage_error": "baseline_absolute_percentage_error",
        }
    )
    comparison_frame["absolute_error_improvement"] = (
        comparison_frame["baseline_absolute_error"]
        - comparison_frame["segmented_absolute_error"]
    )
    comparison_frame["absolute_percentage_error_improvement"] = (
        comparison_frame["baseline_absolute_percentage_error"]
        - comparison_frame["segmented_absolute_percentage_error"]
    )
    comparison_frame["segmented_beats_baseline"] = (
        comparison_frame["absolute_error_improvement"] > 0
    )

    comparison_summary = {
        "metric_delta": compute_metric_delta(
            baseline_evaluation["holdout_metrics"],
            segmented_evaluation["holdout_metrics"],
        ),
        "row_level": {
            "comparison_rows": int(len(comparison_frame)),
            "segmented_better_rate": float(comparison_frame["segmented_beats_baseline"].mean()),
            "average_absolute_error_improvement": float(
                comparison_frame["absolute_error_improvement"].mean()
            ),
            "median_absolute_error_improvement": float(
                comparison_frame["absolute_error_improvement"].median()
            ),
        },
        "tail_metric_delta": compute_metric_delta(
            baseline_evaluation["tail_summary"]["metrics"],
            segmented_evaluation["tail_summary"]["metrics"],
        )
        if baseline_evaluation["tail_summary"]["metrics"] is not None
        and segmented_evaluation["tail_summary"]["metrics"] is not None
        else None,
    }

    return {
        "scenario_name": scenario_name,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "baseline": baseline_evaluation,
        "segmented": segmented_evaluation,
        "comparison_frame": comparison_frame,
        "comparison_summary": comparison_summary,
    }


def build_mid_vs_initial_comparison(
    initial_predictions: pd.DataFrame,
    mid_predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    merged = initial_predictions.merge(
        mid_predictions[
            [
                TEAM_COLUMN,
                "sprint_id",
                TARGET_COLUMN,
                "predicted_sprint_cost",
                "absolute_error",
                "absolute_percentage_error",
            ]
        ].rename(
            columns={
                "predicted_sprint_cost": "mid_predicted_sprint_cost",
                "absolute_error": "mid_absolute_error",
                "absolute_percentage_error": "mid_absolute_percentage_error",
            }
        ),
        on=[TEAM_COLUMN, "sprint_id", TARGET_COLUMN],
        how="inner",
    )
    merged = merged.rename(
        columns={
            "predicted_sprint_cost": "initial_predicted_sprint_cost",
            "absolute_error": "initial_absolute_error",
            "absolute_percentage_error": "initial_absolute_percentage_error",
        }
    )
    merged["mid_absolute_error_improvement"] = (
        merged["initial_absolute_error"] - merged["mid_absolute_error"]
    )
    merged["mid_percentage_error_improvement"] = (
        merged["initial_absolute_percentage_error"] - merged["mid_absolute_percentage_error"]
    )
    merged["mid_beats_initial"] = merged["mid_absolute_error_improvement"] > 0
    summary = {
        "comparison_rows": int(len(merged)),
        "mid_better_rate": float(merged["mid_beats_initial"].mean()),
        "average_absolute_error_improvement": float(
            merged["mid_absolute_error_improvement"].mean()
        ),
        "median_absolute_error_improvement": float(
            merged["mid_absolute_error_improvement"].median()
        ),
    }
    return merged, summary


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
    segmented_predictions_by_model: dict[str, dict[str, pd.DataFrame]] = {
        "group_holdout": {},
        "time_based": {},
    }
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
        final_model = build_segmented_estimator(model_key=model_key, random_state=random_state)
        final_model.fit(
            cleaned_df[spec.raw_features], cleaned_df[TARGET_COLUMN].astype(float)
        )
        feature_importance = extract_feature_importance(final_model)
        model_dir = output_dir / f"{model_key}_model"
        feature_capper = getattr(final_model, "feature_capper_", None)
        scenario_frames = {
            "group_holdout_baseline_predictions": group_evaluation["baseline"]["predictions"],
            "group_holdout_segmented_predictions": group_evaluation["segmented"]["predictions"],
            "group_holdout_comparison": group_evaluation["comparison_frame"],
            "time_based_baseline_predictions": time_evaluation["baseline"]["predictions"],
            "time_based_segmented_predictions": time_evaluation["segmented"]["predictions"],
            "time_based_comparison": time_evaluation["comparison_frame"],
        }
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
                    "baseline": {
                        "cross_validation": group_evaluation["baseline"]["cross_validation"],
                        "holdout_metrics": group_evaluation["baseline"]["holdout_metrics"],
                        "team_error_summary": group_evaluation["baseline"]["team_error_summary"],
                        "sprint_size_error_summary": group_evaluation["baseline"]["sprint_size_error_summary"],
                        "small_vs_large_error_summary": group_evaluation["baseline"][
                            "small_vs_large_error_summary"
                        ],
                        "size_band_thresholds": group_evaluation["baseline"]["size_band_thresholds"],
                        "tail_summary": group_evaluation["baseline"]["tail_summary"],
                    },
                    "segmented": {
                        "cross_validation": group_evaluation["segmented"]["cross_validation"],
                        "holdout_metrics": group_evaluation["segmented"]["holdout_metrics"],
                        "team_error_summary": group_evaluation["segmented"]["team_error_summary"],
                        "sprint_size_error_summary": group_evaluation["segmented"]["sprint_size_error_summary"],
                        "small_vs_large_error_summary": group_evaluation["segmented"][
                            "small_vs_large_error_summary"
                        ],
                        "size_band_thresholds": group_evaluation["segmented"]["size_band_thresholds"],
                        "tail_summary": group_evaluation["segmented"]["tail_summary"],
                    },
                    "comparison_summary": group_evaluation["comparison_summary"],
                },
                "time_based": {
                    "training_rows": time_evaluation["train_rows"],
                    "holdout_rows": time_evaluation["test_rows"],
                    "baseline": {
                        "cross_validation": time_evaluation["baseline"]["cross_validation"],
                        "holdout_metrics": time_evaluation["baseline"]["holdout_metrics"],
                        "team_error_summary": time_evaluation["baseline"]["team_error_summary"],
                        "sprint_size_error_summary": time_evaluation["baseline"]["sprint_size_error_summary"],
                        "small_vs_large_error_summary": time_evaluation["baseline"][
                            "small_vs_large_error_summary"
                        ],
                        "size_band_thresholds": time_evaluation["baseline"]["size_band_thresholds"],
                        "tail_summary": time_evaluation["baseline"]["tail_summary"],
                    },
                    "segmented": {
                        "cross_validation": time_evaluation["segmented"]["cross_validation"],
                        "holdout_metrics": time_evaluation["segmented"]["holdout_metrics"],
                        "team_error_summary": time_evaluation["segmented"]["team_error_summary"],
                        "sprint_size_error_summary": time_evaluation["segmented"]["sprint_size_error_summary"],
                        "small_vs_large_error_summary": time_evaluation["segmented"][
                            "small_vs_large_error_summary"
                        ],
                        "size_band_thresholds": time_evaluation["segmented"]["size_band_thresholds"],
                        "tail_summary": time_evaluation["segmented"]["tail_summary"],
                    },
                    "comparison_summary": time_evaluation["comparison_summary"],
                },
            },
            "feature_outlier_caps": getattr(feature_capper, "bounds_", {}) if feature_capper else {},
            "target_transform": "log1p_only_no_target_clipping",
            "sample_weight_strategy": {
                "q80_weight": 2.0,
                "q95_weight": 4.0,
            },
            "segmentation_summary": getattr(final_model, "segment_summary_", None),
        }
        save_model_artifacts(
            model_dir=model_dir,
            model_key=model_key,
            model=final_model,
            feature_importance=feature_importance,
            scenario_frames=scenario_frames,
            metrics_payload=metrics_payload,
        )
        segmented_predictions_by_model["group_holdout"][model_key] = group_evaluation["segmented"][
            "predictions"
        ]
        segmented_predictions_by_model["time_based"][model_key] = time_evaluation["segmented"][
            "predictions"
        ]
        model_summaries.append(
            {
                "model_key": model_key,
                "model_name": spec.name,
                "artifact_dir": str(model_dir),
                "group_holdout_baseline": group_evaluation["baseline"]["holdout_metrics"],
                "group_holdout_segmented": group_evaluation["segmented"]["holdout_metrics"],
                "time_based_baseline": time_evaluation["baseline"]["holdout_metrics"],
                "time_based_segmented": time_evaluation["segmented"]["holdout_metrics"],
            }
        )

    group_mid_vs_initial_df, group_mid_vs_initial_summary = build_mid_vs_initial_comparison(
        segmented_predictions_by_model["group_holdout"]["initial"],
        segmented_predictions_by_model["group_holdout"]["mid"],
    )
    time_mid_vs_initial_df, time_mid_vs_initial_summary = build_mid_vs_initial_comparison(
        segmented_predictions_by_model["time_based"]["initial"],
        segmented_predictions_by_model["time_based"]["mid"],
    )
    group_mid_vs_initial_df.to_csv(
        output_dir / "group_holdout_mid_vs_initial_comparison.csv", index=False
    )
    time_mid_vs_initial_df.to_csv(
        output_dir / "time_based_mid_vs_initial_comparison.csv", index=False
    )
    (output_dir / "mid_vs_initial_summary.json").write_text(
        json.dumps(
            {
                "group_holdout": group_mid_vs_initial_summary,
                "time_based": time_mid_vs_initial_summary,
            },
            indent=2,
        )
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
        "mid_vs_initial": {
            "group_holdout": group_mid_vs_initial_summary,
            "time_based": time_mid_vs_initial_summary,
        },
    }
    (output_dir / "training_summary.json").write_text(json.dumps(summary_payload, indent=2))

    print(f"Training complete. Artifacts saved to: {output_dir.resolve()}")
    print(f"Held-out teams: {', '.join(group_split_report['held_out_teams'])}")
    for summary in model_summaries:
        print(
            f"{summary['model_name']} -> "
            f"group segmented R^2: {summary['group_holdout_segmented']['r2']:.4f}, "
            f"time segmented R^2: {summary['time_based_segmented']['r2']:.4f}"
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
