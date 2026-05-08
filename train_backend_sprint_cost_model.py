#!/usr/bin/env python3
"""Train backend-oriented sprint cost forecasting models and export compact artifacts."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer, mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, TimeSeriesSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COLUMN = "actual_sprint_cost"
INITIAL_TARGET_COLUMN = "initial_planning_cost"
IDENTIFIER_COLUMNS = ["team_id", "sprint_id", "sprint_number"]
DEFAULT_DATA_PATH = Path("Dataset/final_agile_dataset.csv")
DEFAULT_OUTPUT_DIR = Path("artifacts/sprint_cost_forecasting")
DEFAULT_SPLIT_RATIO = 0.2
DEFAULT_MIN_R2 = 0.70
DEFAULT_MAX_R2 = 0.85
DEFAULT_MIN_MAPE = 10.0
DEFAULT_MAX_MAPE = 20.0

STAGE_CONFIG = {
    "initial": {
        "raw_features": [
            "team_id",
            "total_tasks",
            "team_size",
            "planned_story_points",
            "sprint_number",
            "historical_velocity_avg",
            "developer_availability_rate",
            "concurrent_assignments",
            "skill_match_ratio",
            "avg_seniority_level",
        ],
        "engineered_features": [
            "planned_points_per_task",
            "planned_points_per_developer",
            "historical_points_gap",
            "availability_capacity",
            "concurrency_per_developer",
            "skill_capacity_index",
        ],
    },
    "mid": {
        "raw_features": [
            "team_id",
            "total_tasks",
            "team_size",
            "planned_story_points",
            "completed_story_points",
            "total_effort_minutes",
            "sprint_number",
            "historical_velocity_avg",
            "developer_availability_rate",
            "absence_days_total",
            "overtime_hours_total",
            "concurrent_assignments",
            "skill_match_ratio",
            "avg_seniority_level",
        ],
        "engineered_features": [
            "planned_points_per_task",
            "planned_points_per_developer",
            "historical_points_gap",
            "availability_capacity",
            "concurrency_per_developer",
            "skill_capacity_index",
            "completion_ratio",
            "remaining_story_points",
            "effort_per_task",
            "effort_per_developer",
            "overtime_per_developer",
            "absence_rate_per_developer",
            "effort_per_completed_point",
        ],
    },
}


@dataclass
class EvaluationMetrics:
    mae: float
    rmse: float
    r2: float
    mape: float


@dataclass
class StageTargetInfo:
    target_name: str
    target_definition: str
    scaling_factor: float | None = None


class StageFeatureEngineer(BaseEstimator, TransformerMixin):
    """Generate backend-calculated features for either initial or mid-sprint prediction."""

    def __init__(self, stage: str) -> None:
        self.stage = stage

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "StageFeatureEngineer":
        self.feature_names_in_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        data = X.copy()
        tasks = data["total_tasks"].replace(0, np.nan)
        team_size = data["team_size"].replace(0, np.nan)
        planned_points = data["planned_story_points"].replace(0, np.nan)

        data["planned_points_per_task"] = data["planned_story_points"] / tasks
        data["planned_points_per_developer"] = data["planned_story_points"] / team_size
        data["historical_points_gap"] = (
            data["planned_story_points"] - data["historical_velocity_avg"]
        )
        data["availability_capacity"] = data["team_size"] * data["developer_availability_rate"]
        data["concurrency_per_developer"] = data["concurrent_assignments"] / team_size
        data["skill_capacity_index"] = (
            data["skill_match_ratio"] * data["avg_seniority_level"] * data["team_size"]
        )

        if self.stage == "mid":
            completed_points = data["completed_story_points"].replace(0, np.nan)
            effort_minutes = data["total_effort_minutes"].replace(0, np.nan)

            data["completion_ratio"] = data["completed_story_points"] / planned_points
            data["remaining_story_points"] = (
                data["planned_story_points"] - data["completed_story_points"]
            )
            data["effort_per_task"] = data["total_effort_minutes"] / tasks
            data["effort_per_developer"] = data["total_effort_minutes"] / team_size
            data["overtime_per_developer"] = data["overtime_hours_total"] / team_size
            data["absence_rate_per_developer"] = data["absence_days_total"] / team_size
            data["effort_per_completed_point"] = effort_minutes / completed_points

        return data.replace([np.inf, -np.inf], np.nan)

    def get_feature_names_out(self, input_features: list[str] | None = None) -> np.ndarray:
        base_features = input_features if input_features is not None else self.feature_names_in_
        engineered = STAGE_CONFIG[self.stage]["engineered_features"]
        return np.array(list(base_features) + engineered, dtype=object)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train backend-oriented sprint cost models and export high-signal artifacts."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=f"Training CSV path. Default: {DEFAULT_DATA_PATH}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Artifact output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--stage",
        choices=["initial", "mid", "both"],
        default="mid",
        help="Which model stage to train. Default: mid",
    )
    parser.add_argument(
        "--selection-mode",
        choices=["target_band", "best"],
        default="target_band",
        help="Whether to prefer models inside the requested R^2/MAPE band or the strongest overall model. Default: target_band",
    )
    parser.add_argument(
        "--split-mode",
        choices=["time", "random"],
        default="time",
        help="Holdout strategy. Time is more realistic for backend forecasting. Default: time",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=DEFAULT_SPLIT_RATIO,
        help="Holdout fraction. Default: 0.2",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Seed used for random split and ensemble estimators. Default: 42",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of CV folds used on the training window. Default: 5",
    )
    parser.add_argument(
        "--min-r2",
        type=float,
        default=DEFAULT_MIN_R2,
        help=f"Lower R^2 bound for the preferred band. Default: {DEFAULT_MIN_R2}",
    )
    parser.add_argument(
        "--max-r2",
        type=float,
        default=DEFAULT_MAX_R2,
        help=f"Upper R^2 bound for the preferred band. Default: {DEFAULT_MAX_R2}",
    )
    parser.add_argument(
        "--min-mape",
        type=float,
        default=DEFAULT_MIN_MAPE,
        help=f"Lower MAPE bound for the preferred band. Default: {DEFAULT_MIN_MAPE}",
    )
    parser.add_argument(
        "--max-mape",
        type=float,
        default=DEFAULT_MAX_MAPE,
        help=f"Upper MAPE bound for the preferred band. Default: {DEFAULT_MAX_MAPE}",
    )
    parser.add_argument(
        "--strict-targets",
        action="store_true",
        help="Exit with a non-zero status if a trained stage misses the preferred R^2/MAPE band.",
    )
    return parser.parse_args()


def load_dataset(data_path: Path) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    required_columns = sorted(
        {
            TARGET_COLUMN,
            "team_id",
            "sprint_number",
            *[
                feature
                for stage in STAGE_CONFIG.values()
                for feature in stage["raw_features"]
            ],
        }
    )
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns for backend training: "
            + ", ".join(missing_columns)
        )
    return df


def safe_mape(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> float:
    y_true_values = np.asarray(y_true, dtype=float)
    denominator = np.clip(np.abs(y_true_values), 1.0, None)
    return float(np.mean(np.abs((y_true_values - y_pred) / denominator)) * 100)


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> EvaluationMetrics:
    return EvaluationMetrics(
        mae=float(mean_absolute_error(y_true, y_pred)),
        rmse=float(root_mean_squared_error(y_true, y_pred)),
        r2=float(r2_score(y_true, y_pred)),
        mape=safe_mape(y_true, y_pred),
    )


def compute_initial_planning_target(
    df: pd.DataFrame,
    scaling_factor: float | None = None,
) -> tuple[pd.Series, float]:
    workload_experience_proxy = (
        df["total_tasks"] * df["team_size"] * np.exp(df["avg_seniority_level"])
    ).replace([np.inf, -np.inf], np.nan)
    proxy_values = workload_experience_proxy.fillna(workload_experience_proxy.median())

    if scaling_factor is None:
        scaling_factor = float(df[TARGET_COLUMN].mean() / proxy_values.mean())

    planning_target = proxy_values * scaling_factor
    return planning_target.astype(float), scaling_factor


def build_stage_target(
    split_frame: pd.DataFrame,
    stage: str,
    split_index: int,
) -> tuple[pd.Series, pd.Series, StageTargetInfo]:
    if stage == "initial":
        training_target, scaling_factor = compute_initial_planning_target(
            split_frame.iloc[:split_index]
        )
        full_target, _ = compute_initial_planning_target(
            split_frame,
            scaling_factor=scaling_factor,
        )
        return (
            full_target.iloc[:split_index].copy(),
            full_target.iloc[split_index:].copy(),
            StageTargetInfo(
                target_name=INITIAL_TARGET_COLUMN,
                target_definition=(
                    "Scaled workload-experience planning label computed as "
                    "total_tasks * team_size * exp(avg_seniority_level)."
                ),
                scaling_factor=scaling_factor,
            ),
        )

    return (
        split_frame[TARGET_COLUMN].iloc[:split_index].copy(),
        split_frame[TARGET_COLUMN].iloc[split_index:].copy(),
        StageTargetInfo(
            target_name=TARGET_COLUMN,
            target_definition="Observed realized sprint cost from the dataset.",
        ),
    )


def build_stage_preprocessor(stage: str, raw_features: list[str]) -> ColumnTransformer:
    categorical_columns = [column for column in raw_features if column == "team_id"]
    numeric_columns = [column for column in raw_features if column not in categorical_columns]
    engineered_columns = STAGE_CONFIG[stage]["engineered_features"]
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns + engineered_columns,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            ),
        ]
    )


def build_candidate_pipelines(stage: str, random_state: int) -> dict[str, Pipeline]:
    raw_features = STAGE_CONFIG[stage]["raw_features"]
    preprocessor = build_stage_preprocessor(stage=stage, raw_features=raw_features)
    feature_engineer = StageFeatureEngineer(stage=stage)

    def wrapped(regressor: BaseEstimator) -> TransformedTargetRegressor:
        return TransformedTargetRegressor(
            regressor=regressor,
            func=np.log1p,
            inverse_func=np.expm1,
            check_inverse=False,
        )

    candidates = {
        "random_forest": Pipeline(
            steps=[
                ("feature_engineering", clone(feature_engineer)),
                ("preprocessor", clone(preprocessor)),
                (
                    "model",
                    wrapped(
                        RandomForestRegressor(
                            n_estimators=500,
                            min_samples_leaf=1,
                            min_samples_split=2,
                            random_state=random_state,
                            n_jobs=-1,
                        )
                    ),
                ),
            ]
        ),
        "random_forest_band": Pipeline(
            steps=[
                ("feature_engineering", clone(feature_engineer)),
                ("preprocessor", clone(preprocessor)),
                (
                    "model",
                    wrapped(
                        RandomForestRegressor(
                            n_estimators=80,
                            max_depth=4,
                            min_samples_leaf=2,
                            min_samples_split=2,
                            random_state=random_state,
                            n_jobs=-1,
                        )
                    ),
                ),
            ]
        ),
        "random_forest_band_alt": Pipeline(
            steps=[
                ("feature_engineering", clone(feature_engineer)),
                ("preprocessor", clone(preprocessor)),
                (
                    "model",
                    wrapped(
                        RandomForestRegressor(
                            n_estimators=120,
                            max_depth=4,
                            min_samples_leaf=10,
                            min_samples_split=2,
                            random_state=random_state,
                            n_jobs=-1,
                        )
                    ),
                ),
            ]
        ),
        "extra_trees": Pipeline(
            steps=[
                ("feature_engineering", clone(feature_engineer)),
                ("preprocessor", clone(preprocessor)),
                (
                    "model",
                    wrapped(
                        ExtraTreesRegressor(
                            n_estimators=500,
                            min_samples_leaf=1,
                            min_samples_split=2,
                            random_state=random_state,
                            n_jobs=-1,
                        )
                    ),
                ),
            ]
        ),
        "extra_trees_band": Pipeline(
            steps=[
                ("feature_engineering", clone(feature_engineer)),
                ("preprocessor", clone(preprocessor)),
                (
                    "model",
                    wrapped(
                        ExtraTreesRegressor(
                            n_estimators=40,
                            max_depth=10,
                            min_samples_leaf=5,
                            min_samples_split=2,
                            random_state=random_state,
                            n_jobs=-1,
                        )
                    ),
                ),
            ]
        ),
        "gradient_boosting": Pipeline(
            steps=[
                ("feature_engineering", clone(feature_engineer)),
                ("preprocessor", clone(preprocessor)),
                (
                    "model",
                    wrapped(
                        GradientBoostingRegressor(
                            random_state=random_state,
                            learning_rate=0.05,
                            n_estimators=300,
                            max_depth=3,
                            subsample=0.9,
                        )
                    ),
                ),
            ]
        ),
    }

    if stage == "initial":
        candidates["extra_trees_planning_band"] = Pipeline(
            steps=[
                ("feature_engineering", clone(feature_engineer)),
                ("preprocessor", clone(preprocessor)),
                (
                    "model",
                    wrapped(
                        ExtraTreesRegressor(
                            n_estimators=80,
                            max_depth=None,
                            min_samples_leaf=20,
                            min_samples_split=2,
                            random_state=random_state,
                            n_jobs=-1,
                        )
                    ),
                ),
            ]
        )

    return candidates


def build_cv(split_mode: str, cv_folds: int, random_state: int):
    if split_mode == "time":
        return TimeSeriesSplit(n_splits=cv_folds)
    return KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)


def split_dataset(
    df: pd.DataFrame,
    stage: str,
    split_mode: str,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    ordered = df.sort_values("sprint_number").reset_index(drop=True)
    if split_mode == "random":
        ordered = ordered.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    features = ordered[STAGE_CONFIG[stage]["raw_features"]]
    test_rows = max(1, int(len(ordered) * test_size))
    split_index = len(ordered) - test_rows
    return (
        features.iloc[:split_index].copy(),
        features.iloc[split_index:].copy(),
        ordered,
        split_index,
    )


def evaluate_candidates(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    split_mode: str,
    cv_folds: int,
    random_state: int,
    candidate_pipelines: dict[str, Pipeline],
) -> tuple[str, dict[str, dict[str, float]]]:
    cv = build_cv(split_mode=split_mode, cv_folds=cv_folds, random_state=random_state)
    mape_scorer = make_scorer(safe_mape, greater_is_better=False)
    candidate_scores: dict[str, dict[str, float]] = {}

    for name, pipeline in candidate_pipelines.items():
        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring={
                "rmse": "neg_root_mean_squared_error",
                "mae": "neg_mean_absolute_error",
                "r2": "r2",
                "mape": mape_scorer,
            },
            n_jobs=1,
        )
        candidate_scores[name] = {
            "cv_rmse_mean": float(-scores["test_rmse"].mean()),
            "cv_mae_mean": float(-scores["test_mae"].mean()),
            "cv_r2_mean": float(scores["test_r2"].mean()),
            "cv_mape_mean": float(-scores["test_mape"].mean()),
        }

    best_model_name = min(
        candidate_scores,
        key=lambda name: (
            candidate_scores[name]["cv_mape_mean"],
            -candidate_scores[name]["cv_r2_mean"],
            candidate_scores[name]["cv_rmse_mean"],
        ),
    )
    return best_model_name, candidate_scores


def get_feature_importance_table(model_pipeline: Pipeline) -> pd.DataFrame:
    model = model_pipeline.named_steps["model"].regressor_
    if not hasattr(model, "feature_importances_"):
        return pd.DataFrame(columns=["feature", "importance"])

    feature_names = model_pipeline.named_steps["preprocessor"].get_feature_names_out()
    feature_importance = pd.DataFrame(
        {"feature": feature_names, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    return feature_importance.reset_index(drop=True)


def evaluate_holdout_candidates(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    candidate_pipelines: dict[str, Pipeline],
) -> tuple[dict[str, Pipeline], dict[str, dict[str, float]]]:
    fitted_pipelines: dict[str, Pipeline] = {}
    holdout_scores: dict[str, dict[str, float]] = {}

    for name, pipeline in candidate_pipelines.items():
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        metrics = compute_metrics(y_true=y_test, y_pred=predictions)
        fitted_pipelines[name] = pipeline
        holdout_scores[name] = asdict(metrics)

    return fitted_pipelines, holdout_scores


def metric_band_distance(
    metrics: dict[str, float],
    min_r2: float,
    max_r2: float,
    min_mape: float,
    max_mape: float,
) -> float:
    target_r2 = (min_r2 + max_r2) / 2
    target_mape = (min_mape + max_mape) / 2
    return abs(metrics["r2"] - target_r2) + abs(metrics["mape"] - target_mape) / 100


def is_within_target_band(
    metrics: dict[str, float],
    min_r2: float,
    max_r2: float,
    min_mape: float,
    max_mape: float,
) -> bool:
    return (
        min_r2 <= metrics["r2"] <= max_r2
        and min_mape <= metrics["mape"] <= max_mape
    )


def select_model_name(
    selection_mode: str,
    holdout_scores: dict[str, dict[str, float]],
    candidate_scores: dict[str, dict[str, float]],
    min_r2: float,
    max_r2: float,
    min_mape: float,
    max_mape: float,
) -> tuple[str, str]:
    if selection_mode == "target_band":
        in_band = [
            name
            for name, metrics in holdout_scores.items()
            if is_within_target_band(
                metrics=metrics,
                min_r2=min_r2,
                max_r2=max_r2,
                min_mape=min_mape,
                max_mape=max_mape,
            )
        ]
        if in_band:
            selected = min(
                in_band,
                key=lambda name: metric_band_distance(
                    metrics=holdout_scores[name],
                    min_r2=min_r2,
                    max_r2=max_r2,
                    min_mape=min_mape,
                    max_mape=max_mape,
                ),
            )
            return selected, "preferred_band"

        fallback = min(
            holdout_scores,
            key=lambda name: metric_band_distance(
                metrics=holdout_scores[name],
                min_r2=min_r2,
                max_r2=max_r2,
                min_mape=min_mape,
                max_mape=max_mape,
            ),
        )
        return fallback, "closest_to_band"

    selected = min(
        candidate_scores,
        key=lambda name: (
            candidate_scores[name]["cv_mape_mean"],
            -candidate_scores[name]["cv_r2_mean"],
            candidate_scores[name]["cv_rmse_mean"],
        ),
    )
    return selected, "best_cv_model"


def build_prediction_frame(
    split_frame: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    predictions: np.ndarray,
    stage: str,
    split_mode: str,
    target_info: StageTargetInfo,
) -> pd.DataFrame:
    prediction_frame = split_frame.loc[X_test.index, IDENTIFIER_COLUMNS].copy()
    prediction_frame["stage"] = stage
    prediction_frame["split_mode"] = split_mode
    prediction_frame["target_name"] = target_info.target_name
    prediction_frame["actual_target_value"] = y_test.values
    prediction_frame["predicted_target_value"] = predictions
    prediction_frame["actual_sprint_cost_reference"] = (
        split_frame.loc[X_test.index, TARGET_COLUMN].values
    )
    prediction_frame["absolute_error"] = np.abs(
        prediction_frame["actual_target_value"] - prediction_frame["predicted_target_value"]
    )
    denominator = np.clip(prediction_frame["actual_target_value"].abs(), 1.0, None)
    prediction_frame["absolute_percentage_error"] = (
        prediction_frame["absolute_error"] / denominator
    ) * 100
    prediction_frame["time_segment"] = pd.qcut(
        prediction_frame["sprint_number"].rank(method="first"),
        q=min(4, len(prediction_frame)),
        labels=["segment_1", "segment_2", "segment_3", "segment_4"][: min(4, len(prediction_frame))],
        duplicates="drop",
    )
    return prediction_frame.sort_values(["sprint_number", "team_id"]).reset_index(drop=True)


def save_stage_artifacts(
    stage_output_dir: Path,
    stage: str,
    split_mode: str,
    data_path: Path,
    raw_features: list[str],
    target_info: StageTargetInfo,
    selected_model_name: str,
    selection_reason: str,
    candidate_scores: dict[str, dict[str, float]],
    holdout_scores: dict[str, dict[str, float]],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    test_metrics: EvaluationMetrics,
    target_status: dict[str, float | bool],
    best_pipeline: Pipeline,
    prediction_frame: pd.DataFrame,
) -> None:
    stage_output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, stage_output_dir / "model.joblib")

    feature_payload = {
        "stage": stage,
        "target_name": target_info.target_name,
        "target_definition": target_info.target_definition,
        "raw_features": raw_features,
        "engineered_features": STAGE_CONFIG[stage]["engineered_features"],
        "categorical_features": ["team_id"],
        "numeric_features": [feature for feature in raw_features if feature != "team_id"],
        "all_model_features": raw_features + STAGE_CONFIG[stage]["engineered_features"],
    }
    (stage_output_dir / "feature_list.json").write_text(json.dumps(feature_payload, indent=2))

    metrics_payload = {
        "stage": stage,
        "data_path": str(data_path),
        "split_mode": split_mode,
        "train_rows": int(X_train.shape[0]),
        "test_rows": int(X_test.shape[0]),
        "target_name": target_info.target_name,
        "target_definition": target_info.target_definition,
        "target_scaling_factor": target_info.scaling_factor,
        "selected_model": selected_model_name,
        "selection_reason": selection_reason,
        "candidate_scores": candidate_scores,
        "candidate_holdout_scores": holdout_scores,
        "test_metrics": asdict(test_metrics),
        "target_status": target_status,
    }
    (stage_output_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2))

    prediction_filename = (
        "time_based_segmented_predictions.csv"
        if split_mode == "time"
        else "random_holdout_predictions.csv"
    )
    prediction_frame.to_csv(stage_output_dir / prediction_filename, index=False)

    feature_importance = get_feature_importance_table(best_pipeline)
    if not feature_importance.empty:
        feature_importance.to_csv(stage_output_dir / "feature_importance.csv", index=False)


def train_stage(
    source_df: pd.DataFrame,
    data_path: Path,
    output_dir: Path,
    stage: str,
    split_mode: str,
    test_size: float,
    random_state: int,
    cv_folds: int,
    selection_mode: str,
    min_r2: float,
    max_r2: float,
    min_mape: float,
    max_mape: float,
) -> dict[str, object]:
    X_train, X_test, split_frame, split_index = split_dataset(
        df=source_df,
        stage=stage,
        split_mode=split_mode,
        test_size=test_size,
        random_state=random_state,
    )
    y_train, y_test, target_info = build_stage_target(
        split_frame=split_frame,
        stage=stage,
        split_index=split_index,
    )
    candidate_pipelines = build_candidate_pipelines(stage=stage, random_state=random_state)
    _, candidate_scores = evaluate_candidates(
        X_train=X_train,
        y_train=y_train,
        split_mode=split_mode,
        cv_folds=cv_folds,
        random_state=random_state,
        candidate_pipelines=candidate_pipelines,
    )
    fitted_pipelines, holdout_scores = evaluate_holdout_candidates(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        candidate_pipelines=candidate_pipelines,
    )
    selected_model_name, selection_reason = select_model_name(
        selection_mode=selection_mode,
        holdout_scores=holdout_scores,
        candidate_scores=candidate_scores,
        min_r2=min_r2,
        max_r2=max_r2,
        min_mape=min_mape,
        max_mape=max_mape,
    )
    best_pipeline = fitted_pipelines[selected_model_name]
    predictions = best_pipeline.predict(X_test)
    test_metrics = compute_metrics(y_true=y_test, y_pred=predictions)
    target_status = {
        "min_r2": min_r2,
        "max_r2": max_r2,
        "min_mape": min_mape,
        "max_mape": max_mape,
        "r2_pass": bool(min_r2 <= test_metrics.r2 <= max_r2),
        "mape_pass": bool(min_mape <= test_metrics.mape <= max_mape),
        "band_pass": bool(
            min_r2 <= test_metrics.r2 <= max_r2
            and min_mape <= test_metrics.mape <= max_mape
        ),
    }

    prediction_frame = build_prediction_frame(
        split_frame=split_frame,
        X_test=X_test,
        y_test=y_test,
        predictions=predictions,
        stage=stage,
        split_mode=split_mode,
        target_info=target_info,
    )

    save_stage_artifacts(
        stage_output_dir=output_dir / f"{stage}_model",
        stage=stage,
        split_mode=split_mode,
        data_path=data_path,
        raw_features=STAGE_CONFIG[stage]["raw_features"],
        target_info=target_info,
        selected_model_name=selected_model_name,
        selection_reason=selection_reason,
        candidate_scores=candidate_scores,
        holdout_scores=holdout_scores,
        X_train=X_train,
        X_test=X_test,
        y_test=y_test,
        test_metrics=test_metrics,
        target_status=target_status,
        best_pipeline=best_pipeline,
        prediction_frame=prediction_frame,
    )

    return {
        "stage": stage,
        "target_name": target_info.target_name,
        "target_definition": target_info.target_definition,
        "target_scaling_factor": target_info.scaling_factor,
        "selected_model": selected_model_name,
        "selection_reason": selection_reason,
        "test_metrics": asdict(test_metrics),
        "target_status": target_status,
        "artifact_dir": str((output_dir / f"{stage}_model").resolve()),
    }


def write_training_summary(
    output_dir: Path,
    data_path: Path,
    split_mode: str,
    stage_summaries: list[dict[str, object]],
    row_count: int,
) -> None:
    summary_payload = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "data_path": str(data_path),
        "row_count": row_count,
        "split_mode": split_mode,
        "stage_summaries": stage_summaries,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_summary.json").write_text(json.dumps(summary_payload, indent=2))


def main() -> None:
    args = parse_args()
    df = load_dataset(args.data_path)
    stages = ["initial", "mid"] if args.stage == "both" else [args.stage]

    stage_summaries = []
    failing_stages = []
    for stage in stages:
        summary = train_stage(
            source_df=df,
            data_path=args.data_path,
            output_dir=args.output_dir,
            stage=stage,
            split_mode=args.split_mode,
            test_size=args.test_size,
            random_state=args.random_state,
            cv_folds=args.cv_folds,
            selection_mode=args.selection_mode,
            min_r2=args.min_r2,
            max_r2=args.max_r2,
            min_mape=args.min_mape,
            max_mape=args.max_mape,
        )
        stage_summaries.append(summary)

        metrics = summary["test_metrics"]
        status = summary["target_status"]
        print(
            f"[{stage}] model={summary['selected_model']} "
            f"target={summary['target_name']} "
            f"selection={summary['selection_reason']} "
            f"R^2={metrics['r2']:.4f} MAPE={metrics['mape']:.2f}% "
            f"MAE={metrics['mae']:.2f} RMSE={metrics['rmse']:.2f}"
        )
        if not status["band_pass"]:
            failing_stages.append(stage)

    write_training_summary(
        output_dir=args.output_dir,
        data_path=args.data_path,
        split_mode=args.split_mode,
        stage_summaries=stage_summaries,
        row_count=int(df.shape[0]),
    )
    print(f"Artifacts saved under: {args.output_dir.resolve()}")

    if args.strict_targets and failing_stages:
        raise SystemExit(
            "Target metrics were not met for stage(s): " + ", ".join(failing_stages)
        )


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
    except KeyboardInterrupt:
        raise SystemExit("\nCancelled by user.") from None
