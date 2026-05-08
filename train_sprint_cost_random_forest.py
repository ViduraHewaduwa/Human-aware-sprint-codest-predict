#!/usr/bin/env python3
"""Train and use a Random Forest model for sprint cost prediction."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COLUMN = "actual_sprint_cost"
CATEGORICAL_COLUMNS = ["team_id", "sprint_id"]
DEFAULT_DATA_PATH = Path("Dataset/final_agile_dataset.csv")
DEFAULT_OUTPUT_DIR = Path("artifacts/sprint_cost_random_forest")


@dataclass
class EvaluationMetrics:
    mae: float
    rmse: float
    r2: float
    mape: float


class SprintFeatureEngineer(BaseEstimator, TransformerMixin):
    """Create ratio and workload features from raw sprint data."""

    engineered_columns = [
        "story_point_completion_ratio",
        "effort_per_task",
        "effort_per_developer",
        "availability_adjusted_capacity",
        "overtime_per_person",
        "planned_points_per_task",
        "completed_points_per_task",
    ]

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "SprintFeatureEngineer":
        self.feature_names_in_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        data = X.copy()
        planned = data["planned_story_points"].replace(0, np.nan)
        tasks = data["total_tasks"].replace(0, np.nan)
        team_size = data["team_size"].replace(0, np.nan)

        data["story_point_completion_ratio"] = data["completed_story_points"] / planned
        data["effort_per_task"] = data["total_effort_minutes"] / tasks
        data["effort_per_developer"] = data["total_effort_minutes"] / team_size
        data["availability_adjusted_capacity"] = (
            data["team_size"] * data["developer_availability_rate"]
        )
        data["overtime_per_person"] = data["overtime_hours_total"] / team_size
        data["planned_points_per_task"] = data["planned_story_points"] / tasks
        data["completed_points_per_task"] = data["completed_story_points"] / tasks

        return data.replace([np.inf, -np.inf], np.nan)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train or use a Random Forest model that predicts actual sprint cost."
    )
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train", help="Train the Random Forest sprint cost model.")
    train_parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=f"Path to the training CSV file. Default: {DEFAULT_DATA_PATH}",
    )
    train_parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for trained artifacts. Default: {DEFAULT_OUTPUT_DIR}",
    )
    train_parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of rows reserved for test evaluation. Default: 0.2",
    )
    train_parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used for splitting and model training. Default: 42",
    )
    train_parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds used for evaluation. Default: 5",
    )
    train_parser.add_argument(
        "--n-estimators",
        type=int,
        default=400,
        help="Number of trees in the random forest. Default: 400",
    )
    train_parser.add_argument(
        "--min-samples-leaf",
        type=int,
        default=2,
        help="Minimum samples required at a leaf node. Default: 2",
    )
    train_parser.add_argument(
        "--min-samples-split",
        type=int,
        default=4,
        help="Minimum samples required to split an internal node. Default: 4",
    )

    predict_parser = subparsers.add_parser("predict", help="Generate cost predictions from a CSV.")
    predict_parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "random_forest_model.joblib",
        help=(
            "Path to the saved model artifact. "
            f"Default: {DEFAULT_OUTPUT_DIR / 'random_forest_model.joblib'}"
        ),
    )
    predict_parser.add_argument(
        "--input-path",
        type=Path,
        required=True,
        help="Path to a CSV containing feature columns for inference.",
    )
    predict_parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "predictions.csv",
        help=f"Path for the predictions CSV. Default: {DEFAULT_OUTPUT_DIR / 'predictions.csv'}",
    )

    args = parser.parse_args()
    if args.command is None:
        args.command = "train"
        args.data_path = DEFAULT_DATA_PATH
        args.output_dir = DEFAULT_OUTPUT_DIR
        args.test_size = 0.2
        args.random_state = 42
        args.cv_folds = 5
        args.n_estimators = 400
        args.min_samples_leaf = 2
        args.min_samples_split = 4
    return args


def load_dataset(data_path: Path, require_target: bool = True) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    if require_target and TARGET_COLUMN not in df.columns:
        raise ValueError(f"Expected target column '{TARGET_COLUMN}' in {data_path}")
    return df


def build_preprocessor(feature_columns: list[str]) -> ColumnTransformer:
    numeric_columns = [col for col in feature_columns if col not in CATEGORICAL_COLUMNS]
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
                numeric_columns,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_COLUMNS,
            ),
        ]
    )


def build_pipeline(
    feature_columns: list[str],
    random_state: int,
    n_estimators: int,
    min_samples_leaf: int,
    min_samples_split: int,
) -> Pipeline:
    all_columns = feature_columns + SprintFeatureEngineer.engineered_columns
    return Pipeline(
        steps=[
            ("feature_engineering", SprintFeatureEngineer()),
            ("preprocessor", build_preprocessor(all_columns)),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=n_estimators,
                    min_samples_leaf=min_samples_leaf,
                    min_samples_split=min_samples_split,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def make_regression_strata(y: pd.Series, bins: int = 10) -> pd.Series | None:
    unique_values = y.nunique()
    if unique_values < 2:
        return None

    bin_count = min(bins, unique_values)
    strata = pd.qcut(y.rank(method="first"), q=bin_count, labels=False, duplicates="drop")
    if pd.Series(strata).nunique() < 2:
        return None
    return pd.Series(strata)


def safe_mape(y_true: pd.Series, y_pred: np.ndarray) -> float:
    denominator = np.clip(np.abs(np.asarray(y_true, dtype=float)), 1.0, None)
    return float(np.mean(np.abs((np.asarray(y_true) - y_pred) / denominator)) * 100)


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> EvaluationMetrics:
    return EvaluationMetrics(
        mae=float(mean_absolute_error(y_true, y_pred)),
        rmse=float(root_mean_squared_error(y_true, y_pred)),
        r2=float(r2_score(y_true, y_pred)),
        mape=safe_mape(y_true, y_pred),
    )


def evaluate_model(X_train: pd.DataFrame, y_train: pd.Series, pipeline: Pipeline, cv_folds: int) -> dict[str, float]:
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring={
            "rmse": "neg_root_mean_squared_error",
            "mae": "neg_mean_absolute_error",
            "r2": "r2",
        },
        n_jobs=1,
    )
    return {
        "cv_rmse_mean": float(-scores["test_rmse"].mean()),
        "cv_rmse_std": float(scores["test_rmse"].std()),
        "cv_mae_mean": float(-scores["test_mae"].mean()),
        "cv_r2_mean": float(scores["test_r2"].mean()),
    }


def get_feature_importance_table(model_pipeline: Pipeline) -> pd.DataFrame:
    model = model_pipeline.named_steps["model"]
    feature_names = model_pipeline.named_steps["preprocessor"].get_feature_names_out()
    feature_table = pd.DataFrame(
        {"feature": feature_names, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    return feature_table.reset_index(drop=True)


def train_model(
    data_path: Path,
    output_dir: Path,
    test_size: float,
    random_state: int,
    cv_folds: int,
    n_estimators: int,
    min_samples_leaf: int,
    min_samples_split: int,
) -> None:
    df = load_dataset(data_path, require_target=True)
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    strata = make_regression_strata(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=strata,
    )

    pipeline = build_pipeline(
        feature_columns=list(X.columns),
        random_state=random_state,
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
    )
    cv_metrics = evaluate_model(X_train=X_train, y_train=y_train, pipeline=pipeline, cv_folds=cv_folds)

    pipeline.fit(X_train, y_train)
    test_predictions = pipeline.predict(X_test)
    test_metrics = compute_metrics(y_test, test_predictions)
    feature_importance = get_feature_importance_table(pipeline)

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "random_forest_model.joblib"
    metrics_path = output_dir / "metrics.json"
    holdout_predictions_path = output_dir / "test_predictions.csv"
    feature_importance_path = output_dir / "feature_importance.csv"

    joblib.dump(pipeline, model_path)

    metrics_payload = {
        "data_path": str(data_path),
        "rows": int(df.shape[0]),
        "columns": df.columns.tolist(),
        "target": TARGET_COLUMN,
        "train_rows": int(X_train.shape[0]),
        "test_rows": int(X_test.shape[0]),
        "selected_model": "random_forest",
        "random_forest_params": {
            "n_estimators": n_estimators,
            "min_samples_leaf": min_samples_leaf,
            "min_samples_split": min_samples_split,
            "random_state": random_state,
        },
        "cross_validation": cv_metrics,
        "test_metrics": asdict(test_metrics),
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2))

    prediction_frame = X_test.copy()
    prediction_frame["actual_sprint_cost"] = y_test.values
    prediction_frame["predicted_sprint_cost"] = test_predictions
    prediction_frame["absolute_error"] = np.abs(
        prediction_frame["actual_sprint_cost"] - prediction_frame["predicted_sprint_cost"]
    )
    prediction_frame.to_csv(holdout_predictions_path, index=False)
    feature_importance.to_csv(feature_importance_path, index=False)

    print("Training complete. Model: random_forest")
    print(
        f"Cross-validation -> RMSE: {cv_metrics['cv_rmse_mean']:,.2f} (+/- {cv_metrics['cv_rmse_std']:,.2f}), "
        f"MAE: {cv_metrics['cv_mae_mean']:,.2f}, R^2: {cv_metrics['cv_r2_mean']:.4f}"
    )
    print(
        f"Holdout metrics -> MAE: {test_metrics.mae:,.2f}, "
        f"RMSE: {test_metrics.rmse:,.2f}, R^2: {test_metrics.r2:.4f}, "
        f"MAPE: {test_metrics.mape:.2f}%"
    )
    print(f"Artifacts saved to: {output_dir.resolve()}")


def predict_with_model(model_path: Path, input_path: Path, output_path: Path) -> None:
    if not model_path.exists():
        raise FileNotFoundError(f"Saved model not found: {model_path}")

    model = joblib.load(model_path)
    df = load_dataset(input_path, require_target=False)
    features = df.drop(columns=[TARGET_COLUMN], errors="ignore")
    expected_columns = list(getattr(model, "feature_names_in_", []))
    missing_columns = [column for column in expected_columns if column not in features.columns]
    if missing_columns:
        raise ValueError(
            "Prediction input is missing required columns: "
            + ", ".join(missing_columns)
        )

    predictions = model.predict(features)
    result = df.copy()
    result["predicted_sprint_cost"] = predictions
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    print(f"Predictions saved to: {output_path.resolve()}")
    print(result.head(5).to_string(index=False))


def main() -> None:
    args = parse_args()
    if args.command == "predict":
        predict_with_model(
            model_path=args.model_path,
            input_path=args.input_path,
            output_path=args.output_path,
        )
        return

    train_model(
        data_path=args.data_path,
        output_dir=args.output_dir,
        test_size=args.test_size,
        random_state=args.random_state,
        cv_folds=args.cv_folds,
        n_estimators=args.n_estimators,
        min_samples_leaf=args.min_samples_leaf,
        min_samples_split=args.min_samples_split,
    )


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
    except KeyboardInterrupt:
        raise SystemExit("\nCancelled by user.") from None
