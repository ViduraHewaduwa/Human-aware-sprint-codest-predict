#!/usr/bin/env python3
"""Train and use a sprint cost prediction model."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor


TARGET_COLUMN = "actual_sprint_cost"
CATEGORICAL_COLUMNS = ["team_id", "sprint_id"]
DEFAULT_DATA_PATH = Path("Dataset/final_agile_dataset.csv")
DEFAULT_OUTPUT_DIR = Path("artifacts/sprint_cost_model")


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

    def get_feature_names_out(self, input_features: list[str] | None = None) -> np.ndarray:
        base_features = input_features if input_features is not None else self.feature_names_in_
        return np.array(list(base_features) + self.engineered_columns, dtype=object)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train or use a model that predicts actual sprint cost."
    )
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train", help="Train the sprint cost model.")
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
        help="Number of cross-validation folds used for model selection. Default: 5",
    )

    predict_parser = subparsers.add_parser("predict", help="Generate cost predictions from a CSV.")
    predict_parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "best_model.joblib",
        help=f"Path to the saved model artifact. Default: {DEFAULT_OUTPUT_DIR / 'best_model.joblib'}",
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
    return args


def load_dataset(data_path: Path, require_target: bool = True) -> pd.DataFrame:
    if not data_path.exists():
        if data_path.name == "your_new_data.csv":
            raise FileNotFoundError(
                "Input file 'your_new_data.csv' does not exist. That name is a placeholder.\n"
                "Use a real CSV path instead, for example:\n"
                "  python3 train_sprint_cost_model.py predict "
                "--input-path Dataset/final_agile_dataset.csv "
                "--output-path predictions.csv"
            )
        raise FileNotFoundError(
            f"Dataset not found: {data_path}\n"
            "Provide a valid CSV path for training or prediction."
        )

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


def build_candidate_pipelines(feature_columns: list[str], random_state: int) -> dict[str, Pipeline]:
    feature_engineer = SprintFeatureEngineer()
    all_columns = feature_columns + SprintFeatureEngineer.engineered_columns
    preprocessor = build_preprocessor(all_columns)

    return {
        "random_forest": Pipeline(
            steps=[
                ("feature_engineering", clone(feature_engineer)),
                ("preprocessor", clone(preprocessor)),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=400,
                        min_samples_leaf=2,
                        min_samples_split=4,
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "xgboost": Pipeline(
            steps=[
                ("feature_engineering", clone(feature_engineer)),
                ("preprocessor", clone(preprocessor)),
                (
                    "model",
                    XGBRegressor(
                        n_estimators=400,
                        max_depth=6,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        reg_alpha=0.0,
                        reg_lambda=1.0,
                        objective="reg:squarederror",
                        random_state=random_state,
                        n_jobs=-1,
                        tree_method="hist",
                    ),
                ),
            ]
        ),
    }


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


def evaluate_candidates(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    candidate_pipelines: dict[str, Pipeline],
    cv_folds: int,
) -> tuple[str, dict[str, dict[str, float]]]:
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
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
            },
            n_jobs=1,
        )
        candidate_scores[name] = {
            "cv_rmse_mean": float(-scores["test_rmse"].mean()),
            "cv_rmse_std": float(scores["test_rmse"].std()),
            "cv_mae_mean": float(-scores["test_mae"].mean()),
            "cv_r2_mean": float(scores["test_r2"].mean()),
        }

    best_model_name = min(candidate_scores, key=lambda name: candidate_scores[name]["cv_rmse_mean"])
    return best_model_name, candidate_scores


def get_feature_importance_table(model_pipeline: Pipeline) -> pd.DataFrame:
    model = model_pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return pd.DataFrame(columns=["feature", "importance"])

    feature_names = model_pipeline.named_steps["preprocessor"].get_feature_names_out()
    importances = model.feature_importances_
    feature_table = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=False)
    return feature_table.reset_index(drop=True)


def train_model(
    data_path: Path,
    output_dir: Path,
    test_size: float,
    random_state: int,
    cv_folds: int,
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

    candidate_pipelines = build_candidate_pipelines(list(X.columns), random_state=random_state)
    best_model_name, candidate_scores = evaluate_candidates(
        X_train=X_train,
        y_train=y_train,
        candidate_pipelines=candidate_pipelines,
        cv_folds=cv_folds,
    )

    best_pipeline = candidate_pipelines[best_model_name]
    best_pipeline.fit(X_train, y_train)
    test_predictions = best_pipeline.predict(X_test)
    test_metrics = compute_metrics(y_test, test_predictions)
    feature_importance = get_feature_importance_table(best_pipeline)

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "best_model.joblib"
    metrics_path = output_dir / "metrics.json"
    holdout_predictions_path = output_dir / "test_predictions.csv"
    feature_importance_path = output_dir / "feature_importance.csv"

    joblib.dump(best_pipeline, model_path)

    metrics_payload = {
        "data_path": str(data_path),
        "rows": int(df.shape[0]),
        "columns": df.columns.tolist(),
        "target": TARGET_COLUMN,
        "train_rows": int(X_train.shape[0]),
        "test_rows": int(X_test.shape[0]),
        "selected_model": best_model_name,
        "candidate_scores": candidate_scores,
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

    if not feature_importance.empty:
        feature_importance.to_csv(feature_importance_path, index=False)

    strongest_predictor = (
        df.select_dtypes(include="number")
        .corr(numeric_only=True)[TARGET_COLUMN]
        .drop(labels=[TARGET_COLUMN])
        .abs()
        .sort_values(ascending=False)
        .index[0]
    )

    print(f"Training complete. Best model: {best_model_name}")
    print(
        f"Holdout metrics -> MAE: {test_metrics.mae:,.2f}, "
        f"RMSE: {test_metrics.rmse:,.2f}, R^2: {test_metrics.r2:.4f}, "
        f"MAPE: {test_metrics.mape:.2f}%"
    )
    print(f"Artifacts saved to: {output_dir.resolve()}")
    print(f"Most target-correlated numeric feature: {strongest_predictor}")


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
    )


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
    except KeyboardInterrupt:
        raise SystemExit("\nCancelled by user.") from None
