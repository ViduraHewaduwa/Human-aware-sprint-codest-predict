#!/usr/bin/env python3
"""Evaluate saved sprint cost models on real holdout data."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import joblib
import pandas as pd

from train_backend_sprint_cost_model import (
    DEFAULT_DATA_PATH,
    DEFAULT_OUTPUT_DIR,
    IDENTIFIER_COLUMNS,
    StageFeatureEngineer,
    STAGE_CONFIG,
    TARGET_COLUMN,
    build_stage_target,
    compute_metrics,
    load_dataset,
    split_dataset,
)


# Support loading joblib artifacts saved when the training script ran as __main__.
sys.modules["__main__"].StageFeatureEngineer = StageFeatureEngineer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate saved sprint cost models on real holdout rows."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=f"Dataset used for evaluation. Default: {DEFAULT_DATA_PATH}",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory containing stage model artifacts. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "test_reports",
        help=f"Directory for evaluation reports. Default: {DEFAULT_OUTPUT_DIR / 'test_reports'}",
    )
    parser.add_argument(
        "--stage",
        choices=["initial", "mid", "both"],
        default="both",
        help="Which saved model stage to evaluate. Default: both",
    )
    parser.add_argument(
        "--split-mode",
        choices=["time", "random"],
        default="time",
        help="Use the same holdout style as training. Default: time",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction reserved for holdout evaluation. Default: 0.2",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Seed used when split-mode=random. Default: 42",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=25,
        help="Number of real holdout rows to export per stage. Default: 25",
    )
    return parser.parse_args()


def evaluate_stage(
    df: pd.DataFrame,
    artifacts_dir: Path,
    report_dir: Path,
    stage: str,
    split_mode: str,
    test_size: float,
    random_state: int,
    sample_size: int,
) -> dict[str, object]:
    stage_dir = artifacts_dir / f"{stage}_model"
    model_path = stage_dir / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Saved model not found: {model_path}")

    X_train, X_test, split_frame, split_index = split_dataset(
        df=df,
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
    del X_train, y_train

    actual_cost_series = split_frame.loc[X_test.index, TARGET_COLUMN].copy()
    model = joblib.load(model_path)
    predictions = model.predict(X_test)

    evaluation_target_metrics = compute_metrics(y_test, predictions)
    actual_cost_reference_metrics = compute_metrics(actual_cost_series, predictions)

    sample_frame = split_frame.loc[
        X_test.index,
        IDENTIFIER_COLUMNS + STAGE_CONFIG[stage]["raw_features"],
    ].copy()
    sample_frame = sample_frame.loc[:, ~sample_frame.columns.duplicated()].copy()
    sample_frame["target_name"] = target_info.target_name
    sample_frame["actual_target_value"] = y_test.values
    sample_frame["predicted_value"] = predictions
    sample_frame["actual_sprint_cost_reference"] = actual_cost_series.values
    sample_frame["absolute_target_error"] = (
        sample_frame["actual_target_value"] - sample_frame["predicted_value"]
    ).abs()
    sample_frame["absolute_actual_cost_error"] = (
        sample_frame["actual_sprint_cost_reference"] - sample_frame["predicted_value"]
    ).abs()
    sample_frame = sample_frame.sort_values("sprint_number").head(sample_size).reset_index(drop=True)

    report_dir.mkdir(parents=True, exist_ok=True)
    sample_path = report_dir / f"{stage}_real_data_sample.csv"
    sample_frame.to_csv(sample_path, index=False)

    return {
        "stage": stage,
        "model_path": str(model_path),
        "split_mode": split_mode,
        "holdout_rows": int(len(X_test)),
        "target_name": target_info.target_name,
        "target_definition": target_info.target_definition,
        "target_scaling_factor": target_info.scaling_factor,
        "evaluation_target_metrics": asdict(evaluation_target_metrics),
        "actual_cost_reference_metrics": asdict(actual_cost_reference_metrics),
        "real_data_sample_path": str(sample_path),
        "note": (
            "For the initial stage, actual_cost_reference_metrics are diagnostic only because "
            "the saved model predicts initial_planning_cost rather than actual_sprint_cost."
            if stage == "initial"
            else "For the mid stage, evaluation_target_metrics and actual_cost_reference_metrics refer to the same real target."
        ),
    }


def main() -> None:
    args = parse_args()
    df = load_dataset(args.data_path)
    stages = ["initial", "mid"] if args.stage == "both" else [args.stage]

    stage_reports = []
    for stage in stages:
        report = evaluate_stage(
            df=df,
            artifacts_dir=args.artifacts_dir,
            report_dir=args.report_dir,
            stage=stage,
            split_mode=args.split_mode,
            test_size=args.test_size,
            random_state=args.random_state,
            sample_size=args.sample_size,
        )
        stage_reports.append(report)
        target_metrics = report["evaluation_target_metrics"]
        actual_metrics = report["actual_cost_reference_metrics"]
        print(
            f"[{stage}] target={report['target_name']} "
            f"target_R^2={target_metrics['r2']:.4f} "
            f"target_MAPE={target_metrics['mape']:.2f}% "
            f"actual_cost_R^2={actual_metrics['r2']:.4f} "
            f"actual_cost_MAPE={actual_metrics['mape']:.2f}%"
        )

    summary = {
        "data_path": str(args.data_path),
        "artifacts_dir": str(args.artifacts_dir),
        "split_mode": args.split_mode,
        "test_size": args.test_size,
        "random_state": args.random_state,
        "stage_reports": stage_reports,
    }
    args.report_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.report_dir / "real_data_evaluation.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved evaluation summary to: {summary_path.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
    except KeyboardInterrupt:
        raise SystemExit("\nCancelled by user.") from None
