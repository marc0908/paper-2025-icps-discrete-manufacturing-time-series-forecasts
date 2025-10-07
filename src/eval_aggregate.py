#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregate evaluation logs into CI-friendly CSV and JSON summaries.

Usage:
  python src/eval_aggregate.py --logs-dir <path> --out-dir <path>

Outputs:
  - unified_results.csv: concatenated evaluation rows
  - summary.json: per-model best metrics and overall best by primary metric
"""
import argparse
import json
import os
import sys
from typing import Dict, Any

import pandas as pd


def aggregate_logs(logs_dir: str) -> pd.DataFrame:
    """Read all CSV/tar.gz logs under logs_dir and return a single DataFrame."""
    # Lazy import TFB helpers if available
    try:
        from ts_benchmark.recording import find_record_files, read_record_file
    except Exception:
        # Fallback: simple CSV concatenation
        file_list = []
        for root, _, files in os.walk(logs_dir):
            for f in files:
                if f.endswith(".csv"):
                    file_list.append(os.path.join(root, f))
        if not file_list:
            raise FileNotFoundError(f"No evaluation logs found in {logs_dir}")
        frames = [pd.read_csv(p) for p in file_list]
        return pd.concat(frames, axis=0, ignore_index=True)

    files = find_record_files(logs_dir)
    if not files:
        raise FileNotFoundError(f"No evaluation logs found in {logs_dir}")
    frames = [read_record_file(p) for p in files]
    return pd.concat(frames, axis=0, ignore_index=True)


def normalize_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure consistent metric column names for CI consumption.

    Expected columns commonly include: "metric", "series", "model_name", and one or more
    numeric metric columns depending on the benchmark. We standardize by:
      - Creating a primary_metric column equal to MAE if present, else MSE/RMSE/SMAPE fallback.
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    # Common metric preferences
    preferred = [
        "MAE", "mae", "MeanAbsoluteError",
        "MSE", "mse", "RMSE", "rmse", "SMAPE", "smape",
    ]
    primary = None
    for name in preferred:
        if name in df.columns:
            primary = name
            break
    # fallback to first numeric metric if unknown
    if primary is None:
        candidates = [c for c in numeric_cols if c not in ("seed",)]
        primary = candidates[0] if candidates else None
    if primary is not None:
        df["primary_metric"] = df[primary]
    else:
        df["primary_metric"] = float("nan")

    # Ensure model identification column
    if "model_name" not in df.columns and "model" in df.columns:
        df["model_name"] = df["model"]

    return df


def build_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute per-model best rows and overall best by primary_metric."""
    out: Dict[str, Any] = {}
    if "model_name" not in df.columns:
        return {"error": "model_name column not found"}

    # Best per model (min primary_metric)
    per_model = (
        df.sort_values("primary_metric", ascending=True)
        .groupby("model_name", as_index=False)
        .first()
    )
    overall_best = per_model.sort_values("primary_metric", ascending=True).head(1)

    out["per_model_best"] = per_model.to_dict(orient="records")
    out["overall_best"] = overall_best.to_dict(orient="records")
    return out


def main():
    parser = argparse.ArgumentParser(description="Aggregate evaluation logs")
    parser.add_argument("--logs-dir", type=str, required=True, help="Directory with eval logs")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    logs_dir = os.path.abspath(args.logs_dir)
    out_dir = os.path.abspath(args.out_dir or os.path.join(logs_dir, "..", "aggregated"))
    os.makedirs(out_dir, exist_ok=True)

    df = aggregate_logs(logs_dir)
    df = normalize_metric_columns(df)

    unified_csv = os.path.join(out_dir, "unified_results.csv")
    df.to_csv(unified_csv, index=False)

    summary = build_summary(df)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote {unified_csv}")
    print(f"Wrote {os.path.join(out_dir, 'summary.json')}")


if __name__ == "__main__":
    # Ensure TFB modules can be found if running from repo root
    tfb_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "TFB"))
    if tfb_path not in sys.path:
        sys.path.append(tfb_path)
        os.environ["PYTHONPATH"] = tfb_path
    main()


