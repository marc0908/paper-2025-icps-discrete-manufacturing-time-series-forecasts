#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Post-Tuning Evaluation of Best Models (using eval1_single_lookahead)
--------------------------------------------------------------------
Automatically finds top-performing trials from Ray Tune,
loads their newest checkpoints, and evaluates them with
the single-lookahead evaluation pipeline.
"""

import traceback
from ..core import common
from .eval1_single_lookahead import eval_model
import matplotlib
from typing import Dict, Optional, List

import os
import glob
import pandas as pd

def load_all_experiments(experiment_dir, metric="val_loss", mode="min"):
    """
    Load all Ray Tune trials by scanning subfolders for result.json files,
    combining their contents into a single DataFrame.
    """
    all_rows = []

    # iterate over all trial subfolders
    for trial_dir in glob.glob(os.path.join(experiment_dir, "*")):
        result_path = os.path.join(trial_dir, "result.json")
        if not os.path.exists(result_path):
            continue

        try:
            # Read result.json — may have multiple lines (each line = 1 iteration)
            df = pd.read_json(result_path, lines=True)
            if df.empty:
                continue
            # Keep only the last reported metric (latest training step)
            last_row = df.tail(1).copy()
            last_row["trial_id"] = os.path.basename(trial_dir).split("_")[1]
            last_row["trial_dir"] = trial_dir
            all_rows.append(last_row)
        except Exception as e:
            print(f"⚠️ Could not read {result_path}: {e}")

    if not all_rows:
        raise RuntimeError(f"No valid result.json files found in {experiment_dir}")

    combined_df = pd.concat(all_rows, ignore_index=True)
    combined_df = combined_df.sort_values(metric, ascending=(mode == "min"))
    print(f"✅ Loaded {len(combined_df)} total trials from {experiment_dir}")
    return combined_df

def find_best_trials(experiment_dir, metric="val_loss", mode="min", top_k=3):
    """Load Ray Tune results and return the best trial checkpoints (Ray >=2.9 compatible)."""
    df = load_all_experiments(experiment_dir).sort_values(metric, ascending=(mode == "min"))
    top_trials = df.head(top_k)

    best_trials = []
    for _, row in top_trials.iterrows():
        trial_id = row['trial_id']
        trial_dir = row['trial_dir']

        # Find checkpoint folders
        ckpt_dirs = [
            os.path.join(trial_dir, d)
            for d in os.listdir(trial_dir)
            if d.startswith("checkpoint_") and os.path.isdir(os.path.join(trial_dir, d))
        ]
        if not ckpt_dirs:
            print(f"[⚠️] No checkpoints found in {trial_dir}")
            continue

        # Pick the newest checkpoint
        latest_ckpt_dir = max(ckpt_dirs, key=os.path.getmtime)

        # Find the actual checkpoint file inside (e.g. checkpoint.pt or similar)
        ckpt_files = [
            os.path.join(latest_ckpt_dir, f)
            for f in os.listdir(latest_ckpt_dir)
            if f.endswith(".pt") or f.startswith("checkpoint")
        ]

        if ckpt_files:
            ckpt_path = max(ckpt_files, key=os.path.getmtime)
        else:
            ckpt_path = latest_ckpt_dir  # fallback

        best_trials.append(
            {
                "trial_id": trial_id,
                "config": row.to_dict(),
                "checkpoint": ckpt_path,
            }
        )

    return best_trials

def evaluate_best_models(
        experiment_name: str, 
        data_path: str, 
        top_k: int = 3, 
        n_runs: int = 100,
        experiment_path: Optional[str] = None,
        ):
    """Find and evaluate the best models for a given experiment."""

    if experiment_name != None:
        exp_dir = str(experiment_path)
    else:
        base_dir = os.path.expanduser("~/ray_results")
        exp_dir = os.path.join(base_dir, experiment_name)
        
    if not os.path.exists(exp_dir):
            raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")
    
    print(f"\n=== Evaluating Best Models for Experiment: {experiment_name} ===")

    # Load dataset once
    data = common.load_csv(data_path)

    # Find best trials by val_loss
    best_trials = find_best_trials(exp_dir, top_k=top_k)

    results = {}

    for i, trial in enumerate(best_trials, start=1):
        model_path = trial["checkpoint"]
        if not model_path:
            print(f"[⚠️] No checkpoint found for trial {trial['trial_id']}")
            continue

        print(f"\n▶ Evaluating top-{i} trial {trial['trial_id']}")
        print(f"  Checkpoint: {model_path}")
        print(f"  Config: d_model={trial['config'].get('config/d_model')}, "
              f"d_ff={trial['config'].get('config/d_ff')}, "
              f"dropout={trial['config'].get('config/dropout', 0):.4f}")

        try:
            result_row = eval_model(
                f"Trial_{i}", model_path, data, n_runs=n_runs
            )
            results[f"Trial_{i}"] = result_row
        except Exception as e:
            print(f"⚠️ Evaluation failed for {model_path}: {e}")
            traceback.print_exc()

    print("\n=== Evaluation Summary ===")
    if args.plain:
        eval_common.print_plain_table(data, results)
    else:
        eval_common.print_latex_table(data, results)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate best models from Ray Tune results")
    parser.add_argument("--experiment", required=True, help="Name of the experiment (e.g., TimeXer-Tuning)")
    parser.add_argument("--experiment-path", required=False, help="Path to Experiment (autoamtically found through experiment name)")
    parser.add_argument("--data-path", required=True, help="Path to dataset CSV or ZIP")
    parser.add_argument("--top-k", type=int, default=3, help="Number of best models to evaluate")
    parser.add_argument("--n-runs", type=int, default=100, help="Number of random test runs for evaluation")
    parser.add_argument("--enable-latex", action="store_true", help="Use LaTeX for plot rendering (requires local LaTeX install)")
    parser.add_argument(
        "--plain",
        action="store_true",
        help="Print a plain text table instead of LaTeX table (useful for clusters)."
    )

    args = parser.parse_args()

    import paper_icps.evaluation.eval_common as eval_common

    if args.enable_latex:
        print("🧩 Enabling LaTeX text rendering for plots.")
        matplotlib.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
        })
        eval_common.USE_LATEX = True

        eval_common.name_dict = eval_common._init_label_dicts()
        eval_common.name_est_dict = {
            k: v.replace("$", "$\\hat{")[:-1] + "}$"
            for k, v in eval_common.name_dict.items()
        }
    else:
        print("🖥️  Using mathtext (no LaTeX dependency).")
        matplotlib.rcParams.update({
            "text.usetex": False,
            "font.family": "sans-serif",
            "mathtext.fontset": "stix",
        })
        eval_common.USE_LATEX = False

        eval_common.name_dict = eval_common._init_label_dicts()
        eval_common.name_est_dict = {
            k: v.replace("$", "$\\hat{")[:-1] + "}$"
            for k, v in eval_common.name_dict.items()
        }

    evaluate_best_models(
        experiment_name=args.experiment,
        data_path=args.data_path,
        top_k=args.top_k,
        n_runs=args.n_runs,
        experiment_path=args.experiment_path
    )