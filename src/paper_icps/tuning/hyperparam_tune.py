#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Improved Hyperparameter Optimization with Advanced Techniques
"""
import argparse
import os
import tempfile
import logging
import json
import copy
from datetime import datetime
from typing import Dict, Any, Optional, List

import ray
from ray import tune, train
from ray.train import Checkpoint, get_context
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.logger import TBXLoggerCallback
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search import ConcurrencyLimiter
from ray.tune.stopper import Stopper
from ray.tune import Tuner, TuneConfig, PlacementGroupFactory, RunConfig, CLIReporter
from ray.air import CheckpointConfig, FailureConfig
from tensorboardX import SummaryWriter

from optuna.samplers import TPESampler

from paper_icps.tuning import search_spaces as hyperparam_search_spaces
from paper_icps.core import training, common, config

import numpy as np

# TODO: Move to Config
# Tuning epoch budgets
MAX_EPOCHS_COARSE = 30   # cheap, fast exploration
MAX_EPOCHS_FINE   = 150   # more training for fine stage

def setup_logging():
    """Setup comprehensive logging for debugging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def get_scheduler(
        scheduler_type: str = "asha",
        metric: str = "val_loss",
        mode: str = "min",
        max_epochs: int = 30
    ) -> Any:
    """Get improved scheduler configurations"""
    schedulers = {
        "asha": ASHAScheduler(
            metric=metric,
            mode=mode,
            max_t=max_epochs,
            grace_period=3,
            reduction_factor=3,
            brackets=1,
        ),
        "pbt": PopulationBasedTraining(
            time_attr="training_iteration",
            metric=metric,
            mode=mode,  
            perturbation_interval=4,
            hyperparam_mutations={
                "lr": lambda: tune.loguniform(1e-5, 1e-2).sample(),
                "dropout": lambda: tune.uniform(0.0001, 0.2).sample(),
                "batch_size": lambda: tune.choice([32, 64, 128]).sample(),
            },
            resample_probability=0.2,
            custom_explore_fn=None,  # Use default exploration
        )
    }
    return schedulers.get(scheduler_type, schedulers["asha"])


def _extract_points_from_previous_results(
    previous_results: List[Dict], metric: str
) -> List[Dict]:
    """
    Convert previous Ray results (flattened dicts) into a list of plain
    hparam dicts for points_to_evaluate.
    Supports both:
    - records with key 'config'
    - records with keys 'config/hparams.xxx'
    """
    if not previous_results:
        return []

    # sort by metric if present
    def _metric_val(r):
        v = r.get(metric)
        if v is None:
            return float("inf")
        return v

    sorted_results = sorted(previous_results, key=_metric_val)

    points_to_evaluate: List[Dict] = []
    for r in sorted_results[:10]:
        if "config" in r and isinstance(r["config"], dict):
            # older / manual format
            points_to_evaluate.append(r["config"])
            continue

        # Ray DataFrame style: flatten keys like "config/hparams.d_model"
        cfg: Dict[str, Any] = {}
        for k, v in r.items():
            if isinstance(k, str) and k.startswith("config/hparams."):
                name = k.split("config/hparams.", 1)[1]
                cfg[name] = v
        if cfg:
            points_to_evaluate.append(cfg)

    return points_to_evaluate

def short_trial_name_creator(trial) -> str:
    """
    Create a compact, human-readable trial name for TensorBoard / Ray logs.

    Only includes the most important hparams, not huge stuff like data_path.
    """
    cfg = trial.config or {}
    hparams = cfg.get("hparams", {}) or {}

    interesting_keys = ["d_model", "e_layers", "batch_size", "lr", "dropout"]

    parts: list[str] = []
    for k in interesting_keys:
        v = hparams.get(k)
        if v is None:
            continue

        # Pretty-print floats a bit
        if isinstance(v, float):
            if k == "lr":
                v = f"{v:.1e}"      # scientific notation for lr
            else:
                v = f"{v:.3f}".rstrip("0").rstrip(".")

        parts.append(f"{k}={v}")

    # Fallback if somehow no interesting keys are present
    if not parts:
        return f"trial_{trial.trial_id}"

    return f"trial_{trial.trial_id}_" + "__".join(parts)

def create_progress_reporter() -> CLIReporter:
    """
    Compact console table:
    - only a handful of important hparams
    - a couple of metrics
    """
    parameter_columns = {
        "hparams/batch_size": "bs",
        "hparams/d_model": "d_model",
        "hparams/d_ff": "d_ff",
        "hparams/e_layers": "layers",
    }

    metric_columns = {
        "training_iteration": "iter",
        "val_loss": "val",
    }

    return CLIReporter(
        parameter_columns=parameter_columns,
        metric_columns=metric_columns,
        max_progress_rows=20,       # how many rows to show
        sort_by_metric=True,
        metric="val_loss",
        mode="min",
    )

def build_refined_search_space(
    base_space: Dict[str, Any],
    previous_results: List[Dict],
    metric: str = "val_loss",
    top_k: int = 10,
) -> Dict[str, Any]:
    """
    Build a 'fine stage' search space by restricting each hyperparam
    to the values observed in the top_k previous configs.

    - For each hparam in base_space:
      - collect its values in the top_k configs (from previous_results)
      - if some values exist: use tune.choice over unique values
      - else: fall back to the original base_space entry
    """
    points = _extract_points_from_previous_results(previous_results, metric)
    if not points:
        return base_space

    # ensure we only look at top_k points
    points = points[:top_k]

    refined: Dict[str, Any] = {}
    for name, spec in base_space.items():
        # skip static values (strings, ints, etc.) – just keep them
        if not isinstance(spec, dict) and not hasattr(spec, "get"):
            # simple heuristic: if it's not a Tune sampling object, keep as is
            refined[name] = spec
            continue

        vals = [p[name] for p in points if name in p]
        if not vals:
            # no info from previous runs, keep original spec
            refined[name] = spec
            continue

        unique_vals = sorted(set(vals))
        # Even if there's only 1 value, using tune.choice is fine
        refined[name] = tune.choice(unique_vals)

    return refined

def get_search_algorithm(
    search_algo: str = "optuna",
    previous_results: Optional[List[Dict]] = None,
    metric: str = "val_loss",
    mode: str = "min",
    max_concurrent: int = 16,
    study_name: str | None = None,
    storage_url: str | None = None
) -> Any:
    """Get advanced search algorithms with warm-starting and Optuna dashboard support"""
    
    # Extract points to evaluate from previous results
    points_to_evaluate = _extract_points_from_previous_results(previous_results or [], metric)
    
    # Create Optuna storage object if storage_url is provided
    optuna_storage = None
    if storage_url:
        import optuna
        
        # Handle path expansion for SQLite URLs
        if storage_url.startswith("sqlite:///~/"):
            # Expand ~ in SQLite path
            db_path = storage_url.replace("sqlite:///~/", "")
            expanded_path = os.path.expanduser(f"~/{db_path}")
            # Ensure directory exists
            os.makedirs(os.path.dirname(expanded_path), exist_ok=True)
            storage_url = f"sqlite:///{expanded_path}"
        
        try:
            optuna_storage = optuna.storages.RDBStorage(storage_url)
        except Exception as e:
            print(f"Warning: Failed to create Optuna storage: {e}")
            print("Continuing without dashboard support")
            optuna_storage = None
    
    algorithms = {
        "optuna": OptunaSearch(
            metric=metric,
            mode=mode,
            points_to_evaluate=points_to_evaluate,
            study_name=study_name,
            storage=optuna_storage,
            sampler=TPESampler(multivariate=True, group=True, n_startup_trials=20)
        ),
        "bayesopt": BayesOptSearch(
            metric=metric,
            mode=mode,
            points_to_evaluate=points_to_evaluate,
            random_search_steps=10,  # Initial random exploration
        ),
        "hyperopt": HyperOptSearch(
            metric=metric,
            mode=mode,
            points_to_evaluate=points_to_evaluate,
            n_initial_points=10,
        )
    }
    
    # Wrap with concurrency limiter to prevent resource exhaustion
    base_algo = algorithms.get(search_algo, algorithms["optuna"])
    return ConcurrencyLimiter(base_algo, max_concurrent=max_concurrent)


def callback(val_loss, val_loss_min, model, epoch, metrics_history, extras: dict | None = None):
    """Enhanced callback with additional metrics and early stopping logic"""
    logger = logging.getLogger(__name__)
    
    # Log comprehensive metrics
    logger.info(f"Epoch {epoch}: val_loss={val_loss:.6f}, best_so_far={val_loss_min:.6f}")
    
    # Track metrics history for plateau detection
    metrics_history.append(val_loss)
    
    # Enhanced checkpointing
    if val_loss < val_loss_min:
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            model_file = os.path.join(temp_checkpoint_dir, "checkpoint.pt")
            common.save_model(model, model_file)
            
            checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
            payload = {
                "val_loss": val_loss,
                "epoch": epoch,
                "improvement": val_loss_min - val_loss,
                "training_efficiency": epoch / val_loss if val_loss > 0 else 0,
            }
            if extras:
                payload.update(extras)
            train.report(payload, checkpoint=checkpoint)
            logger.info(f"New best model saved: {val_loss:.6f}")
    else:
        payload = {
            "val_loss": val_loss,
            "epoch": epoch,
            "improvement": 0,
            "training_efficiency": epoch / val_loss if val_loss > 0 else 0,
        }
        if extras:
            payload.update(extras)
        train.report(payload)

class PlateauStopper(Stopper):
    """
    Stops individual trials early if validation loss plateaus.
    """
    def __init__(self, patience=15, min_delta=1e-4, metric="val_loss"):
        self.metric = metric
        self.patience = patience
        self.min_delta = min_delta
        self.history = {}

    def __call__(self, trial_id, result):
        val = result.get(self.metric)
        if val is None:
            return False

        hist = self.history.setdefault(trial_id, [])
        hist.append(val)

        if len(hist) > self.patience:
            recent = np.array(hist[-self.patience:])
            improvement = np.abs(recent[0] - recent[-1])
            if improvement < self.min_delta:
                print(f"PlateauStopper: {trial_id} stopped (Δ={improvement:.6f})")
                return True
        return False

    def stop_all(self):
        return False


def create_experiment_stopper(stage: str = "coarse"):
    """Stage-aware Plateau stopper."""
    if stage == "coarse":
        # be aggressive – if nothing happens within 5 evals, kill it
        return PlateauStopper(patience=5, min_delta=1e-3)
    else:
        # a bit more patience in the fine stage
        return PlateauStopper(patience=10, min_delta=1e-4)

def run_hyperparameter_optimization(
    model_setup: str,
    num_samples: int = 300,
    scheduler_type: str = "asha",
    search_algo: str = "optuna",
    data_path: str = "../dataset/pick_n_place_procedure_dataset.csv",
    data_path_overrides: str = "../dataset/pick_n_place_procedure_w_overrides.csv",
    max_concurrent: int = 3,
    experiment_name: str | None = None,
    resume: bool = False,
    warm_start_path: str | None = None,
    enable_dashboard: bool = False,
    cpu_per_trial: float = 8/3,
    gpu_per_trial: float = 1,
    storage_url: str | None = None,
    enable_tensorboard: bool = True,
    stage: str = "coarse",
    sliding_stride: int = 1
):
    """Run advanced hyperparameter optimization with improvements"""
    
    logger = setup_logging()
    logger.info(
        f"Starting advanced hyperparameter optimization for {model_setup} "
        f"(stage={stage}, num_samples={num_samples})"
    )
    
    # ---------------------------
    # Experiment directory layout
    # ---------------------------
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_dir = os.path.expanduser(
        os.path.join("~", "ray_icps", model_setup, timestamp)
    )
    os.makedirs(base_dir, exist_ok=True)
    ray_dir = os.path.join(base_dir, "ray")
    os.makedirs(ray_dir, exist_ok=True)

    logger.info(f"Experiment directory: {base_dir}")


    # ---------------------------
    # Model & search space setup
    # ---------------------------
    eval_config, model_config, search_space = hyperparam_search_spaces.assemble_setup(
        model_setup
    )
    # Ensure strategy_args exists
    eval_config.setdefault("strategy_args", {})
    eval_config["strategy_args"]["sliding_stride"] = int(sliding_stride)
    
    model_config["models"][0]["input_sampling"] = 1

     # Stage-specific epoch budget
    if stage == "coarse":
        max_epochs = min(MAX_EPOCHS_COARSE, config.max_epochs)
    else:
        max_epochs = min(MAX_EPOCHS_FINE, config.max_epochs)

    # If search_space contains num_epochs, override with stage-specific budget
    if "num_epochs" in search_space:
        search_space["num_epochs"] = max_epochs

    # Load previous results for warm starting if available
    previous_results = None
    if warm_start_path and os.path.exists(warm_start_path):
        try:
            import pickle
            with open(warm_start_path, 'rb') as f:
                previous_results = pickle.load(f)
            logger.info(f"Loaded {len(previous_results)} previous results for warm starting")
        except Exception as e:
            logger.warning(f"Could not load warm start data: {e}")

    # Stage-specific: refine search space in fine stage
    if stage == "fine" and previous_results:
        logger.info(
            f"Building refined search space from {len(previous_results)} previous results..."
        )
        search_space = build_refined_search_space(
            base_space=search_space,
            previous_results=previous_results,
            metric="val_loss",
            top_k=10,  # you can tune this
        )
    elif stage == "fine" and not previous_results:
        logger.warning(
            "Stage 'fine' selected but no previous results found. "
            "Falling back to the original search space."
        )
    
    # Setup Optuna dashboard storage if enabled
    study_name = experiment_name or f"{model_setup}_optimization"
    if enable_dashboard and storage_url is None:
        # Create default SQLite storage with absolute path
        db_dir = os.path.expanduser("~/ray_results")
        os.makedirs(db_dir, exist_ok=True)
        db_path = os.path.join(db_dir, f"{study_name}.db")
        storage_url = f"sqlite:///{db_path}"
        logger.info(f"Optuna dashboard enabled. Storage: {storage_url}")
        logger.info(f"To view dashboard, run: optuna-dashboard {storage_url}")
    
    # Setup advanced components
    scheduler = get_scheduler(
        scheduler_type,
        metric="val_loss",
        mode="min",
        max_epochs=max_epochs
    )
    search_algorithm = get_search_algorithm(
        search_algo, 
        previous_results, 
        max_concurrent=max_concurrent,
        study_name=study_name,
        storage_url=storage_url if enable_dashboard else None
    )
    stopper = create_experiment_stopper(stage=stage)
    
    # Enhanced training function with metrics tracking
    def training_wrapper(config: dict):
        """
        config = {
        "model_setup": "timexer",
        "eval_config": ...,
        "model_config": ...,
        "data_path": ...,
        "data_path_overrides": ...,
        "hparams": { actual search space params... }
        }
        """

        selected_hyperparams = config["hparams"]
        local_model_config = copy.deepcopy(config["model_config"])
        eval_cfg = copy.deepcopy(config["eval_config"])
        local_data_path = config["data_path"]
        local_data_path_overrides = config["data_path_overrides"]

        metrics_history: List[float] = []

        # TB writer in trial logdir
        ctx = get_context()
        trial_dir = ctx.get_trial_dir() if ctx else os.getcwd()
        writer = SummaryWriter(log_dir=trial_dir)

        # --- HParams einmalig loggen (erscheint im HParams-Plugin):
        # Achtung: metric_dict braucht mind. einen Eintrag – initial 0.
        hparam_dict = {
            "d_model": selected_hyperparams.get("d_model"),
            "d_ff": selected_hyperparams.get("d_ff"),
            "e_layers": selected_hyperparams.get("e_layers"),
            "batch_size": selected_hyperparams.get("batch_size"),
            "lr": selected_hyperparams.get("lr"),
            "dropout": selected_hyperparams.get("dropout"),
        }
        writer.add_hparams(hparam_dict, {"val_loss/last": 0.0})

        # Also log each numeric hparam as scalar for convenience
        for k, v in selected_hyperparams.items():
            if isinstance(v, (int, float)):
                writer.add_scalar(f"hparams/{k}", float(v), 0)

        # Status scalar
        writer.add_scalar("status/running", 1, 0)
        writer.flush()

        # write a small meta.json into the trial dir
        meta = {
            "model_setup": config["model_setup"],
            "hparams": selected_hyperparams,
        }
        try:
            with open(os.path.join(trial_dir, "meta.json"), "w") as f:
                json.dump(meta, f, indent=2)
        except Exception:
            pass
            
        def callback_wrapper(
            val_loss, val_loss_min, model, epoch=0, extras: dict | None = None
        ):
            result = callback(
                val_loss, val_loss_min, model, epoch, metrics_history, extras
            )

            writer.add_scalar("loss/val", float(val_loss), epoch)
            writer.add_scalar("loss/best_so_far", float(val_loss_min), epoch)
            if extras:
                if "lr" in extras:
                    writer.add_scalar(
                        "optim/lr", float(extras["lr"]), epoch
                    )
                if "throughput" in extras:
                    writer.add_scalar(
                        "perf/throughput",
                        float(extras["throughput"]),
                        epoch,
                    )
            writer.flush()
            return result
        
        try:
            # inject hparams
            local_model_config["models"][0]["model_hyper_params"] = selected_hyperparams

            result = training.train_model(
                local_model_config,
                eval_cfg,
                training_progress_callback=callback_wrapper,
                data_path=local_data_path,
                data_path_overrides=local_data_path_overrides,
            )
            if isinstance(result, dict):
                train.report(
                    {
                        "val_loss": result.get("val_loss"),
                        "training_time": result.get("training_time"),
                        "model_params_millions": result.get(
                            "model_params_millions"
                        ),
                    }
                )
            return result
        finally:
            writer.add_scalar("status/running", 0, 999999)
            writer.flush()
            writer.close()


    # ---------------------------
    # Ray init
    # ---------------------------
    if not ray.is_initialized():
        ray.init(runtime_env={"working_dir": os.getcwd()})

    # ---------------------------
    # Param space: static + hparams
    # ---------------------------
    base_tune_config = {
        "model_setup": model_setup,
        "eval_config": eval_config,
        "model_config": model_config,
        "data_path": data_path,
        "data_path_overrides": data_path_overrides,
    }

    param_space = {"hparams": search_space}
    param_space.update(base_tune_config)

    placement = PlacementGroupFactory([{"CPU": cpu_per_trial, "GPU": gpu_per_trial}])

    trainable_with_resources = tune.with_resources(
        training_wrapper, resources=placement
    )

    # If PBT is used, search algorithm must be None
    if isinstance(scheduler, PopulationBasedTraining):
        logger.info("Using PBT — disabling external search algorithm.")
        search_algorithm = None

    
    # Register trainable
    trainable_name = experiment_name or f"{model_setup}_advanced_tuning"
    tune.register_trainable(trainable_name, training_wrapper)

    progress_reporter = create_progress_reporter()

    tuner = Tuner(
        trainable_with_resources,
        param_space=param_space,
        tune_config=TuneConfig(
            num_samples=num_samples,
            scheduler=scheduler,
            search_alg=search_algorithm,
            max_concurrent_trials=max_concurrent,
            trial_name_creator=short_trial_name_creator
        ),
        run_config=RunConfig(
            name=trainable_name,
            storage_path=ray_dir,
            stop=stopper,
            verbose=2,
            checkpoint_config=CheckpointConfig(
                num_to_keep=3,
                checkpoint_score_attribute="val_loss",
                checkpoint_score_order="min",
            ),
            failure_config=FailureConfig(max_failures=5),
            callbacks=[TBXLoggerCallback()] if enable_tensorboard else [],
            progress_reporter=progress_reporter,
        ),
    )

    
    # Run optimization
    logger.info("Starting Ray Tune optimization...")
    analysis = tuner.fit()

    # ---------------------------
    # Save results nicely
    # ---------------------------
    results_pkl = os.path.join(base_dir, "results.pkl")
    results_csv = os.path.join(base_dir, "results.csv")
    summary_md = os.path.join(base_dir, "summary.md")
    best_cfg_path_yaml = os.path.join(base_dir, "best_config.yaml")
    best_cfg_path_json = os.path.join(base_dir, "best_config.json")

    try:
        results_df = analysis.get_dataframe()
        results_data = results_df.to_dict("records")

        # pkl for warm-start
        import pickle

        with open(results_pkl, "wb") as f:
            pickle.dump(results_data, f)

        results_df.to_csv(results_csv, index=False)
        logger.info(f"Results saved to {results_pkl} and {results_csv}")
    except Exception as e:
        logger.warning(f"Could not save results: {e}")
        results_df = None

    # Best result (new API)
    try:
        best_result = analysis.get_best_result(
            metric="val_loss", mode="min", scope="last"
        )
    except Exception as e:
        logger.warning(f"Could not obtain best result: {e}")
        best_result = None

    best_val_loss: float | None = None

    if best_result is not None:
        # metrics itself can be None according to type hints
        metrics = getattr(best_result, "metrics", None)
        if isinstance(metrics, dict):
            val = metrics.get("val_loss")
            if isinstance(val, (int, float)):
                best_val_loss = float(val)

        logger.info(f"Best config: {best_result.config}")
        if best_val_loss is not None:
            logger.info(
                f"Best trial final validation loss: {best_val_loss}"
            )
        else:
            logger.info("Best result has no numeric 'val_loss' metric.")

        # Save best config as YAML/JSON
        best_config = best_result.config
        try:
            import yaml

            with open(best_cfg_path_yaml, "w") as f:
                yaml.dump(best_config, f, sort_keys=False)
            logger.info(f"Best config saved to {best_cfg_path_yaml}")
        except Exception as e:
            logger.warning(
                f"Could not save YAML best config ({e}), falling back to JSON."
            )
            with open(best_cfg_path_json, "w") as f:
                json.dump(best_config, f, indent=2)
            logger.info(f"Best config saved to {best_cfg_path_json}")

        # Write human-readable summary.md
        try:
            with open(summary_md, "w") as f:
                f.write("# Hyperparameter Tuning Summary\n\n")
                f.write(f"- Model setup: `{model_setup}`\n")
                f.write(f"- Samples: `{num_samples}`\n")
                f.write(f"- Scheduler: `{scheduler_type}`\n")
                f.write(f"- Search algo: `{search_algo}`\n")
                if results_df is not None:
                    f.write(f"- Total trials: `{len(results_df)}`\n")
                if best_val_loss is not None:
                    f.write(f"- Best val_loss: `{best_val_loss:.6f}`\n\n")
                else:
                    f.write("- Best val_loss: `N/A`\n\n")

                if results_df is not None:
                    topk_df = results_df.sort_values("val_loss").head(10)
                    cols = [
                        c for c in results_df.columns
                        if c.startswith("config/hparams.")
                    ]
                    cols += [
                        "val_loss",
                        "training_time"
                        if "training_time" in results_df.columns
                        else "training_iteration",
                    ]
                    f.write("## Top 10 Trials\n\n")
                    f.write(topk_df[cols].to_markdown(index=False))
                    f.write("\n")
            logger.info(f"Summary saved to {summary_md}")
        except Exception as e:
            logger.warning(f"Could not write summary.md: {e}")
    else:
        logger.warning("No best trial found.")

    return analysis

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Advanced Hyperparameter Optimization"
    )
    parser.add_argument("--model-setup", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=80)
    parser.add_argument(
        "--sliding-stride",
        type=int,
        default=1,
        help="Sliding window stride in samples for dataset windows."
                        )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="asha",
        choices=["asha", "pbt"],
    )
    parser.add_argument(
        "--search-algo",
        type=str,
        default="optuna",
        choices=["optuna", "bayesopt", "hyperopt"],
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="coarse",
        choices=["coarse", "fine"],
        help="Two-stage tuning: 'coarse' for wide search, 'fine' for focused search.",
    )
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--warm-start-path", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--data-path",
        type=str,
        default="../dataset/pick_n_place_procedure_dataset.csv",
    )
    parser.add_argument(
        "--data-path-overrides",
        type=str,
        default="../dataset/pick_n_place_procedure_w_overrides.csv",
    )
    parser.add_argument(
        "--cpu-per-trial",
        type=float,
        default=3,
        help="CPU resources per trial (lower = more parallel trials)",
    )
    parser.add_argument(
        "--gpu-per-trial",
        type=float,
        default=1 / 6,
        help="GPU resources per trial (lower = more parallel trials)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=16,
        help="Maximum concurrent trials",
    )
    parser.add_argument(
        "--enable-dashboard",
        action="store_true",
        help="Enable Optuna dashboard with persistent storage",
    )
    parser.add_argument(
        "--storage-url",
        type=str,
        default=None,
        help="Custom storage URL for Optuna (e.g., sqlite:///path/to/db.sqlite3)",
    )
    parser.add_argument(
        "--enable-tensorboard",
        action="store_true",
        help="Enable TensorBoard logging for Ray Tune trials",
    )

    args = parser.parse_args()
    
    run_hyperparameter_optimization(
        model_setup=args.model_setup,
        num_samples=args.num_samples,
        scheduler_type=args.scheduler,
        search_algo=args.search_algo,
        experiment_name=args.experiment_name,
        warm_start_path=args.warm_start_path,
        resume=args.resume,
        data_path=args.data_path,
        data_path_overrides=args.data_path_overrides,
        cpu_per_trial=args.cpu_per_trial,
        max_concurrent=args.max_concurrent,
        enable_dashboard=args.enable_dashboard,
        storage_url=args.storage_url,
        enable_tensorboard=args.enable_tensorboard,
        stage=args.stage,
        sliding_stride=args.sliding_stride,
    )