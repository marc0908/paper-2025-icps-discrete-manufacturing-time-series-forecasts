#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Improved Hyperparameter Optimization with Advanced Techniques
"""
import argparse
import os
import tempfile
import logging
from typing import Dict, Any, Optional, List

import ray
from ray import tune, train
from ray.train import Checkpoint, CheckpointConfig
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.logger import TBXLoggerCallback
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search import ConcurrencyLimiter
from ray.tune.stopper import Stopper
from ray.tune import Tuner, TuneConfig, PlacementGroupFactory
from ray.air import RunConfig, CheckpointConfig, FailureConfig

from optuna.samplers import TPESampler

from paper_icps.tuning import improved_search_spaces as hyperparam_search_spaces
from paper_icps.core import training, common, config

import numpy as np

def setup_logging():
    """Setup comprehensive logging for debugging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def get_improved_scheduler(scheduler_type: str = "asha_improved") -> Any:
    """Get improved scheduler configurations"""
    schedulers = {
        "asha_improved": ASHAScheduler(
            max_t=config.max_epochs,
            grace_period=10,
            reduction_factor=2,
            brackets=1,
        ),
        "pbt": PopulationBasedTraining(
            time_attr="training_iteration",
            metric="val_loss",
            mode="min",  
            perturbation_interval=4,
            hyperparam_mutations={
                "lr": lambda: tune.loguniform(1e-5, 1e-2).sample(),
                "dropout": lambda: tune.uniform(0.0001, 0.2).sample(),
                "batch_size": lambda: tune.choice([32, 64, 128]).sample(),
            },
            resample_probability=0.2,
            custom_explore_fn=None,  # Use default exploration
        ),
        "asha_conservative": ASHAScheduler(
            max_t=config.max_epochs,
            grace_period=30,  # Very conservative
            reduction_factor=4,  # Even more conservative  
            brackets=2,
        )
    }
    return schedulers.get(scheduler_type, schedulers["asha_improved"])


def get_advanced_search_algorithm(
    search_algo: str = "optuna",
    previous_results: Optional[List[Dict]] = None,
    metric: str = "val_loss",
    mode: str = "min",
    max_concurrent: int = 16,
    study_name: str = None,
    storage_url: str = None
) -> Any:
    """Get advanced search algorithms with warm-starting and Optuna dashboard support"""
    
    # Extract points to evaluate from previous results
    points_to_evaluate = []
    if previous_results:
        # Take top 10 configurations from previous runs
        sorted_results = sorted(previous_results, key=lambda x: x.get(metric, float('inf')))
        points_to_evaluate = [r['config'] for r in sorted_results[:10]]
    
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
        "hyperopt_improved": HyperOptSearch(
            metric=metric,
            mode=mode,
            points_to_evaluate=points_to_evaluate,
            n_initial_points=20,  # Increased initial random points
        )
    }
    
    # Wrap with concurrency limiter to prevent resource exhaustion
    base_algo = algorithms.get(search_algo, algorithms["optuna"])
    return ConcurrencyLimiter(base_algo, max_concurrent=max_concurrent)


def enhanced_callback(val_loss, val_loss_min, model, epoch, metrics_history, extras: dict = None):
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

def create_experiment_stopper():
    """Combine PlateauStopper with ASHA for adaptive early stopping."""
    return PlateauStopper(patience=20, min_delta=1e-4)

def run_advanced_hyperparameter_optimization(
    model_setup: str,
    num_samples: int = 300,
    scheduler_type: str = "asha_improved",
    search_algo: str = "optuna",
    data_path: str = "../dataset/pick_n_place_procedure_dataset.csv",
    data_path_overrides: str = "../dataset/pick_n_place_procedure_w_overrides.csv",
    max_concurrent: int = 3,
    experiment_name: str = None,
    resume: bool = False,
    warm_start_path: str = None,
    enable_dashboard: bool = False,
    cpu_per_trial: float = 8/3,
    gpu_per_trial: float = 1,
    storage_url: str = None,
    enable_tensorboard: bool = True,
):
    """Run advanced hyperparameter optimization with improvements"""
    
    logger = setup_logging()
    logger.info(f"Starting advanced hyperparameter optimization for {model_setup}")
    
    # Get model configuration
    eval_config, model_config, search_space = hyperparam_search_spaces.improved_assemble_setup(model_setup)
    model_config["models"][0]["input_sampling"] = 1
    
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
    scheduler = get_improved_scheduler(scheduler_type)
    search_algorithm = get_advanced_search_algorithm(
        search_algo, 
        previous_results, 
        max_concurrent=max_concurrent,
        study_name=study_name,
        storage_url=storage_url if enable_dashboard else None
    )
    stopper = create_experiment_stopper()
    
    # Enhanced training function with metrics tracking
    def enhanced_training_wrapper(selected_hyperparams):
        metrics_history = []
        
        def enhanced_callback_wrapper(val_loss, val_loss_min, model, epoch=0, extras: dict = None):
            return enhanced_callback(val_loss, val_loss_min, model, epoch, metrics_history, extras)
        
        # Update model config
        model_config["models"][0]["model_hyper_params"] = selected_hyperparams
        
        tune.utils.wait_for_gpu() # Wait for GPU Memory to be cleared

        # Train with enhanced callback
        result = training.train_model(
            model_config,
            eval_config,
            training_progress_callback=enhanced_callback_wrapper,
            data_path=data_path,
            data_path_overrides=data_path_overrides,
        )
        # Report final metrics for analysis
        if isinstance(result, dict):
            train.report({
                "val_loss": result.get("val_loss"),
                "training_time": result.get("training_time"),
                "model_params_millions": result.get("model_params_millions"),
            })
        return result
    
    # Initialize Ray if not already done
    if not ray.is_initialized():
            ray.init(runtime_env={"working_dir": os.getcwd()})
    
    # Register trainable
    trainable_name = experiment_name or f"{model_setup}_advanced_tuning"
    tune.register_trainable(trainable_name, enhanced_training_wrapper)
    
    # Run optimization
    logger.info("Starting Ray Tune optimization...")
    
    # Prepare tune.run arguments
    tune_kwargs = {
        "config": search_space,
        "scheduler": scheduler,
        "search_alg": search_algorithm,
        "num_samples": num_samples,
        #"resources_per_trial": {"cpu": cpu_per_trial, "gpu": gpu_per_trial},  # Configurable CPU per trial
        "metric": "val_loss",
        "mode": "min",
        "checkpoint_config": CheckpointConfig(
            num_to_keep=3,  # Keep top 3 checkpoints
            checkpoint_score_attribute="val_loss",
            checkpoint_score_order="min",
        ),
        "resume": resume,
        "name": trainable_name,
        "storage_path": os.path.expanduser("~/ray_results"),
        "max_failures": 10, 
        "log_to_file": True,
        "verbose": 1,
    }
    
    # Add stopper if it's not None
    if stopper is not None:
        tune_kwargs["stop"] = stopper
    
    # Add TensorBoard logging if enabled
    if enable_tensorboard:
        tbx_cb = TBXLoggerCallback()
        existing_cbs = tune_kwargs.get("callbacks") or []
        tune_kwargs["callbacks"] = [*existing_cbs, tbx_cb]

    placement = PlacementGroupFactory(
        [{"CPU": cpu_per_trial, "GPU": gpu_per_trial}]
    )

    # Wrap your trainable to include these resources
    trainable_with_resources = tune.with_resources(
        enhanced_training_wrapper,
        resources=placement
    )

    # Dont use search algorithm if PBT is used
    if isinstance(scheduler, PopulationBasedTraining):
        search_algorithm = None
        logger.info("⚠️ Using PBT — disabling external search algorithm.")

    tuner = Tuner(
        trainable_with_resources,
        param_space=search_space,
        tune_config=TuneConfig(
            num_samples=num_samples,
            scheduler=scheduler,
            search_alg=search_algorithm,
            max_concurrent_trials=max_concurrent,
        ),
        run_config=RunConfig(
            name=trainable_name,
            storage_path=os.path.expanduser("~/ray_results"),
            checkpoint_config=CheckpointConfig(
                num_to_keep=3,
                checkpoint_score_attribute="val_loss",
                checkpoint_score_order="min",
            ),
            failure_config=FailureConfig(max_failures=5),
            callbacks=[TBXLoggerCallback()] if enable_tensorboard else [],
        ),
    )

    analysis = tuner.fit()
    
    # Save results for future warm starting
    results_path = os.path.expanduser(f"~/ray_results/{trainable_name}_results.pkl")
    try:
        results_df = analysis.results_df
        results_data = results_df.to_dict('records')
        with open(results_path, 'wb') as f:
            import pickle
            pickle.dump(results_data, f)
        # Also save CSV for quick inspection
        csv_path = os.path.expanduser(f"~/ray_results/{trainable_name}_results.csv")
        results_df.to_csv(csv_path, index=False)
        logger.info(f"Results saved to {results_path} and {csv_path}")
    except Exception as e:
        logger.warning(f"Could not save results: {e}")
    
    # Print best results
    best_trial = analysis.get_best_trial("val_loss", "min", "last")
    logger.info(f"Best trial config: {best_trial.config}")
    logger.info(f"Best trial final validation loss: {best_trial.last_result['val_loss']}")
    
    return analysis

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Advanced Hyperparameter Optimization")
    parser.add_argument("--model-setup", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=300)
    parser.add_argument("--scheduler", type=str, default="asha_improved", 
                       choices=["asha_improved", "pbt", "asha_conservative"])
    parser.add_argument("--search-algo", type=str, default="optuna",
                       choices=["optuna", "bayesopt", "hyperopt_improved"])
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--warm-start-path", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--data-path", type=str, 
                       default="../dataset/pick_n_place_procedure_dataset.csv")
    parser.add_argument("--data-path-overrides", type=str,
                       default="../dataset/pick_n_place_procedure_w_overrides.csv")
    parser.add_argument("--cpu-per-trial", type=float, default=3,
                       help="CPU resources per trial (lower = more parallel trials)")
    parser.add_argument("--gpu-per-trial", type=float, default=1/6,
                       help="GPU resources per trial (lower = more parallel trials)")
    parser.add_argument("--max-concurrent", type=int, default=16,
                       help="Maximum concurrent trials")
    parser.add_argument("--enable-dashboard", action="store_true",
                       help="Enable Optuna dashboard with persistent storage")
    parser.add_argument("--storage-url", type=str, default=None,
                       help="Custom storage URL for Optuna (e.g., sqlite:///path/to/db.sqlite3)")
    parser.add_argument("--enable-tensorboard", action="store_true",
                       help="Enable TensorBoard logging for Ray Tune trials")
    
    args = parser.parse_args()
    
    run_advanced_hyperparameter_optimization(
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
    )