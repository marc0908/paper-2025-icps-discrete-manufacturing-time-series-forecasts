"""
Multi-Objective Hyperparameter Optimization for Time Series Models
Optimizes for multiple metrics simultaneously: validation loss, training time, model size, etc.
"""
import os
import time
from typing import Dict, List, Tuple, Any
import logging

import torch
import numpy as np
from ray import tune, train
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler

import config
import training
import common


class MultiObjectiveMetrics:
    """Track multiple objectives during training"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.start_time = time.time()
        self.val_losses = []
        self.train_times = []
        self.memory_usage = []
        self.convergence_epochs = None
        self.model_params = 0
        
    def update(self, val_loss: float, epoch: int, model=None):
        """Update metrics"""
        self.val_losses.append(val_loss)
        self.train_times.append(time.time() - self.start_time)
        
        # Track memory if available
        if torch.cuda.is_available():
            self.memory_usage.append(torch.cuda.max_memory_allocated() / 1024**3)  # GB
        
        # Track convergence (when improvement becomes minimal)
        if len(self.val_losses) > 10 and self.convergence_epochs is None:
            recent_losses = self.val_losses[-10:]
            if max(recent_losses) - min(recent_losses) < 0.001:  # Converged
                self.convergence_epochs = epoch
        
        # Count model parameters once
        if model is not None and self.model_params == 0:
            self.model_params = sum(p.numel() for p in model.parameters())
    
    def get_final_metrics(self) -> Dict[str, float]:
        """Get final multi-objective metrics"""
        if not self.val_losses:
            return {"val_loss": float('inf'), "training_time": float('inf')}
        
        final_val_loss = min(self.val_losses)
        total_training_time = self.train_times[-1] if self.train_times else float('inf')
        avg_memory = np.mean(self.memory_usage) if self.memory_usage else 0
        convergence_speed = self.convergence_epochs or len(self.val_losses)
        
        # Efficiency metrics
        params_millions = self.model_params / 1e6
        time_per_epoch = total_training_time / len(self.val_losses) if self.val_losses else float('inf')
        
        return {
            "val_loss": final_val_loss,
            "training_time": total_training_time,
            "memory_usage_gb": avg_memory,
            "convergence_epochs": convergence_speed,
            "model_params_millions": params_millions,
            "time_per_epoch": time_per_epoch,
            # Composite scores
            "efficiency_score": final_val_loss * (total_training_time / 3600),  # Loss * hours
            "pareto_score": self._calculate_pareto_score(final_val_loss, total_training_time, params_millions),
        }
    
    def _calculate_pareto_score(self, loss: float, time: float, params: float) -> float:
        """Calculate Pareto efficiency score (lower is better)"""
        # Normalize and combine objectives (weights can be tuned)
        normalized_loss = loss  # Already in reasonable range
        normalized_time = min(time / 3600, 10)  # Cap at 10 hours
        normalized_params = min(params / 100, 1)  # Cap at 100M params
        
        # Weighted combination (adjust weights based on priorities)
        return 0.7 * normalized_loss + 0.2 * normalized_time + 0.1 * normalized_params


def multi_objective_callback(val_loss, val_loss_min, model, epoch=0, metrics_tracker=None):
    """Enhanced callback for multi-objective optimization"""
    
    if metrics_tracker is None:
        # Fallback to simple callback
        return simple_callback(val_loss, val_loss_min, model)
    
    # Update multi-objective metrics
    metrics_tracker.update(val_loss, epoch, model)
    final_metrics = metrics_tracker.get_final_metrics()
    
    # Report all metrics to Ray Tune
    should_checkpoint = val_loss < val_loss_min
    
    if should_checkpoint:
        import tempfile
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            model_file = os.path.join(temp_checkpoint_dir, "checkpoint.pt")
            common.save_model(model, model_file)
            
            checkpoint = train.Checkpoint.from_directory(temp_checkpoint_dir)
            train.report(final_metrics, checkpoint=checkpoint)
    else:
        train.report(final_metrics)


def simple_callback(val_loss, val_loss_min, model):
    """Simple fallback callback"""
    import tempfile
    from ray.train import Checkpoint
    
    if val_loss < val_loss_min:
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            model_file = os.path.join(temp_checkpoint_dir, "checkpoint.pt")
            common.save_model(model, model_file)
            checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
            train.report({"val_loss": val_loss}, checkpoint=checkpoint)
    else:
        train.report({"val_loss": val_loss})


def create_multi_objective_search_algorithm(
    objectives: List[str] = ["val_loss", "training_time", "model_params_millions"],
    directions: List[str] = ["minimize", "minimize", "minimize"]
):
    """Create multi-objective search algorithm"""
    
    # Optuna supports multi-objective optimization
    return OptunaSearch(
        # Primary metric for ASHA scheduler
        metric="pareto_score",  # Our composite score
        mode="min",
        # Note: True multi-objective requires OptunaSearch with study configuration
    )


def run_multi_objective_optimization(
    model_setup: str,
    num_samples: int = 200,
    max_training_time_hours: float = 12,
    **kwargs
):
    """Run multi-objective hyperparameter optimization"""
    
    try:
        from improved_search_spaces import improved_assemble_setup
    except ImportError:
        from hyperparam_search_spaces import assemble_setup as improved_assemble_setup
    
    # Setup
    eval_config, model_config, search_space = improved_assemble_setup(model_setup)
    model_config["models"][0]["input_sampling"] = 1
    
    # Multi-objective training wrapper
    def multi_objective_training_wrapper(selected_hyperparams):
        metrics_tracker = MultiObjectiveMetrics()
        
        def enhanced_callback_wrapper(val_loss, val_loss_min, model, epoch=0):
            return multi_objective_callback(
                val_loss, val_loss_min, model, epoch, metrics_tracker
            )
        
        model_config["models"][0]["model_hyper_params"] = selected_hyperparams
        
        try:
            result = training.train_model(
                model_config,
                eval_config,
                training_progress_callback=enhanced_callback_wrapper,
                **kwargs
            )
            return result
        except Exception as e:
            # Return poor metrics for failed trials
            return {
                "val_loss": float('inf'),
                "training_time": float('inf'),
                "pareto_score": float('inf')
            }
    
    # Setup optimization
    scheduler = ASHAScheduler(
        max_t=config.max_epochs,
        grace_period=15,  # Allow some learning before pruning
        reduction_factor=3,
    )
    
    search_alg = create_multi_objective_search_algorithm()
    
    # Time-based stopping
    from ray.tune.stopper import TimeoutStopper
    stopper = TimeoutStopper(timeout=max_training_time_hours * 3600)
    
    # Run optimization
    import ray
    if not ray.is_initialized():
        ray.init()
    
    trainable_name = f"{model_setup}_multi_objective"
    tune.register_trainable(trainable_name, multi_objective_training_wrapper)
    
    analysis = tune.run(
        trainable_name,
        config=search_space,
        scheduler=scheduler,
        search_alg=search_alg,
        stop=stopper,
        num_samples=num_samples,
        resources_per_trial={"cpu": 1.0, "gpu": 0.33},
        metric="pareto_score",
        mode="min",
        name=f"{model_setup}_pareto_optimization",
        local_dir="./ray_results",
        verbose=1,
    )
    
    return analysis


def analyze_pareto_frontier(analysis, objectives=["val_loss", "training_time"]):
    """Analyze and visualize Pareto frontier from optimization results"""
    
    import matplotlib.pyplot as plt
    import pandas as pd
    
    # Get results
    df = analysis.results_df
    
    # Extract objective values
    obj_values = df[objectives].values
    
    # Find Pareto frontier
    def is_pareto_efficient(costs):
        """Find Pareto efficient solutions"""
        is_efficient = np.ones(costs.shape[0], dtype=bool)
        for i, c in enumerate(costs):
            if is_efficient[i]:
                # Remove dominated points
                is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)
                is_efficient[i] = True
        return is_efficient
    
    pareto_mask = is_pareto_efficient(obj_values)
    pareto_solutions = df[pareto_mask]
    
    # Plot Pareto frontier
    plt.figure(figsize=(10, 8))
    plt.scatter(df[objectives[0]], df[objectives[1]], alpha=0.6, label='All solutions')
    plt.scatter(pareto_solutions[objectives[0]], pareto_solutions[objectives[1]], 
                color='red', s=100, label='Pareto frontier')
    plt.xlabel(objectives[0])
    plt.ylabel(objectives[1])
    plt.legend()
    plt.title('Multi-Objective Optimization Results')
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plt.savefig(f'pareto_frontier_{objectives[0]}_{objectives[1]}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return pareto_solutions


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-Objective Hyperparameter Optimization")
    parser.add_argument("--model-setup", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--max-hours", type=float, default=12)
    parser.add_argument("--data-path", type=str, default="datasets/pick_n_place_procedure_dataset.csv")
    parser.add_argument("--data-path-overrides", type=str, default="datasets/pick_n_place_procedure_w_overrides.csv")
    
    args = parser.parse_args()
    
    analysis = run_multi_objective_optimization(
        model_setup=args.model_setup,
        num_samples=args.num_samples,
        max_training_time_hours=args.max_hours,
        data_path=args.data_path,
        data_path_overrides=args.data_path_overrides
    )
    
    # Analyze results
    pareto_solutions = analyze_pareto_frontier(analysis)
    print(f"Found {len(pareto_solutions)} Pareto-optimal solutions")
    print("\nTop 5 Pareto solutions:")
    print(pareto_solutions.head())