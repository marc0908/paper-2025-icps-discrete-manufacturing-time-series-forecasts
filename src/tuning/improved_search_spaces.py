"""
Improved Search Spaces with Better Parameter Distributions and Ranges
"""
from ray import tune
import config


def improved_timexer_searchspace():
    """Enhanced TimeXer search space with better parameter distributions"""
    search_space = {
        # Core architecture parameters
        "batch_size": tune.choice([32, 64, 128]),  # More conservative batch sizes
        "d_model": tune.choice([64, 128, 256, 512]),
        "d_ff": tune.choice([128, 256, 512, 1024, 2048]),  # Expanded range
        "e_layers": tune.choice([1, 2, 3, 4]),  # Added 4 layers
        "n_heads": tune.choice([4, 8, 12, 16]),  # Powers of 2 for efficiency
        
        # Regularization - using log-uniform for better exploration
        "dropout": tune.loguniform(1e-4, 0.3),  # Log-uniform for better exploration
        "weight_decay": tune.loguniform(1e-6, 1e-2),  # Important for generalization
        
        # Learning parameters
        "lr": tune.loguniform(1e-5, 1e-2),
        "lradj": tune.choice(["type1", "type2", "cosine"]),  # More LR schedules
        
        # Model-specific parameters
        "patch_len": tune.choice([8, 16, 32, 64, 100, 200]),  # More granular choices
        "features": "M",  # Fixed for multivariate
        "use_norm": tune.choice([True, False]),  # Let optimization decide
        "factor": tune.randint(1, 10),  # Reduced range for stability
        
        # Training parameters
        "horizon": 400,  # Fixed based on your domain
        "seq_len": 1600,  # Fixed based on your domain
        "loss": "MAE",  # Could be made tunable: tune.choice(["MAE", "MSE"])
        "norm": True,
        "num_epochs": config.max_epochs,
        "patience": tune.choice([10, 15, 20]),  # Tunable patience
        "moving_avg": tune.choice([1, 3, 5, 7]),  # Added 7
        
        # Advanced parameters
        "grad_clip": tune.uniform(0.5, 2.0),  # Gradient clipping
        "label_smoothing": tune.uniform(0.0, 0.1),  # Label smoothing for robustness
    }
    return search_space


def improved_duet_searchspace():
    """Enhanced DUET search space"""
    search_space = {
        "CI": 1,  # Channel independent - fixed
        "batch_size": tune.choice([32, 64, 128]),
        "d_ff": tune.choice([128, 256, 512, 1024]),
        "d_model": tune.choice([128, 256, 512, 768]),  # Added 768
        "e_layers": tune.choice([1, 2, 3, 4]),
        
        # Better regularization
        "dropout": tune.loguniform(1e-4, 0.25),
        "fc_dropout": tune.loguniform(1e-5, 0.2),
        "weight_decay": tune.loguniform(1e-6, 1e-2),
        
        # Learning parameters
        "lr": tune.loguniform(1e-5, 1e-2),
        "lradj": tune.choice(["type1", "type2", "cosine"]),
        
        # Architecture parameters
        "factor": tune.randint(1, 15),  # Slightly reduced
        "n_heads": tune.choice([4, 8, 12, 16, 20]),  # More structured choices
        "num_experts": tune.randint(2, 16),  # Reduced max for stability
        "k": tune.choice([1, 2, 3, 4]),  # Added 4
        
        # Domain-specific
        "horizon": 400,
        "seq_len": 1600,
        "patch_len": tune.choice([8, 16, 32, 50, 100, 200, 400]),
        "moving_avg": tune.choice([1, 3, 5, 7]),
        
        # Fixed parameters
        "loss": "MAE",
        "norm": True,
        "num_epochs": config.max_epochs,
        "patience": tune.choice([10, 15, 20]),
        
        # Additional parameters
        "grad_clip": tune.uniform(0.5, 2.0),
    }
    return search_space


def improved_crossformer_searchspace():
    """Enhanced Crossformer search space"""
    search_space = {
        "batch_size": tune.choice([32, 64, 128]),
        "d_ff": tune.choice([128, 256, 512, 1024]),
        "d_model": tune.choice([64, 128, 256, 512]),
        "e_layers": tune.choice([1, 2, 3, 4]),
        "n_heads": tune.choice([4, 8, 12, 16]),
        
        # Better regularization
        "dropout": tune.loguniform(1e-4, 0.3),
        "weight_decay": tune.loguniform(1e-6, 1e-2),
        
        # Learning
        "lr": tune.loguniform(1e-5, 1e-2),
        "lradj": tune.choice(["type1", "type2", "cosine"]),
        
        # Model-specific
        "seg_len": tune.choice([8, 16, 32, 50, 100, 200]),  # More granular
        "factor": tune.randint(1, 12),
        
        # Fixed
        "horizon": 400,
        "seq_len": 1600,
        "loss": "MAE",
        "norm": True,
        "num_epochs": config.max_epochs,
        "patience": tune.choice([10, 15, 20]),
        "moving_avg": tune.choice([1, 3, 5, 7]),
        
        # Additional
        "grad_clip": tune.uniform(0.5, 2.0),
    }
    return search_space


def improved_itransformer_searchspace():
    """Enhanced iTransformer search space"""
    search_space = {
        "batch_size": tune.choice([32, 64, 128]),
        "d_ff": tune.choice([128, 256, 512, 1024]),
        "d_model": tune.choice([64, 128, 256, 512]),
        "e_layers": tune.choice([1, 2, 3, 4]),
        "n_heads": tune.choice([2, 4, 8, 12, 16]),
        
        # Regularization
        "dropout": tune.loguniform(1e-4, 0.3),
        "weight_decay": tune.loguniform(1e-6, 1e-2),
        
        # Learning
        "lr": tune.loguniform(1e-5, 1e-2),
        "lradj": tune.choice(["type1", "type2", "cosine"]),
        
        # Model-specific
        "period_len": tune.choice([8, 16, 32, 50, 100, 200]),
        
        # Fixed
        "horizon": 400,
        "seq_len": 1600,
        "loss": "MAE",
        "norm": True,
        "num_epochs": config.max_epochs,
        "patience": tune.choice([10, 15, 20]),
        "moving_avg": tune.choice([1, 3, 5, 7]),
        
        # Additional
        "grad_clip": tune.uniform(0.5, 2.0),
    }
    return search_space


def improved_dlinear_searchspace():
    """Enhanced DLinear search space"""
    search_space = {
        "batch_size": tune.choice([32, 64, 128, 256]),  # DLinear can handle larger batches
        "d_ff": tune.choice([64, 128, 256, 512, 1024]),
        
        # Learning parameters
        "lr": tune.loguniform(1e-5, 1e-1),  # DLinear can use higher LR
        "lradj": tune.choice(["type1", "type2", "cosine"]),
        "weight_decay": tune.loguniform(1e-6, 1e-2),
        
        # Regularization
        "dropout": tune.loguniform(1e-4, 0.2),  # Less dropout needed for simpler model
        
        # Fixed
        "horizon": 400,
        "seq_len": 1600,
        "loss": "MAE",
        "norm": True,
        "num_epochs": config.max_epochs,
        "patience": tune.choice([5, 10, 15]),  # Can be more aggressive
        "moving_avg": tune.choice([1, 3, 5, 7, 13]),  # More options for linear model
        
        # Additional
        "grad_clip": tune.uniform(0.5, 1.5),
    }
    return search_space


def get_conditional_search_space(model_name: str, base_search_space: dict):
    """Add conditional hyperparameters based on other parameters"""
    
    # Example: Make some parameters conditional on others
    if model_name in ["timexer", "crossformer"]:
        # Conditional learning rate based on model size
        def conditional_lr(config):
            base_lr = config["lr"]
            if config["d_model"] > 256:
                return base_lr * 0.8  # Reduce LR for larger models
            return base_lr
        
        # This would require more advanced conditional logic in Ray Tune
        # For now, we'll use the improved static spaces
    
    return base_search_space


def multi_fidelity_search_space(base_space: dict, fidelity_param: str = "num_epochs"):
    """Add multi-fidelity support to search space"""
    multi_fidelity_space = base_space.copy()
    
    # Add fidelity parameter
    multi_fidelity_space[fidelity_param] = tune.randint(10, config.max_epochs)
    
    return multi_fidelity_space


# Updated assembly function with improved search spaces
def improved_assemble_setup(setup_name):
    """Assemble setup with improved search spaces"""
    setups = {
        "duet": (
            config.default_eval_config(),
            config.model_config(
                "duet.DUET", decoder_input_required=False, has_loss_importance=True
            ),
            improved_duet_searchspace(),
        ),
        "crossformer": (
            config.default_eval_config(),
            config.model_config(
                "time_series_library.Crossformer", "transformer_adapter"
            ),
            improved_crossformer_searchspace(),
        ),
        "itransformer": (
            config.default_eval_config(),
            config.model_config(
                "time_series_library.iTransformer", "transformer_adapter"
            ),
            improved_itransformer_searchspace(),
        ),
        "timexer": (
            config.default_eval_config(),
            config.model_config(
                "timexer.TimeXer", decoder_input_required=True, has_loss_importance=False
            ),
            improved_timexer_searchspace(),
        ),
        "dlinear": (
            config.default_eval_config(),
            config.model_config("time_series_library.DLinear", "transformer_adapter"),
            improved_dlinear_searchspace(),
        ),
    }

    return setups.get(setup_name, setups["timexer"])  # Default to timexer if not found