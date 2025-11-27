import time
import json
import torch
import argparse
import os
import gc
import numpy as np
from typing import Dict, Any

from paper_icps.core import training, config
from paper_icps.tuning import search_spaces

def measure_vram_and_time(func, *args, **kwargs):
    """
    Wrapper, Meassuing VRAM and Time
    """
    # 1. Cleanup & Reset
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    
    start_time = time.time()
    
    # 2. Run Function
    result = func(*args, **kwargs)
    
    # 3. Measure
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        max_vram_bytes = torch.cuda.max_memory_allocated()
        max_vram_mb = max_vram_bytes / (1024 ** 2)
    else:
        max_vram_mb = 0.0
        
    end_time = time.time()
    duration = end_time - start_time
    
    return result, duration, max_vram_mb

def run_benchmark(
    model_name: str,
    custom_hparams: Dict[str, Any],
    data_path: str,
    data_path_overrides: str,
    device: str = "cuda"
):
    print(f"=== Starting Benchmark for {model_name} ===")
    
    # 1. Load Base Configs
    # Get Setup from search_spaces to get structure
    try:
        base_eval_cfg, base_model_cfg, default_search_space = search_spaces.assemble_setup(model_name)
    except ValueError:
        print(f"Model {model_name} not found in search_spaces.")
        return

    # 2. Merge HParams
    # We take defaults from Search Space (if available) and overwrite them with custom_hparams
    # Attention: Ray Tune Objects need to be resolved (tune.choice) if no custom params are available
    # Easy heuristics: If not custom HParam present - use default one
    
    final_hparams = {}
    
    # Set trivial defaults (everything that is not specified)
    # Important that it doesnt crash if user just chooses "lr"
    dummy_defaults = {
        "batch_size": 32,
        "d_model": 128,
        "d_ff": 256,
        "e_layers": 2,
        "n_heads": 4,
        "lr": 0.0001,
        "dropout": 0.1,
        "seq_len": 1600,
        "horizon": 400,
        "patch_len": 16,
        "factor": 3,
        "moving_avg": 25,
        # ... more defaults if needed
    }
    
    final_hparams.update(dummy_defaults)
    final_hparams.update(custom_hparams)
    
    # Inject in Model Config
    base_model_cfg["models"][0]["model_hyper_params"] = final_hparams
    
    # Set Epochs for Benchmark to low number (or high depending on target)
    base_model_cfg["models"][0]["num_epochs"] = final_hparams.get("num_epochs", 5)

    print("Hyperparameters:")
    print(json.dumps(final_hparams, indent=2))

    # 3. Run Training Benchmark
    print("\n>>> Running Training Benchmark...")
    train_result, train_time, train_vram = measure_vram_and_time(
        training.train_model,
        model_config=base_model_cfg,
        evaluation_config=base_eval_cfg,
        data_path=data_path,
        data_path_overrides=data_path_overrides,
        # Optional: Deactivate callbacks for pure performance
    )
    
    val_loss = train_result.get("val_loss", -1)
    
    # 4. Report
    print("\n" + "="*40)
    print(f"BENCHMARK RESULTS: {model_name}")
    print("="*40)
    print(f"Training Time:    {train_time:.2f} s")
    print(f"Max VRAM Usage:   {train_vram:.2f} MB")
    print(f"Validation Loss:  {val_loss:.6f}")
    print("="*40)
    
    # 5. Optional: Inference Speed Test (How fast is ONE prediction?)
    # We have to extract the trained model from train_results
    # but train_model returns only a dict
    # TODO: Implement
    
    return {
        "model": model_name,
        "train_time_s": train_time,
        "vram_mb": train_vram,
        "val_loss": val_loss,
        "hparams": final_hparams
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="timexer, crossformer, etc.")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON with HParams")
    # Quick overrides via CLI
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=3)
    
    args = parser.parse_args()
    
    # Paths (Hardcoded or via Env)
    DATA = "../dataset/pick_n_place_procedure_dataset.csv.zip"
    DATA_O = "../dataset/pick_n_place_procedure_w_overrides.csv.zip"
    
    # Load HParams
    hparams = {}
    if args.config:
        with open(args.config, 'r') as f:
            hparams = json.load(f)
            
    # CLI Overrides
    if args.batch_size:
        hparams["batch_size"] = args.batch_size
    hparams["num_epochs"] = args.epochs
    
    run_benchmark(args.model, hparams, DATA, DATA_O)