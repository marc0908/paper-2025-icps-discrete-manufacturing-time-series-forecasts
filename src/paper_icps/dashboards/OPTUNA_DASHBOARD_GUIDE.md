# Optuna Dashboard Integration Guide

This guide shows how to use the Optuna dashboard with your hyperparameter optimization.

## Quick Start

### 1. Install Optuna Dashboard
```bash
pip install optuna-dashboard
# or use the helper script
python optuna_dashboard_helper.py --install
```

### 2. Run Hyperparameter Optimization with Dashboard
```bash
# Enable dashboard with default SQLite storage
python improved_hyperparam_tune.py --model-setup timexer --num-samples 50 --enable-dashboard

# Custom storage location  
python improved_hyperparam_tune.py --model-setup timexer --num-samples 50 --enable-dashboard --storage-url "sqlite:///~/my_studies.db"
```

### 3. Launch Dashboard (in another terminal)
```bash
# Using the helper script
python optuna_dashboard_helper.py

# Or directly with optuna-dashboard
optuna-dashboard sqlite:///~/ray_results/timexer_optimization.db
```

### 4. View Dashboard
Open your browser to: `http://localhost:8080`

## Detailed Usage

### Running Optimization with Dashboard

```bash
# Basic usage with dashboard
python improved_hyperparam_tune.py \
    --model-setup timexer \
    --num-samples 100 \
    --enable-dashboard \
    --experiment-name "timexer_experiment_v1"

# Advanced usage with custom settings
python improved_hyperparam_tune.py \
    --model-setup timexer \
    --num-samples 200 \
    --enable-dashboard \
    --storage-url "sqlite:///~/experiments/timexer_study.db" \
    --cpu-per-trial 0.25 \
    --max-concurrent 20 \
    --scheduler asha_improved \
    --search-algo optuna
```

### Dashboard Features

The Optuna dashboard provides:

1. **Real-time Optimization Progress**: Watch trials as they complete
2. **Parameter Importance**: See which hyperparameters matter most
3. **History Plots**: Visualize optimization convergence
4. **Parallel Coordinate Plots**: Understand parameter relationships
5. **Slice Plots**: Analyze individual parameter effects
6. **Trial Details**: Inspect individual trial configurations and results

### Multiple Studies

You can run multiple optimization studies and view them all in one dashboard:

```bash
# Study 1: TimeXer
python improved_hyperparam_tune.py --model-setup timexer --enable-dashboard --experiment-name "timexer_v1"

# Study 2: Crossformer  
python improved_hyperparam_tune.py --model-setup crossformer --enable-dashboard --experiment-name "crossformer_v1"

# Study 3: DLinear
python improved_hyperparam_tune.py --model-setup dlinear --enable-dashboard --experiment-name "dlinear_v1"
```

All studies will appear in the same dashboard if using the same storage URL.

### Dashboard Helper Script

The `optuna_dashboard_helper.py` script provides convenient utilities:

```bash
# Install optuna-dashboard
python optuna_dashboard_helper.py --install

# Create a sample study for testing
python optuna_dashboard_helper.py --create-sample --study-name "test_study"

# Launch dashboard with custom settings
python optuna_dashboard_helper.py --port 8081 --host 127.0.0.1

# Use custom storage
python optuna_dashboard_helper.py --storage "sqlite:///~/my_experiments.db"
```

## Storage Options

### SQLite (Default)
```bash
--storage-url "sqlite:///~/ray_results/my_study.db"
```
- ✅ Simple setup, no external dependencies
- ✅ Persists across runs
- ❌ Single machine only

### PostgreSQL (Advanced)
```bash
--storage-url "postgresql://user:password@localhost/optuna"
```
- ✅ Multi-machine access
- ✅ Better concurrent performance
- ❌ Requires database setup

### MySQL (Advanced)
```bash
--storage-url "mysql://user:password@localhost/optuna"
```
- ✅ Multi-machine access
- ✅ Production ready
- ❌ Requires database setup

## Integration with Your Workflow

### 1. Development Phase
```bash
# Quick test with small samples and dashboard
python improved_hyperparam_tune.py \
    --model-setup timexer \
    --num-samples 20 \
    --enable-dashboard \
    --cpu-per-trial 1.0 \
    --max-concurrent 5
```

### 2. Production Optimization
```bash
# Large-scale optimization with dashboard monitoring
python improved_hyperparam_tune.py \
    --model-setup timexer \
    --num-samples 500 \
    --enable-dashboard \
    --storage-url "sqlite:///~/production/timexer_optimization.db" \
    --cpu-per-trial 0.25 \
    --max-concurrent 20 \
    --scheduler asha_improved
```

### 3. Resume Interrupted Runs
```bash
# Resume with same storage URL - Optuna will continue from where it left off
python improved_hyperparam_tune.py \
    --model-setup timexer \
    --num-samples 500 \
    --enable-dashboard \
    --storage-url "sqlite:///~/production/timexer_optimization.db" \
    --resume
```

## Monitoring and Analysis

### Real-time Monitoring
1. Start your optimization with `--enable-dashboard`
2. Launch dashboard in another terminal
3. Monitor progress in real-time at `http://localhost:8080`

### Post-optimization Analysis
1. Keep the dashboard running after optimization completes
2. Analyze parameter importance and relationships
3. Export results for further analysis

### Best Practices

1. **Use descriptive experiment names**: Makes it easier to identify studies
2. **Monitor resource usage**: Adjust `--cpu-per-trial` and `--max-concurrent` based on your hardware
3. **Backup storage files**: SQLite files contain all your optimization history
4. **Use consistent storage URLs**: To accumulate results across multiple runs

## Troubleshooting

### Dashboard Won't Start
```bash
# Check if optuna-dashboard is installed
pip list | grep optuna-dashboard

# Reinstall if needed
pip install --upgrade optuna-dashboard
```

### Database Locked Errors
- Ensure only one optimization process writes to the same database
- Use different storage URLs for concurrent optimizations

### Can't Access Dashboard
- Check firewall settings for port 8080
- Try `--host 127.0.0.1` for local-only access
- Use different port with `--port 8081`

## Example Complete Workflow

```bash
# Terminal 1: Start optimization with dashboard
cd /work/paper-2025-icps-discrete-manufacturing-time-series-forecasts/src
python improved_hyperparam_tune.py \
    --model-setup timexer \
    --num-samples 100 \
    --enable-dashboard \
    --experiment-name "timexer_production_v1" \
    --cpu-per-trial 0.5 \
    --max-concurrent 16

# Terminal 2: Launch dashboard  
python optuna_dashboard_helper.py

# Browser: Open http://localhost:8080
# Monitor optimization progress in real-time!
```

This setup gives you powerful visualization and monitoring capabilities for your hyperparameter optimization process.