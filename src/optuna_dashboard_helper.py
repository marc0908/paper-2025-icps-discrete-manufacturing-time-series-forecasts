#!/usr/bin/env python3
"""
Optuna Dashboard Setup and Launch Script
"""
import os
import subprocess
import argparse
import time
import webbrowser
from pathlib import Path


def install_optuna_dashboard():
    """Install optuna-dashboard if not already installed"""
    try:
        import optuna_dashboard
        print("✓ optuna-dashboard is already installed")
        return True
    except ImportError:
        print("Installing optuna-dashboard...")
        try:
            subprocess.check_call(["pip", "install", "optuna-dashboard"])
            print("✓ optuna-dashboard installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install optuna-dashboard: {e}")
            return False


def create_sample_study(storage_url: str, study_name: str):
    """Create a sample study for testing dashboard"""
    import optuna
    
    def objective(trial):
        x = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        y = trial.suggest_int("batch_size", 32, 128)
        z = trial.suggest_categorical("optimizer", ["adam", "sgd", "rmsprop"])
        
        # Simulate validation loss
        import random
        loss = random.uniform(0.1, 1.0) + x * 100 + (y - 64) ** 2 / 1000
        return loss
    
    study = optuna.create_study(
        storage=storage_url,
        study_name=study_name,
        direction="minimize",
        load_if_exists=True
    )
    
    print(f"Creating sample study '{study_name}' with 20 trials...")
    study.optimize(objective, n_trials=20)
    print(f"✓ Sample study created with {len(study.trials)} trials")
    return study


def initialize_database(storage_url: str):
    """Initialize Optuna database if it doesn't exist or is corrupted"""
    try:
        import optuna
        
        # Try to create/access the storage to ensure it's properly initialized
        storage = optuna.storages.RDBStorage(storage_url)
        
        # Create a dummy study to initialize the database schema
        study_name = "database_init_test"
        try:
            study = optuna.create_study(
                storage=storage,
                study_name=study_name,
                direction="minimize",
                load_if_exists=True
            )
            
            # Clean up the test study
            try:
                study.delete_study(study_name)
            except:
                pass  # Ignore errors during cleanup
                
        except Exception as e:
            print(f"Note: Database initialization encountered: {e}")
            
        print("✓ Database initialized successfully")
        return True
        
    except Exception as e:
        print(f"✗ Failed to initialize database: {e}")
        return False


def launch_dashboard(storage_url: str, port: int = 8080, host: str = "0.0.0.0"):
    """Launch Optuna dashboard"""
    print(f"Launching Optuna dashboard...")
    print(f"Storage: {storage_url}")
    print(f"URL: http://localhost:{port}")
    print(f"Host: {host}")
    
    # Initialize database first
    print("Initializing database...")
    if not initialize_database(storage_url):
        print("Warning: Database initialization failed, but proceeding anyway...")
    
    print("\nTo stop the dashboard, press Ctrl+C")
    
    try:
        # Launch dashboard
        cmd = ["optuna-dashboard", storage_url, "--host", host, "--port", str(port)]
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
    except FileNotFoundError:
        print("✗ optuna-dashboard command not found. Make sure it's installed and in PATH.")
        return False
    except Exception as e:
        print(f"✗ Error launching dashboard: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Optuna Dashboard Helper")
    parser.add_argument("--storage", type=str, 
                       default="sqlite:///~/ray_results/hyperopt_dashboard.db",
                       help="Storage URL for Optuna studies")
    parser.add_argument("--port", type=int, default=8080,
                       help="Port for dashboard server")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                       help="Host for dashboard server")
    parser.add_argument("--create-sample", action="store_true",
                       help="Create a sample study for testing")
    parser.add_argument("--install", action="store_true",
                       help="Install optuna-dashboard")
    parser.add_argument("--study-name", type=str, default="sample_study",
                       help="Name for sample study")
    parser.add_argument("--clean-db", action="store_true",
                       help="Clean/reset corrupted database")
    
    args = parser.parse_args()
    
    # Normalize storage URL
    storage_url = args.storage
    
    # Handle different input formats
    if not storage_url.startswith("sqlite:///"):
        # If it's just a file path, convert to SQLite URL
        if storage_url.startswith("~/"):
            expanded_path = os.path.expanduser(storage_url)
        else:
            expanded_path = storage_url
        storage_url = f"sqlite:///{expanded_path}"
    
    # Handle tilde expansion in SQLite URLs
    if storage_url.startswith("sqlite:///~/"):
        expanded_path = os.path.expanduser(storage_url.replace("sqlite:///~/", "~/"))
        storage_url = f"sqlite:///{expanded_path}"
    
    # Extract actual file path for directory creation
    if storage_url.startswith("sqlite:///"):
        db_path = storage_url.replace("sqlite:///", "")
        db_dir = os.path.dirname(db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
    
    # Install dashboard if requested
    if args.install:
        if not install_optuna_dashboard():
            return
    
    # Clean database if requested
    if args.clean_db:
        if storage_url.startswith("sqlite:///"):
            db_path = storage_url.replace("sqlite:///", "")
            if os.path.exists(db_path):
                print(f"Removing corrupted database: {db_path}")
                os.remove(db_path)
                print("✓ Database cleaned")
            else:
                print("Database file doesn't exist, nothing to clean")
        else:
            print("Clean database option only works with SQLite databases")
        return
    
    # Create sample study if requested
    if args.create_sample:
        if not install_optuna_dashboard():
            print("Cannot create sample study without optuna-dashboard")
            return
        try:
            create_sample_study(storage_url, args.study_name)
        except Exception as e:
            print(f"✗ Error creating sample study: {e}")
            return
    
    # Launch dashboard
    launch_dashboard(storage_url, args.port, args.host)


if __name__ == "__main__":
    main()