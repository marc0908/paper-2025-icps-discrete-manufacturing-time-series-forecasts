#!/usr/bin/env python3
"""
Check what studies exist in an Optuna database
"""
import argparse
import os


def list_studies_in_database(storage_url):
    """List all studies in an Optuna database"""
    import optuna
    
    try:
        print(f"Connecting to database: {storage_url}")
        storage = optuna.storages.RDBStorage(storage_url)
        
        # Get all study summaries
        summaries = storage.get_all_study_summaries()
        
        if not summaries:
            print("No studies found in this database.")
            return
        
        print(f"\nFound {len(summaries)} studies:")
        print("-" * 80)
        
        for summary in summaries:
            print(f"Study Name: {summary.study_name}")
            print(f"Study ID: {summary.study_id}")
            print(f"Direction: {summary.direction}")
            print(f"Number of trials: {len(summary.trials) if hasattr(summary, 'trials') else 'Unknown'}")
            
            # Try to get more details
            try:
                study = optuna.load_study(
                    storage=storage,
                    study_name=summary.study_name
                )
                print(f"Trials: {len(study.trials)}")
                if study.trials:
                    best_trial = study.best_trial
                    print(f"Best value: {best_trial.value}")
                    print(f"Best params: {best_trial.params}")
            except Exception as e:
                print(f"Could not load study details: {e}")
            
            print("-" * 80)
            
    except Exception as e:
        print(f"Error accessing database: {e}")
        return


def main():
    parser = argparse.ArgumentParser(description="List Optuna Studies")
    parser.add_argument("storage", nargs='?',
                       default="sqlite:////root/ray_results/hyperopt_dashboard.db",
                       help="Storage URL (e.g., sqlite:///path/to/db.sqlite3)")
    
    args = parser.parse_args()
    
    # Handle file path vs URL
    storage_url = args.storage
    if not storage_url.startswith("sqlite:///"):
        if storage_url.startswith("~/"):
            storage_url = os.path.expanduser(storage_url)
        storage_url = f"sqlite:///{storage_url}"
    
    list_studies_in_database(storage_url)


if __name__ == "__main__":
    main()