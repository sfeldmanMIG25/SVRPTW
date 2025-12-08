import optuna
import os
import glob
import numpy as np
import json
from functools import partial
from concurrent.futures import ProcessPoolExecutor

# Import your existing solver classes
# NOTE: You must update auction_solver.py to accept 'config' (see Step 2)
from auction_solver import RobustCentralizedController, load_instance

# --- CONFIGURATION ---
TUNE_INSTANCES_COUNT = 10  # Number of files to use for tuning (subset)
TUNE_DAYS = 15            # Simulation days per file (shorter for speed)
N_TRIALS = 200             # Total optimization trials

def objective(trial):
    """
    Optuna objective function. 
    Defines the search space and evaluates the solver performance.
    """
    # 1. Define Hyperparameter Search Space
    params = {
        # Estimator Priors
        'travel_alpha': trial.suggest_float('travel_alpha', 1.0, 5.0),
        'travel_beta': trial.suggest_float('travel_beta', 1.0, 5.0),
        
        # Risk Logic (The "Magic Numbers" in get_travel_risk)
        'risk_buffer_high': trial.suggest_float('risk_buffer_high', 45.0, 90.0), # Was 60.0
        'risk_buffer_mid': trial.suggest_float('risk_buffer_mid', 20.0, 45.0),   # Was 30.0
        'risk_buffer_low': trial.suggest_float('risk_buffer_low', 5.0, 20.0),    # Was 15.0
        'risk_ramp_steepness': trial.suggest_float('risk_ramp_steepness', 0.5, 2.0), # Multiplier scaler
        
        # Pruning & Assignment
        'pruning_multiplier': trial.suggest_float('pruning_multiplier', 1.5, 4.0), # Was 2.5
        'hungarian_rounds': trial.suggest_int('hungarian_rounds', 2, 8),           # Was 4
        'hungarian_batch_size': trial.suggest_int('hungarian_batch_size', 15, 50), # Was 30
        
        # Replanning
        'replan_buffer_ratio': trial.suggest_float('replan_buffer_ratio', 0.1, 0.4) # Was 0.2
    }
    
    # 2. Load Subset of Data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    instance_dir = os.path.join(script_dir, 'instances', 'data')
    all_files = sorted(glob.glob(os.path.join(instance_dir, '*.json')))
    
    # Deterministically select the same subset every time so trials are comparable
    rng = np.random.RandomState(42)
    selected_files = rng.choice(all_files, min(len(all_files), TUNE_INSTANCES_COUNT), replace=False)
    
    total_costs = []
    
    # 3. Run Simulation
    # We run this sequentially here to avoid fighting for resources with Optuna
    for filepath in selected_files:
        try:
            data = load_instance(filepath)
            # Handle the dataframe vs list quirk
            if not isinstance(data['customers'], list):
                data['customers'] = data['customers'].to_dict(orient='records')
            
            # Initialize Controller with dynamic params
            controller = RobustCentralizedController(data, config=params)
            
            daily_costs = []
            for d in range(TUNE_DAYS):
                res = controller.run_day(d)
                daily_costs.append(res['total_cost'])
            
            total_costs.append(np.mean(daily_costs))
            
        except Exception as e:
            # If a set of params breaks the solver, punish it heavily
            print(f"Trial failed: {e}")
            return float('inf')

    # Metric to minimize: Average Total Cost across selected instances
    return np.mean(total_costs)

def run_tuner():
    print("--- Starting Hyperparameter Optimization ---")
    
    # Create study
    study = optuna.create_study(direction='minimize')
    
    # Optimize
    study.optimize(objective, n_trials=N_TRIALS)
    
    print("\n--- Tuning Complete ---")
    print("Best Parameters:")
    print(json.dumps(study.best_params, indent=4))
    print(f"Best Cost: {study.best_value:,.2f}")
    
    # Save best params to file
    with open('best_solver_params.json', 'w') as f:
        json.dump(study.best_params, f, indent=4)

if __name__ == "__main__":
    run_tuner()