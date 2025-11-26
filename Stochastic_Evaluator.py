import numpy as np
import pandas as pd
import json
import os
import glob
from simulator import SVRPTW_Simulator
from deterministic_policy_generator import load_instance

# --- CONFIGURATION ---
N_SIMULATIONS = 30 # Reduced sample size for full-history storage

def evaluate_strategy_stochastically(instance_path, strategy_data, results_dir):
    """
    Runs a deterministic policy against N_SIMULATIONS stochastic days.
    Saves FULL sequence data for every vehicle in every simulation.
    """
    
    # 1. Initialize Simulator
    instance_data = load_instance(instance_path)
    if isinstance(instance_data['customers'], pd.DataFrame):
        instance_data['customers'] = instance_data['customers'].to_dict(orient='records')

    simulator = SVRPTW_Simulator(instance_data)
    routes = strategy_data['routes']
    
    # 2. Run Simulations and Collect Full History
    daily_results_storage = []
    
    # Track aggregates for summary
    costs = []
    lates = []

    for day_idx in range(N_SIMULATIONS):
        day_result = simulator.run_policy_for_day(routes)
        
        costs.append(day_result['total_cost'])
        lates.append(day_result['hard_late_penalty_count'])
        
        # Store the full day object
        daily_results_storage.append({
            'day_index': day_idx,
            'total_cost': day_result['total_cost'],
            'total_distance': day_result['total_distance_mi'],
            'hard_lates': day_result['hard_late_penalty_count'],
            'vehicle_traces': day_result['vehicle_traces'] # Detailed Actions
        })

    # 3. Calculate Summary Statistics
    summary_stats = {
        'mean_cost': np.mean(costs),
        'std_cost': np.std(costs),
        'min_cost': np.min(costs),
        'max_cost': np.max(costs),
        'mean_hard_lates': np.mean(lates),
        'std_hard_lates': np.std(lates)
    }
    
    # 4. Save Heavy JSON
    base_filename = os.path.basename(instance_path).replace('.json', '')
    output_filename = f"{base_filename}_stochastic_results.json"
    output_filepath = os.path.join(results_dir, output_filename)
    
    final_output = {
        'instance_file': strategy_data['instance_file'],
        'policy_type': 'ORTools_Deterministic',
        'N_simulations': N_SIMULATIONS,
        'summary_metrics': summary_stats,
        'daily_simulation_logs': daily_results_storage # The Full Sequences
    }

    with open(output_filepath, 'w') as f:
        json.dump(final_output, f, indent=4)

def run_batch_evaluation():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    instance_data_dir = os.path.join(script_dir, 'instances', 'data')
    strategy_source_dir = os.path.join(script_dir, 'solutions', 'ORTools', 'strategy')
    results_target_dir = os.path.join(script_dir, 'solutions', 'ORTools', 'simulation_results')
    
    os.makedirs(results_target_dir, exist_ok=True)
    
    strategy_files = sorted(glob.glob(os.path.join(strategy_source_dir, '*_deterministic_strategy.json')))
    
    print(f"\n--- OR-Tools Stochastic Eval (N={N_SIMULATIONS}, Full History) ---")
    
    for i, strategy_filepath in enumerate(strategy_files):
        strategy_filename = os.path.basename(strategy_filepath)
        base_name = strategy_filename.replace('_deterministic_strategy.json', '')
        instance_filepath = os.path.join(instance_data_dir, f"{base_name}.json")
        
        if not os.path.exists(instance_filepath): continue
            
        with open(strategy_filepath, 'r') as f:
            strategy_data = json.load(f)
        
        if i % 5 == 0: print(f"Processing {i+1}/{len(strategy_files)}: {base_name}")
            
        evaluate_strategy_stochastically(instance_filepath, strategy_data, results_target_dir)

    print(f"Results saved to: {results_target_dir}")

if __name__ == '__main__':
    run_batch_evaluation()