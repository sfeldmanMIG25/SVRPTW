import numpy as np
import pandas as pd
import json
import os
import glob
from simulator import SVRPTW_Simulator
from deterministic_policy_generator import load_instance # Reuse load function
from strategy_visualizer import visualize_strategy # Reuse visualization

# --- CONFIGURATION ---
N_SIMULATIONS = 100 # Number of stochastic "days" to run per instance

def evaluate_strategy_stochastically(instance_path, strategy_data, results_dir, visuals_dir):
    """
    Runs a deterministic policy (strategy) against N_SIMULATIONS stochastic days,
    aggregates the results, and saves the statistics.
    """
    
    # 1. Initialize Simulator
    instance_data = load_instance(instance_path)
    
    # CRITICAL FIX: Convert customers DataFrame to list of dicts for the Simulator class
    if isinstance(instance_data['customers'], pd.DataFrame):
        instance_data['customers'] = instance_data['customers'].to_dict(orient='records')

    simulator = SVRPTW_Simulator(instance_data)
    
    routes = strategy_data['routes']
    
    # 2. Run Simulations
    all_day_results = []
    
    # print(f"  -> Running {N_SIMULATIONS} stochastic days...")

    for _ in range(N_SIMULATIONS):
        day_result = simulator.run_policy_for_day(routes)
        all_day_results.append(day_result)

    # 3. Aggregate Results (Calculate Mean and Standard Deviation)
    df_results = pd.DataFrame(all_day_results)
    
    # Calculate aggregate statistics
    agg_stats = {
        'policy_type': 'Deterministic_ORTools_Evaluated',
        'N_simulations': N_SIMULATIONS,
        'deterministic_cost_benchmark': strategy_data['total_cost'], 
        
        'mean_stochastic_cost': df_results['total_cost'].mean(),
        'std_stochastic_cost': df_results['total_cost'].std(),
        
        'mean_hard_late_penalties': df_results['hard_late_penalty_count'].mean(),
        'std_hard_late_penalties': df_results['hard_late_penalty_count'].std(),
        
        # FIX: Use 'total_service_efficiency' to match simulator output
        'mean_service_efficiency': df_results['total_service_efficiency'].mean(),
        'mean_transit_min': df_results['total_transit_min'].mean(),
        'mean_wait_min': df_results.get('total_wait_min', df_results.get('total_billable_wait_min', 0)).mean(),
    }
    
    # 4. Save Aggregated Results
    base_filename = os.path.basename(instance_path).replace('.json', '')
    output_filename = f"{base_filename}_stochastic_results.json"
    output_filepath = os.path.join(results_dir, output_filename)
    
    # Merge metadata from strategy data
    final_output = {
        'instance_file': strategy_data['instance_file'],
        'N': strategy_data['N'],
        'V': strategy_data['V'],
        'Q': strategy_data['Q'],
        'strategy_solve_time_seconds': strategy_data.get('solve_time_seconds', 0),
        'metrics': agg_stats
    }

    with open(output_filepath, 'w') as f:
        json.dump(final_output, f, indent=4)
    
    # 5. Generate Visualization (Optional)
    # visualize_strategy(instance_data, strategy_data, visuals_dir)


def run_batch_evaluation():
    """Manages the entire process: finding strategies and running stochastic evaluation."""
    
    # PATH FIX: Use script location, not CWD
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define Source and Target Directories
    instance_data_dir = os.path.join(script_dir, 'instances', 'data')
    strategy_source_dir = os.path.join(script_dir, 'solutions', 'ORTools', 'strategy')
    results_target_dir = os.path.join(script_dir, 'solutions', 'ORTools', 'simulation_results')
    visuals_target_dir = os.path.join(script_dir, 'solutions', 'ORTools', 'visuals')
    
    os.makedirs(results_target_dir, exist_ok=True)
    os.makedirs(visuals_target_dir, exist_ok=True)
    
    # 1. Check availability
    if not os.path.exists(strategy_source_dir):
        print(f"Strategy directory not found: {strategy_source_dir}")
        print("Please run deterministic_policy_generator.py first.")
        return

    # 2. Find Strategy Files
    strategy_files = sorted(glob.glob(os.path.join(strategy_source_dir, '*_deterministic_strategy.json')))
    
    if not strategy_files:
        print("Error: No strategy files found.")
        return

    print(f"\n--- Starting Stochastic Evaluation on {len(strategy_files)} Strategies ---")
    print(f"Reading strategies from: {strategy_source_dir}")
    
    for i, strategy_filepath in enumerate(strategy_files):
        strategy_filename = os.path.basename(strategy_filepath)
        
        # Infer the original instance file name from the strategy file
        base_name = strategy_filename.replace('_deterministic_strategy.json', '')
        instance_filepath = os.path.join(instance_data_dir, f"{base_name}.json")
        
        if not os.path.exists(instance_filepath):
            print(f"Warning: Original instance file not found for {base_name}. Skipping.")
            continue
            
        with open(strategy_filepath, 'r') as f:
            strategy_data = json.load(f)
        
        if i % 10 == 0:
            print(f"Evaluating {i+1}/{len(strategy_files)}: {strategy_filename}")
            
        evaluate_strategy_stochastically(instance_filepath, strategy_data, results_target_dir, visuals_target_dir)

    print("\n--- Batch Stochastic Evaluation Complete ---")
    print(f"Results saved to: {results_target_dir}")
    print(f"Visuals saved to: {visuals_target_dir}")


if __name__ == '__main__':
    run_batch_evaluation()