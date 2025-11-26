import numpy as np
import pandas as pd
import os
import json
import re
from collections import defaultdict
from config import DEPOT_E_TIME, DEPOT_L_TIME

# --- CONFIGURATION ---
SOLUTIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'solutions')

def get_metadata_from_filename(filename):
    """Parses N and V from filename (e.g., N020_V005_...) as fallback."""
    match = re.search(r'N(\d+)_V(\d+)', filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return 0, 0

def calculate_instance_metrics(filepath):
    """
    Parses a 'new style' JSON. 
    Computes Sample Standard Deviation (ddof=1) for Cost, Missed, and Util.
    """
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        if 'daily_simulation_logs' not in data:
            return None

        logs = data['daily_simulation_logs']
        if not logs:
            return None
            
        # Robust Retrieval of V (Vehicle Count)
        num_vehicles = data.get('V', 0)
        if num_vehicles == 0:
            _, num_vehicles = get_metadata_from_filename(os.path.basename(filepath))
            
        # Fleet Capacity (minutes)
        if num_vehicles > 0:
            fleet_capacity_min = num_vehicles * (DEPOT_L_TIME - DEPOT_E_TIME)
        else:
            fleet_capacity_min = 1.0 # Prevent div/0 if parsing fails

        daily_costs = []
        daily_missed = [] 
        daily_utils = []

        for log in logs:
            # 1. Cost
            daily_costs.append(log.get('total_cost', 0.0))
            
            # 2. Missed (Strict Missed + Hard Lates as Total Failures)
            failures = log.get('missed_customers', 0) + log.get('hard_lates', 0)
            daily_missed.append(failures)
            
            # 3. Utilization
            total_service_min = 0.0
            if 'vehicle_traces' in log:
                for trace in log['vehicle_traces']:
                    for step in trace:
                        total_service_min += step.get('service_duration', 0.0)
            
            util_ratio = total_service_min / max(1, fleet_capacity_min)
            daily_utils.append(util_ratio)

        # --- Compute Sample Standard Deviations (ddof=1) ---
        # If N < 2, Std Dev is 0
        if len(logs) > 1:
            std_cost = np.std(daily_costs, ddof=1)
            std_missed = np.std(daily_missed, ddof=1)
            std_util = np.std(daily_utils, ddof=1)
        else:
            std_cost = 0.0
            std_missed = 0.0
            std_util = 0.0

        return {
            'policy': data.get('policy_type', 'Unknown'),
            'mean_cost': np.mean(daily_costs),
            'std_cost': std_cost,
            'mean_missed': np.mean(daily_missed),
            'std_missed': std_missed,
            'mean_util': np.mean(daily_utils),
            'std_util': std_util
        }

    except Exception as e:
        return None

def run_aggregator():
    print(f"--- Metrics Aggregator (Grouped by Policy) ---")
    print(f"Scanning: {SOLUTIONS_DIR}")
    
    result_files = []
    for root, dirs, files in os.walk(SOLUTIONS_DIR):
        for file in files:
            if file.endswith('_results.json'):
                result_files.append(os.path.join(root, file))
    
    if not result_files:
        print("No result files found.")
        return

    # Accumulate metrics by policy
    policy_stats = defaultdict(list)
    
    for f in result_files:
        res = calculate_instance_metrics(f)
        if res:
            policy_stats[res['policy']].append(res)
            
    if not policy_stats:
        print("No valid new-style JSON files found.")
        return

    # --- Print Table ---
    # Layout: Policy | Count | Cost (Mean/Std) | Fail (Mean/Std) | Util (Mean/Std)
    header = (f"{'Policy / Method':<30} | {'Inst':<5} | "
              f"{'Avg Cost':<10} | {'Avg Std':<10} | "
              f"{'Avg Fail':<8} | {'Avg Std':<8} | "
              f"{'Avg Util':<8} | {'Avg Std':<8}")
              
    print("\n" + "="*115)
    print(header)
    print("-" * 115)
    
    for policy, results in sorted(policy_stats.items()):
        n = len(results)
        
        # Calculate Grand Averages across instances
        avg_cost = np.mean([r['mean_cost'] for r in results])
        avg_std_cost = np.mean([r['std_cost'] for r in results])
        
        avg_miss = np.mean([r['mean_missed'] for r in results])
        avg_std_miss = np.mean([r['std_missed'] for r in results])
        
        avg_util = np.mean([r['mean_util'] for r in results])
        avg_std_util = np.mean([r['std_util'] for r in results])
        
        print(f"{policy:<30} | {n:<5} | "
              f"${avg_cost:<9,.0f} | "
              f"${avg_std_cost:<9,.0f} | "
              f"{avg_miss:<8.2f} | "
              f"{avg_std_miss:<8.2f} | "
              f"{avg_util*100:<7.1f}% | "
              f"{avg_std_util*100:<7.2f}%") 

    print("="*115)
    print("Note: 'Avg Std' is the average of Sample Standard Deviations across instances.")

if __name__ == '__main__':
    run_aggregator()