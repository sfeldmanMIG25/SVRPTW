import numpy as np
import pandas as pd
import os
import json
import re
from collections import defaultdict
from config import DEPOT_E_TIME, DEPOT_L_TIME, HARD_LATE_PENALTY

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
    Computes Sample Standard Deviation (ddof=1) for:
      - Pure Op Cost (No Penalties)
      - Full Cost (Op + Missed + Lates)
      - Failures (Missed + Lates)
      - Utilization
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
            fleet_capacity_min = 1.0 

        # Vectors
        vec_op_cost = []
        vec_full_cost = []
        vec_failures = []
        vec_util = []

        for log in logs:
            # 1. Extract Raw Data
            # Note: Solvers report 'total_cost' which typically includes Hard Late penalties
            # We must strip them to get "Pure Op Cost"
            reported_total_cost = log.get('total_cost', 0.0)
            
            # Count Failures
            missed = log.get('missed_customers', 0)
            # Handle key variations for hard lates
            hard_lates = log.get('hard_lates', log.get('hard_late_count', 0))
            
            total_failures = missed + hard_lates
            
            # 2. Derive Pure Op Cost (Wages + Transit)
            # Remove the penalty component that exists in reported_total_cost
            # Simulator logic: total_cost = wages + transit + (hard_lates * 1000)
            pure_op_cost = reported_total_cost - (hard_lates * HARD_LATE_PENALTY)
            
            # 3. Derive Full Economic Cost
            # Full = Pure Op + (All Failures * Penalty)
            full_cost = pure_op_cost + (total_failures * HARD_LATE_PENALTY)
            
            # 4. Utilization
            total_service_min = 0.0
            if 'vehicle_traces' in log:
                for trace in log['vehicle_traces']:
                    for step in trace:
                        total_service_min += step.get('service_duration', 0.0)
            util_ratio = total_service_min / max(1, fleet_capacity_min)

            # Store
            vec_op_cost.append(pure_op_cost)
            vec_full_cost.append(full_cost)
            vec_failures.append(total_failures)
            vec_util.append(util_ratio)

        # --- Compute Statistics (ddof=1 for Sample Std Dev) ---
        if len(logs) > 1:
            std_op = np.std(vec_op_cost, ddof=1)
            std_full = np.std(vec_full_cost, ddof=1)
            std_fail = np.std(vec_failures, ddof=1)
            std_util = np.std(vec_util, ddof=1)
        else:
            std_op = 0.0; std_full = 0.0; std_fail = 0.0; std_util = 0.0

        return {
            'policy': data.get('policy_type', 'Unknown'),
            # Means
            'mean_op': np.mean(vec_op_cost),
            'mean_full': np.mean(vec_full_cost),
            'mean_fail': np.mean(vec_failures),
            'mean_util': np.mean(vec_util),
            # Deviations (Average of these will be taken later)
            'std_op': std_op,
            'std_full': std_full,
            'std_fail': std_fail,
            'std_util': std_util
        }

    except Exception as e:
        # print(f"Error parsing {filepath}: {e}")
        return None

def run_aggregator():
    print(f"--- Metrics Aggregator (Revised Cost Logic) ---")
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
    # Layout: Policy | N | Op (Mean/Std) | Full (Mean/Std) | Fail (Mean/Std) | Util (Mean/Std)
    
    header = (f"{'Policy':<22} | {'N':<3} | "
              f"{'Op Mean':<8} | {'Op Std':<7} | "
              f"{'Full Mean':<9} | {'Full Std':<8} | "
              f"{'Fail':<5} | {'F.Std':<5} | "
              f"{'Util':<5} | {'U.Std':<5}")
              
    print("\n" + "="*115)
    print(header)
    print("-" * 115)
    
    for policy, results in sorted(policy_stats.items()):
        n = len(results)
        
        # Grand Averages (Mean of Means, Mean of Stds)
        
        # Op Cost
        avg_op = np.mean([r['mean_op'] for r in results])
        avg_op_std = np.mean([r['std_op'] for r in results])
        
        # Full Cost
        avg_full = np.mean([r['mean_full'] for r in results])
        avg_full_std = np.mean([r['std_full'] for r in results])
        
        # Failures
        avg_fail = np.mean([r['mean_fail'] for r in results])
        avg_fail_std = np.mean([r['std_fail'] for r in results])
        
        # Utilization
        avg_util = np.mean([r['mean_util'] for r in results])
        avg_util_std = np.mean([r['std_util'] for r in results])
        
        print(f"{policy:<22} | {n:<3} | "
              f"${avg_op:<7,.0f} | ${avg_op_std:<6,.0f} | "
              f"${avg_full:<8,.0f} | ${avg_full_std:<7,.0f} | "
              f"{avg_fail:<5.2f} | {avg_fail_std:<5.2f} | "
              f"{avg_util*100:<4.1f}% | {avg_util_std*100:<4.2f}%") 

    print("="*115)
    print(f"Logic: Full Cost = Pure Op Cost + ((Missed + Hard Lates) * ${HARD_LATE_PENALTY})")
    print("       Pure Op Cost = Wages + Transit (No Penalties)")
    print("       Std Devs are calculated per instance (across 30 days) then averaged.")

if __name__ == '__main__':
    run_aggregator()