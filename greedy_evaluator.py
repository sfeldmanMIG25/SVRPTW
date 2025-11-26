import numpy as np
import pandas as pd
import json
import os
import glob
from simulator import SVRPTW_Simulator
from deterministic_policy_generator import load_instance
from data_generator import euclidean_distance

# --- CONFIGURATION ---
N_SIMULATIONS = 30 # Reduced sample size

def run_greedy_policy_construction(simulator):
    """Generates the static greedy route plan."""
    num_vehicles = simulator.instance['num_vehicles']
    vehicle_capacity = simulator.instance['vehicle_capacity']
    unvisited = set(simulator.customer_map.keys()) - {0}
    
    vehicle_states = []
    for _ in range(num_vehicles):
        vehicle_states.append({
            'curr_node': 0, 'load': 0, 'route': [{'node_id': 0}], 'finished': False
        })
        
    while unvisited and any(not v['finished'] for v in vehicle_states):
        active = [v for v in vehicle_states if not v['finished']]
        if not active: break

        for v in active:
            if not unvisited:
                v['finished'] = True
                continue

            curr = v['curr_node']
            curr_coords = simulator.coordinates[curr]
            
            best_node = 0
            min_dist = float('inf')
            found = False
            
            for cid in unvisited:
                c_data = simulator._get_node_data(cid)
                if v['load'] + c_data['demand'] > vehicle_capacity: continue
                
                dist = euclidean_distance(curr_coords, simulator.coordinates[cid])
                if dist < min_dist:
                    min_dist = dist
                    best_node = cid
                    found = True
            
            if found:
                v['route'].append({'node_id': best_node})
                v['curr_node'] = best_node
                v['load'] += simulator._get_node_data(best_node)['demand']
                unvisited.discard(best_node)
            else:
                if v['route'][-1]['node_id'] != 0: v['route'].append({'node_id': 0})
                v['finished'] = True

    # Close loops
    for v in vehicle_states:
        if v['route'][-1]['node_id'] != 0: v['route'].append({'node_id': 0})
        
    return [v['route'] for v in vehicle_states]

def evaluate_greedy_stochastically(instance_path, results_dir):
    # 1. Setup
    instance_data = load_instance(instance_path)
    if isinstance(instance_data['customers'], pd.DataFrame):
        instance_data['customers'] = instance_data['customers'].to_dict(orient='records')

    simulator = SVRPTW_Simulator(instance_data)
    
    # 2. Construct Policy
    greedy_routes = run_greedy_policy_construction(simulator)
    
    # 3. Simulate Days
    daily_results_storage = []
    costs = []
    lates = []
    
    for day_idx in range(N_SIMULATIONS):
        day_result = simulator.run_policy_for_day(greedy_routes)
        
        costs.append(day_result['total_cost'])
        lates.append(day_result['hard_late_penalty_count'])
        
        daily_results_storage.append({
            'day_index': day_idx,
            'total_cost': day_result['total_cost'],
            'total_distance': day_result['total_distance_mi'],
            'hard_lates': day_result['hard_late_penalty_count'],
            'vehicle_traces': day_result['vehicle_traces']
        })
        
    # 4. Save
    summary_stats = {
        'mean_cost': np.mean(costs),
        'std_cost': np.std(costs),
        'mean_hard_lates': np.mean(lates),
        'std_hard_lates': np.std(lates)
    }
    
    base_filename = os.path.basename(instance_path).replace('.json', '')
    output_filename = f"{base_filename}_greedy_results.json"
    output_filepath = os.path.join(results_dir, output_filename)
    
    final_output = {
        'instance_file': os.path.basename(instance_path),
        'policy_type': 'Greedy_NearestNeighbor',
        'N_simulations': N_SIMULATIONS,
        'summary_metrics': summary_stats,
        'routes': greedy_routes, # The plan
        'daily_simulation_logs': daily_results_storage # The reality
    }

    with open(output_filepath, 'w') as f:
        json.dump(final_output, f, indent=4)

def run_batch_greedy_evaluation():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    instance_data_dir = os.path.join(script_dir, 'instances', 'data')
    results_target_dir = os.path.join(script_dir, 'solutions', 'Greedy', 'simulation_results')
    
    os.makedirs(results_target_dir, exist_ok=True)
    instance_files = sorted(glob.glob(os.path.join(instance_data_dir, '*.json')))
    
    print(f"\n--- Greedy Stochastic Eval (N={N_SIMULATIONS}, Full History) ---")
    
    for i, filepath in enumerate(instance_files):
        if i % 5 == 0: print(f"Processing {i+1}/{len(instance_files)}: {os.path.basename(filepath)}")
        evaluate_greedy_stochastically(filepath, results_target_dir)

    print(f"Results saved to: {results_target_dir}")

if __name__ == '__main__':
    run_batch_greedy_evaluation()