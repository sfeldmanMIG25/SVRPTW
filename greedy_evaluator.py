import numpy as np
import pandas as pd
import json
import os
import glob
import time
import matplotlib.pyplot as plt 
from simulator import SVRPTW_Simulator
from deterministic_policy_generator import load_instance
from data_generator import euclidean_distance

# --- CONFIGURATION ---
N_SIMULATIONS = 100 # Number of stochastic "days" to run per instance

# --- CORE GREEDY POLICY LOGIC (Sequential Decision Maker) ---
def run_greedy_policy(simulator: SVRPTW_Simulator):
    """
    Runs the Nearest Neighbor (Greedy) Policy to construct a static route plan.
    
    The Greedy heuristic:
    1. Starts at Depot.
    2. Always chooses the closest feasible (capacity-wise) unvisited customer.
    3. Ignores Time Windows for decision making (distance-only greedy), 
       but the Simulator will penalize this if it results in lateness.
    
    Returns:
        tuple: (Aggregated construction metrics, list of planned vehicle states including routes)
    """
    
    num_vehicles = simulator.instance['num_vehicles']
    vehicle_capacity = simulator.instance['vehicle_capacity']
    
    # Initialize the state
    unvisited_customer_ids = set(simulator.customer_map.keys()) - {0} # Exclude Depot
    
    vehicle_states = []
    for _ in range(num_vehicles):
        vehicle_states.append({
            'current_node_id': 0, # Start at depot
            'current_load': 0,
            'route_plan': [{'node_id': 0}], # Track the planned sequence
            'is_finished': False
        })
        
    # Loop while there are unvisited customers or unfinished vehicles
    while unvisited_customer_ids and any(not v['is_finished'] for v in vehicle_states):
        
        # Round-robin assignment or parallel? 
        # Standard Greedy Construction usually fills one vehicle then the next, 
        # or moves all vehicles one step at a time. 
        # Moving all one step at a time balances the fleet better.
        
        vehicles_to_process = [v for v in vehicle_states if not v['is_finished']]
        
        if not vehicles_to_process:
            break

        for vehicle in vehicles_to_process:
            if not unvisited_customer_ids:
                vehicle['is_finished'] = True
                continue

            current_node_id = vehicle['current_node_id']
            current_load = vehicle['current_load']
            
            # --- 1. Identify Feasible Next Steps (Nearest Neighbor) ---
            current_coord = simulator.coordinates[current_node_id]
            best_next_node_id = 0 # Default action: return to depot
            min_distance = float('inf')
            found_candidate = False
            
            # Find the Nearest Neighbor that is feasible
            for customer_id in unvisited_customer_ids:
                customer_data = simulator._get_node_data(customer_id)
                
                # Check 1: Capacity Feasibility
                if current_load + customer_data['demand'] > vehicle_capacity:
                    continue
                
                # Check 2: Distance
                target_coord = simulator.coordinates[customer_id]
                distance = euclidean_distance(current_coord, target_coord)
                
                if distance < min_distance:
                    min_distance = distance
                    best_next_node_id = customer_id
                    found_candidate = True
                    
            # --- 2. Execute Decision ---
            if found_candidate:
                # Go to Customer
                vehicle['route_plan'].append({'node_id': best_next_node_id})
                unvisited_customer_ids.discard(best_next_node_id)
                vehicle['current_node_id'] = best_next_node_id
                vehicle['current_load'] += simulator._get_node_data(best_next_node_id)['demand']
            else:
                # No feasible customer found (Capacity full or no customers left)
                # Return to Depot
                if vehicle['route_plan'][-1]['node_id'] != 0:
                     vehicle['route_plan'].append({'node_id': 0})
                vehicle['is_finished'] = True
                
    # Ensure all vehicles return to depot at the end
    for vehicle in vehicle_states:
        if vehicle['route_plan'][-1]['node_id'] != 0:
             vehicle['route_plan'].append({'node_id': 0})

    # For the "Construction Cost", we could run a deterministic sim, 
    # but here we just return the plan. The evaluation happens later.
    return {}, vehicle_states


def evaluate_greedy_stochastically(instance_path, results_dir):
    """
    Runs the Greedy Nearest Neighbor policy against N_SIMULATIONS stochastic days.
    """
    
    # 1. Initialize Simulator and Load Data
    instance_data = load_instance(instance_path)
    
    # CRITICAL FIX: DataFrame to Dict conversion
    if isinstance(instance_data['customers'], pd.DataFrame):
        instance_data['customers'] = instance_data['customers'].to_dict(orient='records')

    simulator = SVRPTW_Simulator(instance_data)
    
    # 2. Generate Policy (Routes) Once
    # The Greedy strategy generates a static set of routes based on initial state.
    _, static_vehicle_plans = run_greedy_policy(simulator)
    
    # Extract route plans: list of lists of Dicts [{'node_id': id}]
    greedy_routes_policy = [v['route_plan'] for v in static_vehicle_plans]
    
    # 3. Run Simulation N times
    all_day_results = []
    # print(f"  -> Running {N_SIMULATIONS} stochastic days...")

    for _ in range(N_SIMULATIONS):
        day_result = simulator.run_policy_for_day(greedy_routes_policy)
        all_day_results.append(day_result)
        
    # 4. Aggregate Results
    df_results = pd.DataFrame(all_day_results)
    
    agg_stats = {
        'policy_type': 'Greedy_NearestNeighbor_Evaluated',
        'N_simulations': N_SIMULATIONS,
        
        'mean_stochastic_cost': df_results['total_cost'].mean(),
        'std_stochastic_cost': df_results['total_cost'].std(),
        
        'mean_hard_late_penalties': df_results['hard_late_penalty_count'].mean(),
        'std_hard_late_penalties': df_results['hard_late_penalty_count'].std(),
        
        # Updated Metric Keys matching Simulator
        'mean_service_efficiency': df_results['total_service_efficiency'].mean(),
        'mean_transit_min': df_results['total_transit_min'].mean(),
        'mean_wait_min': df_results.get('total_wait_min', df_results.get('total_billable_wait_min', 0)).mean(),
        
        # Additional useful stats
        'mean_missed_customers': df_results.get('missed_customers', 0).mean() if 'missed_customers' in df_results else 0
    }
    
    # 5. Save Results
    base_filename = os.path.basename(instance_path).replace('.json', '')
    output_filename = f"{base_filename}_greedy_results.json"
    output_filepath = os.path.join(results_dir, output_filename)
    
    strategy_output = {
        'instance_file': os.path.basename(instance_path),
        'N': instance_data['num_customers'],
        'V': instance_data['num_vehicles'],
        'Q': instance_data['vehicle_capacity'],
        'routes': greedy_routes_policy, 
        'metrics': agg_stats
    }

    with open(output_filepath, 'w') as f:
        json.dump(strategy_output, f, indent=4)
    
    # 6. Generate Visualization (Optional, uses local helper or shared logic)
    visuals_dir = os.path.join(os.path.dirname(results_dir), 'visuals')
    os.makedirs(visuals_dir, exist_ok=True)
    
    viz_strategy_data = {
        'instance_file': os.path.basename(instance_path),
        'total_distance': df_results['total_distance_mi'].mean(),
        'routes': greedy_routes_policy
    }
    
    # We call the local helper _visualize_greedy_strategy
    _visualize_greedy_strategy(instance_data, viz_strategy_data, visuals_dir)


def _visualize_greedy_strategy(instance, strategy, output_dir):
    """Plots the planned routes (strategy) generated by the greedy heuristic."""
    
    routes = strategy.get('routes', [])
    num_vehicles = instance.get('num_vehicles', 0)
    
    plt.figure(figsize=(10, 10))
    plt.title(f"Greedy Strategy: {strategy['instance_file']} | Mean Dist: {strategy['total_distance']:.0f} mi", fontsize=14)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    
    colors = plt.cm.gist_rainbow(np.linspace(0, 1, num_vehicles))
    
    # Plot Depot
    depot_x, depot_y = instance['depot']['x'], instance['depot']['y']
    plt.scatter(depot_x, depot_y, marker='s', color='black', s=300, label='Depot', zorder=5)
    
    # Plot Customers
    # Handle DataFrame vs Dict difference if helper called internally
    if isinstance(instance['customers'], list):
        customer_coords = [(c['x'], c['y']) for c in instance['customers']]
        demands = [c['demand'] for c in instance['customers']]
    else:
        customer_coords = [(row['x'], row['y']) for _, row in instance['customers'].iterrows()]
        demands = [row['demand'] for _, row in instance['customers'].iterrows()]

    plt.scatter([c[0] for c in customer_coords], [c[1] for c in customer_coords], 
                c='gray', s=np.array(demands) * 3, alpha=0.6, label='Customers', zorder=3)

    # Plot Routes
    # Need map for coordinates
    cust_map = {c['id']: c for c in instance['customers']} if isinstance(instance['customers'], list) else \
               {row['id']: row.to_dict() for _, row in instance['customers'].iterrows()}
    cust_map[0] = instance['depot']

    for v, route in enumerate(routes):
        if not route: continue

        route_x = []
        route_y = []
        
        for step in route:
            node = cust_map[step['node_id']]
            route_x.append(node['x'])
            route_y.append(node['y'])
        
        plt.plot(route_x, route_y, color=colors[v % num_vehicles], linewidth=2, alpha=0.7)
        
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    
    output_filename = f"{strategy['instance_file'].replace('.json', '')}_greedy_strategy.png"
    plt.savefig(os.path.join(output_dir, output_filename))
    plt.close()


def run_batch_greedy_evaluation():
    """Finds all instance files and runs the stochastic evaluation for the Greedy policy."""
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    instance_data_dir = os.path.join(script_dir, 'instances', 'data')
    
    results_target_dir = os.path.join(script_dir, 'solutions', 'Greedy', 'simulation_results')
    visuals_target_dir = os.path.join(script_dir, 'solutions', 'Greedy', 'visuals')
    
    os.makedirs(results_target_dir, exist_ok=True)
    os.makedirs(visuals_target_dir, exist_ok=True)
    
    if not os.path.exists(instance_data_dir):
        print(f"Error: Directory not found: {instance_data_dir}")
        return

    instance_files = sorted(glob.glob(os.path.join(instance_data_dir, '*.json')))
    
    if not instance_files:
        print("Error: No instance JSON files found.")
        return

    print(f"\n--- Starting Greedy Policy Evaluation on {len(instance_files)} Instances ---")
    
    for i, filepath in enumerate(instance_files):
        if i % 10 == 0:
            print(f"Evaluating instance {i+1}/{len(instance_files)}: {os.path.basename(filepath)}")
            
        evaluate_greedy_stochastically(filepath, results_target_dir)

    print("\n--- Batch Greedy Evaluation Complete ---")
    print(f"Results saved to: {results_target_dir}")


if __name__ == '__main__':
    run_batch_greedy_evaluation()