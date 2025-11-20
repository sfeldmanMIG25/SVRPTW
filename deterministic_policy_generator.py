import numpy as np
import pandas as pd
import json
import os
import time
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

from config import (
    WAGE_COST_PER_MINUTE, TRANSIT_COST_PER_MILE, HARD_LATE_PENALTY
)
from data_generator import euclidean_distance

# --- UTILITIES ---

def load_instance(filepath):
    """Loads a VRP instance from a JSON file, handling potential read errors."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        # Convert 'customers' list back into a DataFrame for easier handling
        data['customers'] = pd.DataFrame(data['customers'])
        return data
    except (json.JSONDecodeError, FileNotFoundError, Exception) as e:
        print(f"ERROR: Failed to load instance file {os.path.basename(filepath)}. Reason: {e}")
        return None

def create_data_model(instance):
    """Creates the data model required for the OR-Tools VRP solver."""
    customers_df = instance['customers']
    depot = instance['depot']
    
    # 1. Prepare Depot Data
    depot_data = depot.copy()
    depot_data['mean_service_time'] = 0.0 
    depot_df = pd.DataFrame([depot_data])
    
    # 2. Combine Depot and Customers
    all_nodes = pd.concat([depot_df, customers_df], ignore_index=True)
    all_nodes = all_nodes.fillna(0)

    # 3. Calculate Distance Matrix
    num_nodes = len(all_nodes)
    distance_matrix = np.zeros((num_nodes, num_nodes))
    
    coords = all_nodes[['x', 'y']].values
    
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                distance_matrix[i, j] = euclidean_distance(coords[i], coords[j])
    
    # 4. Extract Constraints
    time_windows = [(int(row['E']), int(row['L'])) for _, row in all_nodes.iterrows()]
    service_times = [int(row['mean_service_time']) for _, row in all_nodes.iterrows()] 
    demands = [int(row.get('demand', 0)) for _, row in all_nodes.iterrows()]

    data = {
        'distance_matrix': distance_matrix.round().astype(int).tolist(),
        'time_windows': time_windows,
        'service_times': service_times,
        'demands': demands,
        'num_vehicles': instance['num_vehicles'],
        'vehicle_capacities': [instance['vehicle_capacity']] * instance['num_vehicles'],
        'depot': 0
    }
    return data

def solve_vrptw_strategy(data, customers_df):
    """Solves the deterministic VRPTW using OR-Tools."""
    manager = pywrapcp.RoutingIndexManager(
        len(data['distance_matrix']), data['num_vehicles'], data['depot']
    )
    routing = pywrapcp.RoutingModel(manager)

    # 1. Distance/Transit Callback
    transit_callback_index = routing.RegisterTransitCallback(
        lambda from_index, to_index: data['distance_matrix'][manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]
    )
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # 2. Capacity Constraint
    demand_callback_index = routing.RegisterUnaryTransitCallback(
        lambda from_index: data['demands'][manager.IndexToNode(from_index)]
    )
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index, 0, data['vehicle_capacities'], True, 'Capacity'
    )

    # 3. Time Window Constraint
    def time_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index) 
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node] + data['service_times'][from_node]

    time_callback_index = routing.RegisterTransitCallback(time_callback)
    max_time = data['time_windows'][0][1] * 2 
    
    routing.AddDimension(
        time_callback_index, max_time, max_time, False, 'Time'
    )
    time_dimension = routing.GetDimensionOrDie('Time')
    
    for node_index in range(len(data['time_windows'])):
        index = manager.NodeToIndex(node_index)
        time_dimension.SetCumulVarRange(index, data['time_windows'][node_index][0], data['time_windows'][node_index][1])

    # 4. ADDED: Allow Dropping Nodes (Disjunctions)
    # This ensures a solution is ALWAYS found. If a node is impossible to visit, 
    # the solver drops it and incurs the HARD_LATE_PENALTY.
    penalty = int(HARD_LATE_PENALTY)
    for node_index in range(1, len(data['distance_matrix'])): # Skip depot (0)
        routing.AddDisjunction([manager.NodeToIndex(node_index)], penalty)

    # 5. Solve
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.time_limit.seconds = 30 
    search_parameters.log_search = False

    solution = routing.SolveWithParameters(search_parameters)
    
    if not solution:
        return {'routes': [], 'status': 'No solution found'}

    # 6. Extract Routes
    routes = []
    objective_value_distance = 0 
    
    for vehicle_id in range(data['num_vehicles']):
        route = []
        index = routing.Start(vehicle_id)
        
        while not routing.IsEnd(index):
            current_node = manager.IndexToNode(index)
            time_var = time_dimension.CumulVar(index)
            
            # Track distance manually since ObjectiveValue now includes penalties
            if not routing.IsEnd(solution.Value(routing.NextVar(index))):
                next_index = solution.Value(routing.NextVar(index))
                objective_value_distance += data['distance_matrix'][current_node][manager.IndexToNode(next_index)]

            route.append({
                'node_id': current_node,
                'arrival_time_det': solution.Min(time_var)
            })
            index = solution.Value(routing.NextVar(index))

        # Add final depot
        route.append({
            'node_id': data['depot'],
            'arrival_time_det': solution.Min(time_dimension.CumulVar(index))
        })
        routes.append(route)
    
    # Calculate deterministic cost 
    # Note: The solution.ObjectiveValue() now includes penalties, so we reconstruct the "Operational Cost"
    total_cost = (objective_value_distance * TRANSIT_COST_PER_MILE) + \
                 (customers_df['mean_service_time'].sum() * WAGE_COST_PER_MINUTE)

    return {
        'routes': routes,
        'total_distance': objective_value_distance,
        'total_cost': total_cost,
        'status': 'Optimal/Feasible'
    }

def process_single_instance(filepath, strategy_dir):
    """Worker function to process one file."""
    try:
        instance = load_instance(filepath)
        if instance is None: return None
            
        data_model = create_data_model(instance)
        
        start_time = time.time()
        strategy_data = solve_vrptw_strategy(data_model, instance['customers']) 
        solve_time = time.time() - start_time
        
        base_filename = os.path.basename(filepath).replace('.json', '')
        
        strategy_data.update({
            'instance_file': os.path.basename(filepath),
            'N': instance['num_customers'],
            'V': instance['num_vehicles'],
            'Q': instance['vehicle_capacity'],
            'solve_time_seconds': solve_time,
            'policy_type': 'Deterministic_ORTools_Strategy'
        })

        output_filename = f"{base_filename}_deterministic_strategy.json"
        output_filepath = os.path.join(strategy_dir, output_filename)
        
        with open(output_filepath, 'w') as f:
            json.dump(strategy_data, f, indent=4)
            
        return f"Generated: {output_filename} ({solve_time:.2f}s)"
        
    except Exception as e:
        return f"Error processing {filepath}: {str(e)}"

def run_batch_strategy_generation():
    """Main parallel execution function."""
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    instance_data_dir = os.path.join(script_dir, 'instances', 'data')
    strategy_dir = os.path.join(script_dir, 'solutions', 'ORTools', 'strategy')
    
    os.makedirs(strategy_dir, exist_ok=True)
    
    if not os.path.exists(instance_data_dir):
        print(f"Error: Directory not found: {instance_data_dir}")
        return

    instance_files = sorted(glob.glob(os.path.join(instance_data_dir, '*.json')))
    
    if not instance_files:
        print("Error: No instance JSON files found.")
        return

    print(f"\n--- Starting Parallel Policy Generation on {len(instance_files)} Instances ---")
    print(f"Using {os.cpu_count()} cores.")
    
    start_total = time.time()
    
    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(process_single_instance, f, strategy_dir): f 
            for f in instance_files
        }
        
        completed = 0
        for future in as_completed(futures):
            completed += 1
            result = future.result()
            if result:
                print(f"[{completed}/{len(instance_files)}] {result}")
    
    total_time = time.time() - start_total
    print(f"\n--- Generation Complete in {total_time:.2f}s ---")
    print(f"Strategies saved to: {strategy_dir}")

if __name__ == '__main__':
    run_batch_strategy_generation()