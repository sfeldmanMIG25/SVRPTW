import numpy as np
import pandas as pd
import os
import json
import copy
import heapq
import random
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import namedtuple

# Import existing environment tools
from config import (
    WAGE_COST_PER_MINUTE, TRANSIT_COST_PER_MILE,
    DEPOT_E_TIME, DEPOT_L_TIME, HARD_LATE_PENALTY,
    COORDINATE_BOUNDS
)
from simulator import StochasticSampler
from deterministic_policy_generator import load_instance
from data_generator import euclidean_distance

# --- HYPERPARAMETERS ---
REVENUE_PER_UNIT_DEMAND = 5.0
GAMMA = 0.9
EPSILON = 0.2

# Training Settings
TRAIN_EPISODES = 3000 
EVAL_SIMULATIONS_PER_INSTANCE = 50 # Number of stochastic runs per instance during evaluation

# Adaptive Learning Rate Constants
LEARNING_RATE_CONSTANT = 200.0 

# --- PATH CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DATA_DIR = os.path.join(SCRIPT_DIR, 'instances', 'data')
ADP_STRATEGY_DIR = os.path.join(SCRIPT_DIR, 'instances', 'solutions', 'ADP', 'strategy')
ADP_RESULTS_DIR = os.path.join(SCRIPT_DIR, 'instances', 'solutions', 'ADP', 'simulation_results')

# Ensure directories exist
os.makedirs(ADP_STRATEGY_DIR, exist_ok=True)
os.makedirs(ADP_RESULTS_DIR, exist_ok=True)


# --- COMPONENT 1: VALUE FUNCTION APPROXIMATION (VFA) ---
class LinearValueFunction:
    def __init__(self, num_features):
        self.weights = np.zeros(num_features)

    def estimate(self, features):
        return np.dot(self.weights, features)
    
    def update(self, prev_features, target_value, iteration):
        """SGD update with Adaptive Learning Rate."""
        current_est = self.estimate(prev_features)
        error = target_value - current_est
        
        # Search-then-Converge
        alpha = LEARNING_RATE_CONSTANT / (LEARNING_RATE_CONSTANT + iteration)
        self.weights += alpha * error * prev_features

# --- COMPONENT 2: FEATURE ENGINEERING ---
def extract_features(state, vehicle_idx, instance, global_statics):
    """Extracts features scaled by GLOBAL maximums."""
    current_v = state['vehicle_states'][vehicle_idx]
    current_loc_id = current_v['loc']
    
    if current_loc_id == 0:
        curr_coord = (instance['depot']['x'], instance['depot']['y'])
    else:
        # Assumes 'cust_map' is injected into instance
        cust = instance['cust_map'][current_loc_id]
        curr_coord = (cust['x'], cust['y'])

    f_bias = 1.0
    
    unvisited_demand = 0
    closest_dist = global_statics['max_possible_dist'] 
    sum_tightness = 0
    
    for cid in state['unvisited_ids']:
        c = instance['cust_map'][cid]
        unvisited_demand += c['demand']
        
        d = euclidean_distance(curr_coord, (c['x'], c['y']))
        if d < closest_dist:
            closest_dist = d
            
        tightness = max(0, c['L'] - state['current_time'])
        sum_tightness += tightness

    if not state['unvisited_ids']:
        closest_dist = 0.0

    available_vehicles = sum(1 for v in state['vehicle_states'] 
                             if v['time_avail'] <= state['current_time'])
    
    total_current_capacity = sum(v['capacity'] for v in state['vehicle_states'])

    # Normalization
    f_demand = unvisited_demand / max(1, global_statics['max_global_demand'])
    f_dist = closest_dist / global_statics['max_possible_dist']
    
    time_left = DEPOT_L_TIME - state['current_time']
    f_time_left = time_left / global_statics['time_horizon']
    
    num_unvisited = len(state['unvisited_ids'])
    if num_unvisited > 0:
        avg_tightness = sum_tightness / num_unvisited
        f_tightness = avg_tightness / global_statics['time_horizon']
    else:
        f_tightness = 0.0

    f_fleet_avail = available_vehicles / max(1, global_statics['max_global_vehicles'])
    f_fleet_cap = total_current_capacity / max(1, global_statics['max_global_fleet_capacity'])

    return np.array([
        f_bias, f_demand, f_dist, f_time_left, f_tightness, f_fleet_avail, f_fleet_cap
    ])

# --- COMPONENT 3: ADP ENGINE ---
class ADP_Engine:
    def __init__(self, global_statics, weights=None):
        self.global_statics = global_statics
        self.vfa = LinearValueFunction(num_features=7)
        if weights is not None:
            self.vfa.weights = np.array(weights)
            
        self.total_steps = 0
        self.current_instance = None 

    def set_instance(self, instance_data):
        self.current_instance = instance_data
        # Inject optimized lookups
        if 'cust_map' not in self.current_instance:
            self.current_instance['cust_map'] = {c['id']: c for c in self.current_instance['customers']}
            self.current_instance['cust_map'][0] = self.current_instance['depot']
            self.current_instance['cust_coords'] = {c['id']: (c['x'], c['y']) for c in self.current_instance['customers']}
            self.current_instance['cust_coords'][0] = (self.current_instance['depot']['x'], self.current_instance['depot']['y'])

    def get_feasible_actions(self, vehicle_idx, state):
        v_state = state['vehicle_states'][vehicle_idx]
        actions = [0] 
        
        curr_loc_id = v_state['loc']
        curr_coord = self.current_instance['cust_coords'][curr_loc_id]
        current_time = v_state['time_avail']
        
        for cid in state['unvisited_ids']:
            cust = self.current_instance['cust_map'][cid]
            if cust['demand'] > v_state['capacity']: continue
                
            # Deterministic feasibility check (Masking)
            dist = euclidean_distance(curr_coord, (cust['x'], cust['y']))
            expected_arrival = current_time + dist 
            if expected_arrival <= cust['L']:
                actions.append(cid)
        return actions

    def calculate_contribution_and_transition(self, vehicle_idx, state, action_id, stochastic=False):
        instance = self.current_instance
        v_state = state['vehicle_states'][vehicle_idx]
        curr_loc_id = v_state['loc']
        curr_coord = instance['cust_coords'][curr_loc_id]
        
        target_coord = instance['cust_coords'][action_id]
        dist = euclidean_distance(curr_coord, target_coord)
        
        travel_time = StochasticSampler.sample_travel_time(dist) if stochastic else dist
        arrival_time = v_state['time_avail'] + travel_time
        
        transit_cost = dist * TRANSIT_COST_PER_MILE 
        wage_billable_minutes = travel_time 
        
        revenue = 0.0
        penalty = 0.0
        service_time = 0.0
        
        next_unvisited = state['unvisited_ids'].copy()
        next_cap = v_state['capacity']
        
        if action_id == 0:
            service_start = arrival_time
        else:
            cust = instance['cust_map'][action_id]
            
            if arrival_time > cust['L']: # Late
                penalty = HARD_LATE_PENALTY
                next_unvisited.discard(action_id) 
                service_start = arrival_time
            elif arrival_time < cust['E']: # Early
                wait_time = cust['E'] - arrival_time
                wage_billable_minutes += wait_time 
                service_start = cust['E']
                service_time = StochasticSampler.sample_service_time(cust['mean_service_time']) if stochastic else cust['mean_service_time']
                wage_billable_minutes += service_time
                revenue = cust['demand'] * REVENUE_PER_UNIT_DEMAND
                next_cap -= cust['demand']
                next_unvisited.discard(action_id)
            else: # On Time
                service_start = arrival_time
                service_time = StochasticSampler.sample_service_time(cust['mean_service_time']) if stochastic else cust['mean_service_time']
                wage_billable_minutes += service_time
                revenue = cust['demand'] * REVENUE_PER_UNIT_DEMAND
                next_cap -= cust['demand']
                next_unvisited.discard(action_id)

        wage_cost = wage_billable_minutes * WAGE_COST_PER_MINUTE
        contribution = revenue - (transit_cost + wage_cost + penalty)
        finish_time = service_start + service_time
        
        next_v_state = {'loc': action_id, 'time_avail': finish_time, 'capacity': next_cap}
        next_state = {
            'current_time': state['current_time'], 
            'unvisited_ids': next_unvisited,
            'vehicle_states': copy.deepcopy(state['vehicle_states'])
        }
        next_state['vehicle_states'][vehicle_idx] = next_v_state
        
        return contribution, next_state, finish_time, (transit_cost, wage_cost, penalty, service_time, travel_time)

    def run_episode(self, train=True):
        instance = self.current_instance
        state = {
            'current_time': DEPOT_E_TIME,
            'unvisited_ids': set(c['id'] for c in instance['customers']),
            'vehicle_states': []
        }
        
        events = []
        for v in range(instance['num_vehicles']):
            state['vehicle_states'].append({'loc': 0, 'time_avail': DEPOT_E_TIME, 'capacity': instance['vehicle_capacity']})
            heapq.heappush(events, (DEPOT_E_TIME, v))
            
        # Episode Metrics
        ep_stats = {
            'total_cost': 0.0, 'total_transit_cost': 0.0, 'total_wage_cost': 0.0, 
            'total_penalty': 0.0, 'missed_customers': 0, 
            'total_service_time': 0.0, 'total_transit_time': 0.0,
            'hard_late_count': 0
        }
        
        while events:
            time_now, v_idx = heapq.heappop(events)
            if time_now > DEPOT_L_TIME: break
                
            state['current_time'] = time_now
            state['vehicle_states'][v_idx]['time_avail'] = time_now 
            
            feasible_actions = self.get_feasible_actions(v_idx, state)
            
            # Policy Selection
            if train and np.random.rand() < EPSILON:
                action = np.random.choice(feasible_actions)
            else:
                best_val = -float('inf')
                best_action = 0 
                for a in feasible_actions:
                    contrib, post_state, _, _ = self.calculate_contribution_and_transition(v_idx, state, a, stochastic=False)
                    feats = extract_features(post_state, v_idx, instance, self.global_statics)
                    future_val = self.vfa.estimate(feats)
                    if contrib + (GAMMA * future_val) > best_val:
                        best_val = contrib + (GAMMA * future_val)
                        best_action = a
                action = best_action
            
            # Execution
            real_contrib, next_state, real_finish_time, costs = self.calculate_contribution_and_transition(v_idx, state, action, stochastic=True)
            t_cost, w_cost, pen, s_time, tr_time = costs
            
            # Track Metrics
            ep_stats['total_transit_cost'] += t_cost
            ep_stats['total_wage_cost'] += w_cost
            ep_stats['total_penalty'] += pen
            ep_stats['total_cost'] += (t_cost + w_cost + pen)
            ep_stats['total_service_time'] += s_time
            ep_stats['total_transit_time'] += tr_time
            if pen > 0: ep_stats['hard_late_count'] += 1

            # Training Update
            if train:
                _, post_state_chosen, _, _ = self.calculate_contribution_and_transition(v_idx, state, action, stochastic=False)
                feats_current = extract_features(post_state_chosen, v_idx, instance, self.global_statics)
                feats_next = extract_features(next_state, v_idx, instance, self.global_statics)
                target = real_contrib + (GAMMA * self.vfa.estimate(feats_next))
                self.total_steps += 1
                self.vfa.update(feats_current, target, self.total_steps)

            state = next_state
            
            if real_finish_time < DEPOT_L_TIME:
                if action != 0:
                    heapq.heappush(events, (real_finish_time, v_idx))
                elif action == 0 and state['unvisited_ids']:
                    # Retry heuristic
                    next_try = time_now + 30
                    if next_try < DEPOT_L_TIME: heapq.heappush(events, (next_try, v_idx))

        ep_stats['missed_customers'] = len(state['unvisited_ids'])
        return ep_stats

# --- WORKER FUNCTION FOR PARALLEL EVALUATION ---
def evaluate_single_instance_worker(filepath, weights, global_statics):
    """
    Worker that runs the ADP policy on a single instance for N simulations.
    """
    try:
        data = load_instance(filepath)
        if isinstance(data['customers'], pd.DataFrame):
            data['customers'] = data['customers'].to_dict(orient='records')
            
        # Create local engine with trained weights
        agent = ADP_Engine(global_statics, weights)
        agent.set_instance(data)
        
        sim_results = []
        for _ in range(EVAL_SIMULATIONS_PER_INSTANCE):
            sim_results.append(agent.run_episode(train=False))
            
        # Aggregate
        df = pd.DataFrame(sim_results)
        metrics = {
            'mean_total_cost': df['total_cost'].mean(),
            'std_total_cost': df['total_cost'].std(),
            'mean_missed_customers': df['missed_customers'].mean(),
            'mean_hard_late_penalties': df['hard_late_count'].mean(),
            'mean_service_time': df['total_service_time'].mean(),
            'fleet_utilization': df['total_service_time'].mean() / max(1, (data['num_vehicles'] * global_statics['time_horizon']))
        }
        
        # Save Result
        base_name = os.path.basename(filepath).replace('.json', '')
        output_file = os.path.join(ADP_RESULTS_DIR, f"{base_name}_adp_results.json")
        
        output_data = {
            'instance_file': os.path.basename(filepath),
            'policy_type': 'ADP_Master_Policy',
            'metrics': metrics
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=4)
            
        return (metrics['mean_total_cost'], metrics['mean_missed_customers'])
        
    except Exception as e:
        return f"Error: {str(e)}"

# --- MAIN PIPELINE ---
def run_adp_pipeline():
    # 1. Load ALL Instances for Training Data
    instance_files = sorted([os.path.join(BASE_DATA_DIR, f) for f in os.listdir(BASE_DATA_DIR) if f.endswith('.json')])
    
    if not instance_files:
        print(f"No instances found in {BASE_DATA_DIR}")
        return

    print(f"Loading {len(instance_files)} instances for global scaling...")
    all_instances = []
    for f in instance_files:
        data = load_instance(f)
        if isinstance(data['customers'], pd.DataFrame):
            data['customers'] = data['customers'].to_dict(orient='records')
        all_instances.append(data)
        
    # 2. Calculate Global Statics
    max_demand = max(sum(c['demand'] for c in inst['customers']) for inst in all_instances)
    max_vehicles = max(inst['num_vehicles'] for inst in all_instances)
    max_fleet_cap = max(inst['num_vehicles'] * inst['vehicle_capacity'] for inst in all_instances)
    max_coord = max(COORDINATE_BOUNDS)
    max_dist = euclidean_distance((0,0), (max_coord, max_coord))
    
    global_statics = {
        'max_global_demand': max_demand,
        'max_possible_dist': max_dist,
        'max_global_vehicles': max_vehicles,
        'max_global_fleet_capacity': max_fleet_cap,
        'time_horizon': DEPOT_L_TIME - DEPOT_E_TIME
    }
    
    # 3. TRAIN
    print(f"--- Starting Training ({TRAIN_EPISODES} Episodes) ---")
    start_train = time.time()
    agent = ADP_Engine(global_statics)
    
    for e in range(TRAIN_EPISODES):
        agent.set_instance(random.choice(all_instances))
        agent.run_episode(train=True)
        if (e+1) % 500 == 0:
            print(f"  > Trained {e+1} episodes...")
            
    print(f"Training Complete ({time.time() - start_train:.1f}s)")
    
    # 4. SAVE WEIGHTS
    weights_file = os.path.join(ADP_STRATEGY_DIR, 'adp_master_weights.pkl')
    with open(weights_file, 'wb') as f:
        save_data = {'weights': agent.vfa.weights, 'global_statics': global_statics}
        pickle.dump(save_data, f)
    print(f"Weights saved to: {weights_file}")
    print(f"Final Weights: {agent.vfa.weights}")

    # 5. PARALLEL EVALUATION
    print(f"\n--- Starting Parallel Evaluation on {len(instance_files)} Instances ---")
    print(f"Simulations per instance: {EVAL_SIMULATIONS_PER_INSTANCE}")
    
    eval_costs = []
    eval_missed = []
    
    with ProcessPoolExecutor() as executor:
        # Pass weights explicitly to workers
        futures = {
            executor.submit(evaluate_single_instance_worker, f, agent.vfa.weights, global_statics): f 
            for f in instance_files
        }
        
        completed = 0
        for future in as_completed(futures):
            completed += 1
            res = future.result()
            
            if isinstance(res, tuple):
                eval_costs.append(res[0])
                eval_missed.append(res[1])
                if completed % 10 == 0:
                    print(f"  [{completed}/{len(instance_files)}] Processed. Avg Cost: {res[0]:.0f}")
            else:
                print(f"  [{completed}/{len(instance_files)}] Failed: {res}")

    # 6. Final Summary
    print("\n" + "="*40)
    print("ADP MASTER POLICY EVALUATION RESULTS")
    print("="*40)
    print(f"Average Cost across Dataset: ${np.mean(eval_costs):,.2f}")
    print(f"Average Missed Customers: {np.mean(eval_missed):.2f}")
    print(f"Results saved to: {ADP_RESULTS_DIR}")
    print("="*40)

if __name__ == "__main__":
    run_adp_pipeline()