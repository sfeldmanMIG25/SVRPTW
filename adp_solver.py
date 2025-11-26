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
RISK_AVERSION_FACTOR = 2.0 

# Training Settings
TRAIN_EPISODES = 3000 
TRAIN_BATCH_SIZE = 20 # Number of episodes to run in parallel per batch
EVAL_SIMULATIONS_PER_INSTANCE = 30

# Adaptive Learning Rate
LEARNING_RATE_CONSTANT = 200.0 

# --- PATH CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DATA_DIR = os.path.join(SCRIPT_DIR, 'instances', 'data')
ADP_STRATEGY_DIR = os.path.join(SCRIPT_DIR, 'solutions', 'ADP', 'strategy')
ADP_RESULTS_DIR = os.path.join(SCRIPT_DIR, 'solutions', 'ADP', 'simulation_results')

os.makedirs(ADP_STRATEGY_DIR, exist_ok=True)
os.makedirs(ADP_RESULTS_DIR, exist_ok=True)

# --- FEATURE NAMES MAPPING ---
FEATURE_NAMES = [
    "Bias (Intercept)",
    "Norm Demand (Unvisited)",
    "Norm Distance (Closest)",
    "Norm Time Left",
    "Norm Time Window Tightness",
    "Norm Fleet Availability",
    "Norm Fleet Capacity"
]

# --- COMPONENT 1: VALUE FUNCTION APPROXIMATION (VFA) ---
class LinearValueFunction:
    def __init__(self, num_features):
        self.weights = np.zeros(num_features)

    def estimate(self, features):
        return np.dot(self.weights, features)
    
    def update(self, prev_features, target_value, iteration):
        current_est = self.estimate(prev_features)
        error = target_value - current_est
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
                
            dist = euclidean_distance(curr_coord, (cust['x'], cust['y']))
            expected_arrival = current_time + dist 
            if expected_arrival <= cust['L']:
                actions.append(cid)
        return actions

    def calculate_transition(self, vehicle_idx, state, action_id, stochastic=False):
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
        wait_time = 0.0
        outcome = 'SUCCESS'
        
        next_unvisited = state['unvisited_ids'].copy()
        next_cap = v_state['capacity']
        
        if action_id == 0:
            service_start = arrival_time
        else:
            cust = instance['cust_map'][action_id]
            
            if arrival_time > cust['L']: 
                penalty = HARD_LATE_PENALTY
                next_unvisited.discard(action_id) 
                service_start = arrival_time
                outcome = 'LATE_SKIP'
                
            elif arrival_time < cust['E']: 
                wait_time = cust['E'] - arrival_time
                service_start = cust['E']
                if curr_loc_id == 0 and action_id == 0:
                    pass 
                else:
                    wage_billable_minutes += wait_time 
                
                service_time = StochasticSampler.sample_service_time(cust['mean_service_time']) if stochastic else cust['mean_service_time']
                wage_billable_minutes += service_time
                
                revenue = cust['demand'] * REVENUE_PER_UNIT_DEMAND
                next_cap -= cust['demand']
                next_unvisited.discard(action_id)
                
            else: 
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
        
        log_data = {
            'node_id': action_id,
            'outcome': outcome,
            'arrival_time': arrival_time,
            'service_start': service_start,
            'departure_time': finish_time,
            'wait_time': wait_time,
            'service_duration': service_time,
            'transit_cost': transit_cost,
            'wage_cost': wage_cost,
            'penalty_cost': penalty,
            'dist': dist
        }
        
        return contribution, next_state, finish_time, log_data

    def run_episode(self, train=True, collect_data=False):
        """
        Runs a simulation episode.
        If train=True and collect_data=True, returns a list of (features, target) tuples 
        instead of updating weights immediately. This enables parallel training batches.
        """
        instance = self.current_instance
        state = {
            'current_time': DEPOT_E_TIME,
            'unvisited_ids': set(c['id'] for c in instance['customers']),
            'vehicle_states': []
        }
        
        training_experiences = [] # For parallel data collection
        
        vehicle_traces = []
        for v in range(instance['num_vehicles']):
            state['vehicle_states'].append({'loc': 0, 'time_avail': DEPOT_E_TIME, 'capacity': instance['vehicle_capacity']})
            vehicle_traces.append([{
                'node_id': 0,
                'type': 'DEPOT_START',
                'arrival_time': DEPOT_E_TIME,
                'service_start': DEPOT_E_TIME,
                'departure_time': DEPOT_E_TIME,
                'wait_time': 0,
                'service_duration': 0
            }])

        events = [(DEPOT_E_TIME, v) for v in range(instance['num_vehicles'])]
        heapq.heapify(events)
        
        ep_stats = {
            'total_cost': 0.0, 
            'total_distance': 0.0,
            'hard_late_count': 0,
            'missed_customers': 0
        }
        
        while events:
            time_now, v_idx = heapq.heappop(events)
            if time_now > DEPOT_L_TIME: break
                
            state['current_time'] = time_now
            state['vehicle_states'][v_idx]['time_avail'] = time_now 
            
            feasible_actions = self.get_feasible_actions(v_idx, state)
            
            # --- DECISION LOGIC ---
            if train and np.random.rand() < EPSILON:
                action = np.random.choice(feasible_actions)
            else:
                best_val = -float('inf')
                best_action = 0 
                
                for a in feasible_actions:
                    contrib, post_state, f_time, _ = self.calculate_transition(v_idx, state, a, stochastic=False)
                    
                    risk_penalty = 0.0
                    if a != 0:
                        cust = instance['cust_map'][a]
                        slack = cust['L'] - f_time
                        if slack < 15: 
                            risk_penalty = RISK_AVERSION_FACTOR * (15 - slack)

                    feats = extract_features(post_state, v_idx, instance, self.global_statics)
                    future_val = self.vfa.estimate(feats)
                    val = contrib + (GAMMA * future_val) - risk_penalty
                    if val > best_val:
                        best_val = val
                        best_action = a
                action = best_action
            
            # --- EXECUTION ---
            real_contrib, next_state, real_finish_time, log = self.calculate_transition(v_idx, state, action, stochastic=True)
            
            step_cost = log['transit_cost'] + log['wage_cost'] + log['penalty_cost']
            ep_stats['total_cost'] += step_cost
            ep_stats['total_distance'] += log['dist']
            if log['outcome'] == 'LATE_SKIP': ep_stats['hard_late_count'] += 1

            vehicle_traces[v_idx].append(log)

            if train:
                _, post_state_chosen, _, _ = self.calculate_transition(v_idx, state, action, stochastic=False)
                feats_current = extract_features(post_state_chosen, v_idx, instance, self.global_statics)
                feats_next = extract_features(next_state, v_idx, instance, self.global_statics)
                target = real_contrib + (GAMMA * self.vfa.estimate(feats_next))
                
                if collect_data:
                    # Parallel mode: Store data for main process update
                    training_experiences.append((feats_current, target))
                else:
                    # Sequential mode: Update immediately
                    self.total_steps += 1
                    self.vfa.update(feats_current, target, self.total_steps)

            state = next_state
            
            if real_finish_time < DEPOT_L_TIME:
                if action != 0:
                    heapq.heappush(events, (real_finish_time, v_idx))
                elif action == 0 and state['unvisited_ids']:
                    next_try = time_now + 30
                    if next_try < DEPOT_L_TIME: heapq.heappush(events, (next_try, v_idx))

        if collect_data:
            return training_experiences
            
        ep_stats['missed_customers'] = len(state['unvisited_ids'])
        
        full_log = {
            'total_cost': ep_stats['total_cost'],
            'hard_lates': ep_stats['hard_late_count'],
            'missed_customers': ep_stats['missed_customers'],
            'vehicle_traces': vehicle_traces
        }
        
        return full_log

# --- WORKER FUNCTIONS ---

def load_instance_worker(filepath):
    """Helper to load instances in parallel."""
    try:
        data = load_instance(filepath)
        if isinstance(data['customers'], pd.DataFrame):
            data['customers'] = data['customers'].to_dict(orient='records')
        return data
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def train_batch_worker(instance, weights, global_statics):
    """
    Worker to run one training episode in parallel.
    Returns collected training experiences (features, targets).
    """
    try:
        # Reconstruct local agent
        agent = ADP_Engine(global_statics, weights)
        agent.set_instance(instance)
        # Run episode in data collection mode
        experiences = agent.run_episode(train=True, collect_data=True)
        return experiences
    except Exception as e:
        print(f"Training worker error: {e}")
        return []

def evaluate_single_instance_worker(filepath, weights, global_statics):
    try:
        data = load_instance(filepath)
        if isinstance(data['customers'], pd.DataFrame):
            data['customers'] = data['customers'].to_dict(orient='records')
            
        agent = ADP_Engine(global_statics, weights)
        agent.set_instance(data)
        
        daily_logs = []
        
        for i in range(EVAL_SIMULATIONS_PER_INSTANCE):
            log = agent.run_episode(train=False)
            log['day_index'] = i
            daily_logs.append(log)
            
        output_data = {
            'instance_file': os.path.basename(filepath),
            'policy_type': 'ADP_Master_Policy',
            'N': data['num_customers'],
            'V': data['num_vehicles'],
            'daily_simulation_logs': daily_logs
        }
        
        base_name = os.path.basename(filepath).replace('.json', '')
        output_file = os.path.join(ADP_RESULTS_DIR, f"{base_name}_adp_results.json")
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=4)
            
        avg_cost = np.mean([l['total_cost'] for l in daily_logs])
        return avg_cost
        
    except Exception as e:
        return f"Error: {str(e)}"

# --- MAIN PIPELINE ---
def run_adp_pipeline():
    instance_files = sorted([os.path.join(BASE_DATA_DIR, f) for f in os.listdir(BASE_DATA_DIR) if f.endswith('.json')])
    if not instance_files: return

    # 1. PARALLEL LOAD
    print(f"Loading {len(instance_files)} instances (Parallel)...")
    all_instances = []
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(load_instance_worker, f): f for f in instance_files}
        for future in as_completed(futures):
            res = future.result()
            if res: all_instances.append(res)
            
    # 2. Global Statics
    max_demand = max(sum(c['demand'] for c in inst['customers']) for inst in all_instances)
    max_vehicles = max(inst['num_vehicles'] for inst in all_instances)
    max_fleet_cap = max(inst['num_vehicles'] * inst['vehicle_capacity'] for inst in all_instances)
    max_dist = euclidean_distance((0,0), (max(COORDINATE_BOUNDS), max(COORDINATE_BOUNDS)))
    
    global_statics = {
        'max_global_demand': max_demand,
        'max_possible_dist': max_dist,
        'max_global_vehicles': max_vehicles,
        'max_global_fleet_capacity': max_fleet_cap,
        'time_horizon': DEPOT_L_TIME - DEPOT_E_TIME
    }
    
    # 3. PARALLEL TRAINING (Mini-Batch SGD)
    print(f"--- Training ADP Policy ({TRAIN_EPISODES} Episodes, Batch Size {TRAIN_BATCH_SIZE}) ---")
    start_train = time.time()
    agent = ADP_Engine(global_statics)
    
    total_episodes_completed = 0
    
    # We will submit batches of work
    with ProcessPoolExecutor() as executor:
        while total_episodes_completed < TRAIN_EPISODES:
            current_batch_size = min(TRAIN_BATCH_SIZE, TRAIN_EPISODES - total_episodes_completed)
            
            # Submit batch of episodes
            futures = []
            for _ in range(current_batch_size):
                inst = random.choice(all_instances)
                # Pass current weights copies
                futures.append(executor.submit(train_batch_worker, inst, agent.vfa.weights.copy(), global_statics))
            
            # Collect results and update weights
            batch_experiences = []
            for future in as_completed(futures):
                batch_experiences.extend(future.result())
                
            # Perform Updates in Main Process (Mini-Batch Update)
            # We preserve the iteration count logic for learning rate decay
            for feats, target in batch_experiences:
                agent.total_steps += 1
                agent.vfa.update(feats, target, agent.total_steps)
                
            total_episodes_completed += current_batch_size
            
            if total_episodes_completed % 100 == 0:
                print(f"  > Completed {total_episodes_completed} episodes...")
            
    print(f"Training Complete ({time.time() - start_train:.1f}s)")
    
    # 4. SAVE WEIGHTS
    weights_pickle = os.path.join(ADP_STRATEGY_DIR, 'adp_master_weights.pkl')
    with open(weights_pickle, 'wb') as f:
        pickle.dump({'weights': agent.vfa.weights, 'global_statics': global_statics}, f)
        
    weights_json = os.path.join(ADP_STRATEGY_DIR, 'adp_master_weights.json')
    readable_weights = {
        'feature_names': FEATURE_NAMES,
        'weights': agent.vfa.weights.tolist(),
        'global_statics': global_statics
    }
    with open(weights_json, 'w') as f:
        json.dump(readable_weights, f, indent=4)
        
    print(f"Weights saved to: {ADP_STRATEGY_DIR}")
    print("\n--- Final Learned Weights ---")
    for name, w in zip(FEATURE_NAMES, agent.vfa.weights):
        print(f"  {name:<30}: {w:.4f}")

    # 5. PARALLEL EVALUATION
    print(f"\n--- Evaluating ({EVAL_SIMULATIONS_PER_INSTANCE} runs/instance) ---")
    
    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(evaluate_single_instance_worker, f, agent.vfa.weights, global_statics): f 
            for f in instance_files
        }
        
        completed = 0
        costs = []
        for future in as_completed(futures):
            completed += 1
            res = future.result()
            if isinstance(res, (int, float)):
                costs.append(res)
                if completed % 10 == 0:
                    print(f"  [{completed}/{len(instance_files)}] Avg Cost: ${res:.0f}")
            else:
                print(f"  [{completed}/{len(instance_files)}] Failed: {res}")

    print("\n" + "="*40)
    if costs:
        print(f"ADP AVERAGE COST: ${np.mean(costs):,.0f}")
    print(f"Results saved to: {ADP_RESULTS_DIR}")
    print("="*40)

if __name__ == "__main__":
    run_adp_pipeline()