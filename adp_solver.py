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

# --- HYPERPARAMETERS (TUNED FOR AGGRESSION) ---
# Massive revenue to overpower operational costs
REVENUE_PER_UNIT_DEMAND = 50.0 
# Lower gamma to focus on "Get points NOW" rather than fearing the distant future
GAMMA = 0.85 
# High exploration to find the routes initially
EPSILON = 0.25 
# Low risk aversion - let it try to serve, even if tight
RISK_AVERSION_FACTOR = 0.1 

# Training Settings
TRAIN_EPISODES = 5000 
TRAIN_BATCH_SIZE = 20 
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

# --- FEATURE NAMES ---
FEATURE_NAMES = [
    "Bias",
    "Norm Demand",
    "Norm Dist (Me)",
    "Relative Dist (Me - Peer)",
    "Is Closest (Binary)", 
    "Sector Density",
    "Time Left",
    "Tightness",
    "Capacity"
]

# --- COMPONENT 1: VFA ---
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

# --- COMPONENT 2: COOPERATIVE FEATURES ---
def extract_features(state, vehicle_idx, instance, global_statics):
    current_v = state['vehicle_states'][vehicle_idx]
    current_loc_id = current_v['loc']
    
    if current_loc_id == 0:
        my_coord = (instance['depot']['x'], instance['depot']['y'])
    else:
        cust = instance['cust_map'][current_loc_id]
        my_coord = (cust['x'], cust['y'])

    # Peers
    peer_coords = []
    for idx, v in enumerate(state['vehicle_states']):
        if idx == vehicle_idx: continue
        if v['time_avail'] < DEPOT_L_TIME:
            lid = v['loc']
            if lid == 0: c = (instance['depot']['x'], instance['depot']['y'])
            else: c = (instance['cust_map'][lid]['x'], instance['cust_map'][lid]['y'])
            peer_coords.append(c)

    # Stats
    unvisited_demand = 0
    closest_dist = global_statics['max_possible_dist'] 
    sum_tightness = 0
    
    # Cooperation
    am_i_closest_count = 0
    avg_rel_dist = 0.0
    
    for cid in state['unvisited_ids']:
        c = instance['cust_map'][cid]
        unvisited_demand += c['demand']
        target = (c['x'], c['y'])
        
        d_me = euclidean_distance(my_coord, target)
        if d_me < closest_dist: closest_dist = d_me
        
        d_peer = global_statics['max_possible_dist']
        if peer_coords:
            d_peer = min([euclidean_distance(p, target) for p in peer_coords])
            
        avg_rel_dist += (d_me - d_peer)
        if d_me < d_peer: am_i_closest_count += 1
        sum_tightness += max(0, c['L'] - state['current_time'])

    num_unvisited = len(state['unvisited_ids'])
    if num_unvisited > 0:
        avg_rel_dist /= num_unvisited
    else:
        closest_dist = 0.0

    # Sector
    quad_x = int(my_coord[0] // 50)
    quad_y = int(my_coord[1] // 50)
    peers_in_sector = 0
    for p in peer_coords:
        if int(p[0]//50) == quad_x and int(p[1]//50) == quad_y:
            peers_in_sector += 1

    return np.array([
        1.0, 
        unvisited_demand / max(1, global_statics['max_global_demand']),
        closest_dist / global_statics['max_possible_dist'],
        avg_rel_dist / global_statics['max_possible_dist'], 
        am_i_closest_count / max(1, num_unvisited), 
        peers_in_sector / max(1, global_statics['max_global_vehicles']),
        (DEPOT_L_TIME - state['current_time']) / global_statics['time_horizon'],
        (sum_tightness / max(1, num_unvisited)) / global_statics['time_horizon'],
        sum(v['capacity'] for v in state['vehicle_states']) / max(1, global_statics['max_global_fleet_capacity'])
    ])

# --- COMPONENT 3: ADP ENGINE ---
class ADP_Engine:
    def __init__(self, global_statics, weights=None):
        self.global_statics = global_statics
        self.vfa = LinearValueFunction(num_features=9)
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
            if current_time + dist <= cust['L']:
                actions.append(cid)
        return actions

    def calculate_transition(self, vehicle_idx, state, action_id, stochastic=False):
        instance = self.current_instance
        v_state = state['vehicle_states'][vehicle_idx]
        curr_loc = instance['cust_coords'][v_state['loc']]
        target_loc = instance['cust_coords'][action_id]
        
        dist = euclidean_distance(curr_loc, target_loc)
        travel_time = StochasticSampler.sample_travel_time(dist) if stochastic else dist
        arrival_time = v_state['time_avail'] + travel_time
        
        transit_cost = dist * TRANSIT_COST_PER_MILE 
        wage_billable = travel_time 
        
        revenue = 0.0; penalty = 0.0; service_time = 0.0; wait_time = 0.0
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
                if v_state['loc'] != 0 or action_id != 0: wage_billable += wait_time 
                service_time = StochasticSampler.sample_service_time(cust['mean_service_time']) if stochastic else cust['mean_service_time']
                wage_billable += service_time
                revenue = cust['demand'] * REVENUE_PER_UNIT_DEMAND
                next_cap -= cust['demand']
                next_unvisited.discard(action_id)
            else: 
                service_start = arrival_time
                service_time = StochasticSampler.sample_service_time(cust['mean_service_time']) if stochastic else cust['mean_service_time']
                wage_billable += service_time
                revenue = cust['demand'] * REVENUE_PER_UNIT_DEMAND
                next_cap -= cust['demand']
                next_unvisited.discard(action_id)

        wage_cost = wage_billable * WAGE_COST_PER_MINUTE
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
            'node_id': action_id, 'outcome': outcome, 'arrival_time': arrival_time,
            'service_start': service_start, 'departure_time': finish_time,
            'wait_time': wait_time, 'service_duration': service_time,
            'transit_cost': transit_cost, 'wage_cost': wage_cost, 'penalty_cost': penalty, 'dist': dist
        }
        return contribution, next_state, finish_time, log_data

    def run_episode(self, train=True, collect_data=False):
        instance = self.current_instance
        state = {
            'current_time': DEPOT_E_TIME,
            'unvisited_ids': set(c['id'] for c in instance['customers']),
            'vehicle_states': [{'loc': 0, 'time_avail': DEPOT_E_TIME, 'capacity': instance['vehicle_capacity']} for _ in range(instance['num_vehicles'])]
        }
        
        training_experiences = []
        vehicle_traces = []
        for v in range(instance['num_vehicles']):
            vehicle_traces.append([{
                'node_id': 0, 'outcome': 'DEPOT_START', 'arrival_time': DEPOT_E_TIME, 
                'service_start': DEPOT_E_TIME, 'departure_time': DEPOT_E_TIME, 'wait_time': 0, 
                'service_duration': 0, 'transit_cost': 0, 'wage_cost': 0, 'penalty_cost': 0, 'dist': 0
            }])

        events = [(DEPOT_E_TIME, v) for v in range(instance['num_vehicles'])]
        heapq.heapify(events)
        
        ep_stats = {'total_cost': 0.0, 'hard_lates': 0, 'missed': 0}
        
        while events:
            time_now, v_idx = heapq.heappop(events)
            if time_now > DEPOT_L_TIME: break
            state['current_time'] = time_now
            state['vehicle_states'][v_idx]['time_avail'] = time_now 
            
            feasible = self.get_feasible_actions(v_idx, state)
            
            if train and np.random.rand() < EPSILON:
                action = np.random.choice(feasible)
            else:
                best_val = -float('inf')
                best_action = 0 
                for a in feasible:
                    contrib, post, f_time, _ = self.calculate_transition(v_idx, state, a, stochastic=False)
                    risk = 0.0
                    if a != 0:
                        cust = instance['cust_map'][a]
                        slack = cust['L'] - f_time
                        if slack < 15: risk = RISK_AVERSION_FACTOR * (15 - slack)

                    feats = extract_features(post, v_idx, instance, self.global_statics)
                    future = self.vfa.estimate(feats)
                    val = contrib + (GAMMA * future) - risk
                    if val > best_val:
                        best_val = val; best_action = a
                action = best_action
            
            contrib, next_state, finish, log = self.calculate_transition(v_idx, state, action, stochastic=True)
            
            ep_stats['total_cost'] += (log['transit_cost'] + log['wage_cost'] + log['penalty_cost'])
            if log['outcome'] == 'LATE_SKIP': ep_stats['hard_lates'] += 1
            vehicle_traces[v_idx].append(log)

            if train:
                _, post_det, _, _ = self.calculate_transition(v_idx, state, action, stochastic=False)
                feats_curr = extract_features(post_det, v_idx, instance, self.global_statics)
                feats_next = extract_features(next_state, v_idx, instance, self.global_statics)
                target = contrib + (GAMMA * self.vfa.estimate(feats_next))
                if collect_data: training_experiences.append((feats_curr, target))
                else:
                    self.total_steps += 1
                    self.vfa.update(feats_curr, target, self.total_steps)

            state = next_state
            if finish < DEPOT_L_TIME:
                if action != 0: heapq.heappush(events, (finish, v_idx))
                elif action == 0 and state['unvisited_ids']:
                    nxt = time_now + 30
                    if nxt < DEPOT_L_TIME: heapq.heappush(events, (nxt, v_idx))

        ep_stats['missed'] = len(state['unvisited_ids'])
        
        # TERMINAL UPDATE: Weakened to prevent "Fear of God" effect
        if train:
            # We only penalize if we missed things, but not 1000 per customer immediate
            # We let the cumulative negative reward from missing "revenue" do the work
            pass 

        if collect_data: return training_experiences
            
        return {
            'total_cost': ep_stats['total_cost'],
            'hard_lates': ep_stats['hard_lates'],
            'missed_customers': ep_stats['missed'],
            'vehicle_traces': vehicle_traces
        }

# --- WORKERS & PIPELINE ---
def load_instance_worker(filepath):
    try:
        data = load_instance(filepath)
        if isinstance(data['customers'], pd.DataFrame): data['customers'] = data['customers'].to_dict(orient='records')
        return data
    except Exception: return None

def train_batch_worker(instance, weights, global_statics):
    try:
        agent = ADP_Engine(global_statics, weights)
        agent.set_instance(instance)
        return agent.run_episode(train=True, collect_data=True)
    except Exception: return []

def evaluate_single_instance_worker(filepath, weights, global_statics):
    try:
        data = load_instance(filepath)
        if isinstance(data['customers'], pd.DataFrame): data['customers'] = data['customers'].to_dict(orient='records')
        agent = ADP_Engine(global_statics, weights)
        agent.set_instance(data)
        daily_logs = []
        for i in range(EVAL_SIMULATIONS_PER_INSTANCE):
            log = agent.run_episode(train=False)
            log['day_index'] = i
            daily_logs.append(log)
        output_data = {
            'instance_file': os.path.basename(filepath), 'policy_type': 'ADP_Master_Policy',
            'N': data['num_customers'], 'V': data['num_vehicles'], 'daily_simulation_logs': daily_logs
        }
        base_name = os.path.basename(filepath).replace('.json', '')
        output_file = os.path.join(ADP_RESULTS_DIR, f"{base_name}_adp_results.json")
        with open(output_file, 'w') as f: json.dump(output_data, f, indent=4)
        return np.mean([l['total_cost'] for l in daily_logs])
    except Exception as e: return f"Error: {e}"

def run_adp_pipeline():
    instance_files = sorted([os.path.join(BASE_DATA_DIR, f) for f in os.listdir(BASE_DATA_DIR) if f.endswith('.json')])
    if not instance_files: return

    print(f"Loading {len(instance_files)} instances...")
    all_instances = []
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(load_instance_worker, f): f for f in instance_files}
        for future in as_completed(futures):
            res = future.result()
            if res: all_instances.append(res)
            
    max_demand = max(sum(c['demand'] for c in inst['customers']) for inst in all_instances)
    max_vehicles = max(inst['num_vehicles'] for inst in all_instances)
    max_fleet_cap = max(inst['num_vehicles'] * inst['vehicle_capacity'] for inst in all_instances)
    max_dist = euclidean_distance((0,0), (max(COORDINATE_BOUNDS), max(COORDINATE_BOUNDS)))
    
    global_statics = {
        'max_global_demand': max_demand, 'max_possible_dist': max_dist,
        'max_global_vehicles': max_vehicles, 'max_global_fleet_capacity': max_fleet_cap,
        'time_horizon': DEPOT_L_TIME - DEPOT_E_TIME
    }
    
    print(f"--- Training ADP (Aggressive Cooperative) | {TRAIN_EPISODES} Episodes ---")
    start_train = time.time()
    agent = ADP_Engine(global_statics)
    
    completed = 0
    with ProcessPoolExecutor() as executor:
        while completed < TRAIN_EPISODES:
            batch = min(TRAIN_BATCH_SIZE, TRAIN_EPISODES - completed)
            futures = []
            for _ in range(batch):
                inst = random.choice(all_instances)
                futures.append(executor.submit(train_batch_worker, inst, agent.vfa.weights.copy(), global_statics))
            
            for future in as_completed(futures):
                exps = future.result()
                for feats, target in exps:
                    agent.total_steps += 1
                    agent.vfa.update(feats, target, agent.total_steps)
            
            completed += batch
            if completed % 500 == 0: print(f"  > {completed} Episodes")
            
    print(f"Training Complete ({time.time() - start_train:.1f}s)")
    
    with open(os.path.join(ADP_STRATEGY_DIR, 'adp_master_weights.pkl'), 'wb') as f:
        pickle.dump({'weights': agent.vfa.weights, 'global_statics': global_statics}, f)
        
    print("\n--- Learned Weights ---")
    for n, w in zip(FEATURE_NAMES, agent.vfa.weights):
        print(f"  {n:<40}: {w:.4f}")

    print(f"\n--- Evaluating... ---")
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(evaluate_single_instance_worker, f, agent.vfa.weights, global_statics): f for f in instance_files}
        for future in as_completed(futures): future.result()

    print("Pipeline Complete.")

if __name__ == "__main__":
    run_adp_pipeline()