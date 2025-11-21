import numpy as np
import pandas as pd
import os
import json
import copy
import heapq
import math
import time
import random
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed

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
MCTS_ITERATIONS = 100    
EXPLORATION_CONSTANT = 2.0 
ROLLOUT_DEPTH = 10      
REVENUE_PER_UNIT = 10.0
MAX_CHILDREN = 8        
EXPANSION_SAMPLES = 3   

# Evaluation
EVAL_SIMULATIONS = 100 # Increased to 100 for robustness

# --- PATHS ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DATA_DIR = os.path.join(SCRIPT_DIR, 'instances', 'data')
GENETIC_WEIGHTS_FILE = os.path.join(SCRIPT_DIR, 'solutions', 'Genetic', 'genetic_policy_weights.pkl')
HYBRID_SOLUTIONS_DIR = os.path.join(SCRIPT_DIR, 'solutions', 'MCTS_Hybrid')
HYBRID_RESULTS_DIR = os.path.join(HYBRID_SOLUTIONS_DIR, 'simulation_results')

os.makedirs(HYBRID_SOLUTIONS_DIR, exist_ok=True)
os.makedirs(HYBRID_RESULTS_DIR, exist_ok=True)

# --- HELPER: FEATURE EXTRACTION ---
def extract_features(state, vehicle_idx, instance, global_statics):
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
        if d < closest_dist: closest_dist = d
        sum_tightness += max(0, c['L'] - state['current_time'])

    if not state['unvisited_ids']: closest_dist = 0.0

    available_vehicles = sum(1 for v in state['vehicle_states'] if v['time_avail'] <= state['current_time'])
    total_current_capacity = sum(v['capacity'] for v in state['vehicle_states'])

    f_demand = unvisited_demand / max(1, global_statics['max_global_demand'])
    f_dist = closest_dist / global_statics['max_possible_dist']
    f_time_left = (DEPOT_L_TIME - state['current_time']) / global_statics['time_horizon']
    
    num_unvisited = len(state['unvisited_ids'])
    f_tightness = (sum_tightness / num_unvisited) / global_statics['time_horizon'] if num_unvisited > 0 else 0.0

    f_fleet_avail = available_vehicles / max(1, global_statics['max_global_vehicles'])
    f_fleet_cap = total_current_capacity / max(1, global_statics['max_global_fleet_capacity'])

    return np.array([f_bias, f_demand, f_dist, f_time_left, f_tightness, f_fleet_avail, f_fleet_cap])

# --- MCTS NODE CLASS ---
class MCTSNode:
    def __init__(self, state, parent=None, action_from_parent=None):
        self.state = state
        self.parent = parent
        self.action_from_parent = action_from_parent
        self.children = []
        self.visits = 0
        self.value = 0.0 
        self.untried_actions = None 

    def is_fully_expanded(self):
        return self.untried_actions is not None and len(self.untried_actions) == 0

    def best_child(self, c_param=EXPLORATION_CONSTANT):
        choices_weights = [
            (child.value / child.visits) + c_param * math.sqrt((2 * math.log(self.visits) / child.visits))
            for child in self.children
        ]
        return self.children[np.argmax(choices_weights)]

# --- HYBRID ENGINE ---
class MCTSEngine:
    def __init__(self, instance, adp_weights, global_statics):
        self.instance = instance
        self.adp_weights = adp_weights
        self.global_statics = global_statics
        
        self.cust_map = self.instance.get('cust_map')
        self.cust_coords = self.instance.get('cust_coords')

    def get_adp_value(self, state, vehicle_idx):
        if self.adp_weights is None: return 0.0
        feats = extract_features(state, vehicle_idx, self.instance, self.global_statics)
        return np.dot(self.adp_weights, feats)

    def get_feasible_actions(self, vehicle_idx, state, prune=True):
        v_state = state['vehicle_states'][vehicle_idx]
        curr_loc = self.cust_coords[v_state['loc']]
        curr_time = v_state['time_avail']
        
        candidates = []
        for cid in state['unvisited_ids']:
            cust = self.cust_map[cid]
            if cust['demand'] > v_state['capacity']: continue
            
            dist = euclidean_distance(curr_loc, (cust['x'], cust['y']))
            if curr_time + dist <= cust['L']:
                candidates.append((dist, cid))
        
        if prune and len(candidates) > MAX_CHILDREN:
            candidates.sort(key=lambda x: x[0]) 
            actions = [c[1] for c in candidates[:MAX_CHILDREN]]
        else:
            actions = [c[1] for c in candidates]
            
        actions.append(0) 
        return actions

    def step(self, vehicle_idx, state, action_id, stochastic=False):
        v_state = state['vehicle_states'][vehicle_idx]
        curr_loc = self.cust_coords[v_state['loc']]
        target_loc = self.cust_coords[action_id]
        
        dist = euclidean_distance(curr_loc, target_loc)
        travel_time = StochasticSampler.sample_travel_time(dist) if stochastic else dist
        arrival_time = v_state['time_avail'] + travel_time
        
        transit_cost = dist * TRANSIT_COST_PER_MILE
        wage_min = travel_time
        revenue = 0.0
        penalty = 0.0
        service_time = 0.0
        
        next_unvisited = state['unvisited_ids'].copy()
        next_cap = v_state['capacity']
        
        if action_id != 0:
            cust = self.cust_map[action_id]
            if arrival_time > cust['L']:
                penalty = HARD_LATE_PENALTY
                next_unvisited.discard(action_id)
                service_start = arrival_time
            elif arrival_time < cust['E']:
                wait = cust['E'] - arrival_time
                wage_min += wait
                service_start = cust['E']
                service_time = StochasticSampler.sample_service_time(cust['mean_service_time']) if stochastic else cust['mean_service_time']
                wage_min += service_time
                revenue = cust['demand'] * REVENUE_PER_UNIT
                next_cap -= cust['demand']
                next_unvisited.discard(action_id)
            else:
                service_start = arrival_time
                service_time = StochasticSampler.sample_service_time(cust['mean_service_time']) if stochastic else cust['mean_service_time']
                wage_min += service_time
                revenue = cust['demand'] * REVENUE_PER_UNIT
                next_cap -= cust['demand']
                next_unvisited.discard(action_id)
        else:
            service_start = arrival_time
            
        wage_cost = wage_min * WAGE_COST_PER_MINUTE
        contribution = revenue - (transit_cost + wage_cost + penalty)
        finish_time = service_start + service_time
        
        # --- OPTIMIZED STATE UPDATE (No DeepCopy) ---
        # Copy list of vehicles (shallow copy of list)
        new_vehicle_states = list(state['vehicle_states'])
        # Create NEW dict for the specific vehicle being updated
        next_v_state = {'loc': action_id, 'time_avail': finish_time, 'capacity': next_cap}
        new_vehicle_states[vehicle_idx] = next_v_state
        
        next_state = {
            'current_time': state['current_time'], 
            'unvisited_ids': next_unvisited,
            'vehicle_states': new_vehicle_states
        }
        
        return contribution, next_state, finish_time

# --- HYBRID MCTS AGENT ---
class HybridMCTSAgent:
    def __init__(self, instance, adp_weights, global_statics):
        self.engine = MCTSEngine(instance, adp_weights, global_statics)
        self.instance = instance

    def run_mcts(self, root_state, vehicle_idx):
        root = MCTSNode(state=root_state)
        
        possible_actions = self.engine.get_feasible_actions(vehicle_idx, root_state, prune=True)
        if not possible_actions: return 0 
        root.untried_actions = possible_actions
        
        for _ in range(MCTS_ITERATIONS):
            node = root
            current_v_idx = vehicle_idx
            
            while not node.untried_actions and node.children:
                node = node.best_child()
                
            if node.untried_actions:
                action = node.untried_actions.pop()
                
                avg_reward = 0.0
                rep_next_state = None
                
                for s_idx in range(EXPANSION_SAMPLES):
                    r, next_s, _ = self.engine.step(current_v_idx, node.state, action, stochastic=True)
                    avg_reward += r
                    if s_idx == 0: rep_next_state = next_s
                
                avg_reward /= EXPANSION_SAMPLES
                
                child_node = MCTSNode(state=rep_next_state, parent=node, action_from_parent=action)
                child_node.value = avg_reward 
                child_node.untried_actions = self.engine.get_feasible_actions(current_v_idx, rep_next_state, prune=True)
                node.children.append(child_node)
                node = child_node
                reward = avg_reward
            else:
                reward = 0
                
            # Rollout: ADP GUIDED
            # Optimization: Use reference since step() is non-destructive
            temp_state = node.state
            rollout_reward = reward 
            
            for _ in range(ROLLOUT_DEPTH):
                if not temp_state['unvisited_ids']: break
                
                acts = self.engine.get_feasible_actions(current_v_idx, temp_state, prune=True)
                if not acts: break
                
                best_a = 0
                best_val = -float('inf')
                
                for a in acts:
                    contrib, s_next, _ = self.engine.step(current_v_idx, temp_state, a, stochastic=False)
                    
                    if a == 0: v_adp = 0 
                    else: v_adp = self.engine.get_adp_value(s_next, current_v_idx)
                        
                    val = contrib + v_adp
                    if val > best_val:
                        best_val = val
                        best_a = a
                
                r, temp_state, _ = self.engine.step(current_v_idx, temp_state, best_a, stochastic=True)
                rollout_reward += r
                if best_a == 0: break 
            
            # Penalize unvisited customers at end of rollout (Training Guidance)
            unvisited_count = len(temp_state['unvisited_ids'])
            if unvisited_count > 0:
                rollout_reward -= (unvisited_count * HARD_LATE_PENALTY)
            
            while node is not None:
                node.visits += 1
                node.value += rollout_reward
                node = node.parent
                
        if not root.children: return 0
        return max(root.children, key=lambda c: c.visits).action_from_parent

    def run_episode(self):
        state = {
            'current_time': DEPOT_E_TIME,
            'unvisited_ids': set(c['id'] for c in self.instance['customers']),
            'vehicle_states': [{'loc': 0, 'time_avail': DEPOT_E_TIME, 'capacity': self.instance['vehicle_capacity']} 
                               for _ in range(self.instance['num_vehicles'])]
        }
        
        events = [(DEPOT_E_TIME, v) for v in range(self.instance['num_vehicles'])]
        heapq.heapify(events)
        
        stats = {'cost': 0.0, 'missed': 0, 'util_time': 0.0, 'hard_late': 0}
        
        while events:
            time_now, v_idx = heapq.heappop(events)
            if time_now > DEPOT_L_TIME: break
            
            state['current_time'] = time_now
            state['vehicle_states'][v_idx]['time_avail'] = time_now
            
            action = self.run_mcts(state, v_idx)
            
            revenue = 0
            contrib, next_state, finish_time = self.engine.step(v_idx, state, action, stochastic=True)
            
            if action != 0:
                if action not in next_state['unvisited_ids'] and action in state['unvisited_ids']:
                    revenue = self.engine.cust_map[action]['demand'] * REVENUE_PER_UNIT
            
            op_cost = revenue - contrib
            stats['cost'] += op_cost
            if op_cost >= HARD_LATE_PENALTY: stats['hard_late'] += 1
            
            if action != 0 and op_cost < HARD_LATE_PENALTY:
                stats['util_time'] += (finish_time - time_now) 

            state = next_state
            
            if finish_time < DEPOT_L_TIME:
                if action != 0:
                    heapq.heappush(events, (finish_time, v_idx))
                elif action == 0 and state['unvisited_ids']:
                     if time_now + 30 < DEPOT_L_TIME: heapq.heappush(events, (time_now + 30, v_idx))

        # Record missed stats (Reporting Only - No added cost)
        stats['missed'] = len(state['unvisited_ids'])
        return stats

# --- WORKER ---
def process_instance(filepath, adp_weights, global_statics):
    try:
        data = load_instance(filepath)
        if isinstance(data['customers'], pd.DataFrame):
            data['customers'] = data['customers'].to_dict(orient='records')
            
        data['cust_map'] = {c['id']: c for c in data['customers']}
        data['cust_map'][0] = data['depot']
        data['cust_coords'] = {c['id']: (c['x'], c['y']) for c in data['customers']}
        data['cust_coords'][0] = (data['depot']['x'], data['depot']['y'])

        agent = HybridMCTSAgent(data, adp_weights, global_statics)
        
        costs = []
        missed = []
        utils = []
        
        for _ in range(EVAL_SIMULATIONS):
            res = agent.run_episode()
            costs.append(res['cost'])
            missed.append(res['missed'] + res['hard_late'])
            cap = data['num_vehicles'] * (DEPOT_L_TIME - DEPOT_E_TIME)
            utils.append(res['util_time'] / max(1, cap))
            
        avg_cost = np.mean(costs)
        avg_miss = np.mean(missed)
        avg_util = np.mean(utils)
        
        res_data = {
            'instance_file': os.path.basename(filepath),
            'policy_type': 'MCTS_Hybrid_ADP',
            'metrics': {
                'mean_total_cost': avg_cost,
                'mean_missed_customers': avg_miss,
                'fleet_utilization': avg_util
            }
        }
        
        out_name = os.path.basename(filepath).replace('.json', '') + '_mcts_hybrid_results.json'
        with open(os.path.join(HYBRID_RESULTS_DIR, out_name), 'w') as f:
            json.dump(res_data, f, indent=4)
            
        return f"{os.path.basename(filepath)}: Cost ${avg_cost:.0f}"
        
    except Exception as e:
        return f"Error {filepath}: {e}"

def run_hybrid_pipeline():
    files = sorted([os.path.join(BASE_DATA_DIR, f) for f in os.listdir(BASE_DATA_DIR) if f.endswith('.json')])
    if not files: return

    if not os.path.exists(GENETIC_WEIGHTS_FILE):
        print("Error: Genetic weights not found. Please run genetic_optimizer.py first.")
        return
    
    with open(GENETIC_WEIGHTS_FILE, 'rb') as f:
        saved_data = pickle.load(f)
        adp_weights = saved_data['weights']
        global_statics = saved_data['global_statics']
    
    print(f"--- Starting HYBRID MCTS Evaluation on {len(files)} Instances ---")
    print(f"Using Weights: {adp_weights}")
    
    num_workers = os.cpu_count()
    print(f"Utilizing {num_workers} cores.")
    
    start = time.time()
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_instance, f, adp_weights, global_statics): f for f in files}
        
        completed = 0
        for future in as_completed(futures):
            completed += 1
            print(f"[{completed}/{len(files)}] {future.result()}")
            
    print(f"Completed in {time.time() - start:.1f}s")

if __name__ == '__main__':
    run_hybrid_pipeline()