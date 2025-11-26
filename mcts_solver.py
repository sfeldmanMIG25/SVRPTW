import numpy as np
import os
import pandas as pd
import json
import copy
import heapq
import math
import time
from tqdm import tqdm 
from concurrent.futures import ProcessPoolExecutor, as_completed

from config import (
    WAGE_COST_PER_MINUTE, TRANSIT_COST_PER_MILE,
    DEPOT_E_TIME, DEPOT_L_TIME, HARD_LATE_PENALTY,
    SERVICE_TIME_BASE_MEAN # Added for heuristic estimates
)
from simulator import StochasticSampler
from deterministic_policy_generator import load_instance
from data_generator import euclidean_distance

# --- HYPERPARAMETERS ---
MCTS_ITERATIONS = 250       
EXPLORATION_CONSTANT = 1.41 
ROLLOUT_DEPTH = 12          
MAX_CHILDREN = 8            
URGENCY_WEIGHT = 0.2        

# Evaluation Settings
EVAL_SIMULATIONS = 30       

# --- PATHS ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DATA_DIR = os.path.join(SCRIPT_DIR, 'instances', 'data')
MCTS_SOLUTIONS_DIR = os.path.join(SCRIPT_DIR, 'solutions', 'MCTS')
MCTS_RESULTS_DIR = os.path.join(MCTS_SOLUTIONS_DIR, 'simulation_results')

os.makedirs(MCTS_SOLUTIONS_DIR, exist_ok=True)
os.makedirs(MCTS_RESULTS_DIR, exist_ok=True)

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
        # Vectorized UCB calculation for speed
        if not self.children: return None
        
        visits = np.array([child.visits for child in self.children])
        values = np.array([child.value for child in self.children])
        
        # Avoid div by zero
        mask = visits > 0
        
        scores = np.zeros(len(self.children))
        
        # Unvisited children get high priority (infinite UCB)
        # But in standard MCTS we expand untried first, so children here usually have visits.
        # If visits is 0, make score infinite
        scores[~mask] = float('inf')
        
        if self.visits > 0:
            log_term = math.log(self.visits)
            # UCB1 Formula
            exploitation = np.zeros_like(values)
            exploitation[mask] = values[mask] / visits[mask]
            
            exploration = np.zeros_like(visits, dtype=float)
            exploration[mask] = c_param * np.sqrt(2 * log_term / visits[mask])
            
            scores[mask] = exploitation[mask] + exploration[mask]
            
        return self.children[np.argmax(scores)]

# --- OPTIMIZED ENGINE ---
class MCTSEngine:
    def __init__(self, instance):
        self.instance = instance
        # Pre-compute lookups
        self.cust_map = {c['id']: c for c in instance['customers']}
        self.cust_map[0] = instance['depot']
        self.cust_coords = {c['id']: (c['x'], c['y']) for c in instance['customers']}
        self.cust_coords[0] = (instance['depot']['x'], instance['depot']['y'])
        
        # Cache for get_feasible_actions
        self.action_cache = {}

    def _get_state_key(self, vehicle_idx, state):
        """
        Generates a hashable key for the state to use in caching.
        Uses Bitmasks for the unvisited set for speed.
        """
        # 1. Current Vehicle State (Loc, Time)
        v_state = state['vehicle_states'][vehicle_idx]
        v_tuple = (v_state['loc'], round(v_state['time_avail'], 2))
        
        # 2. Unvisited Set (Bitmask)
        # Assuming IDs are 1..N. Node 0 is depot (never in unvisited).
        # Python handles large integers automatically, so this works for N=100.
        unvisited_mask = 0
        for cid in state['unvisited_ids']:
            unvisited_mask |= (1 << cid)
            
        return (vehicle_idx, v_tuple, unvisited_mask)

    def shallow_copy_state(self, state):
        """
        Replaces copy.deepcopy with a manual shallow copy.
        Much faster because it doesn't recurse into immutable objects (ints/tuples).
        """
        return {
            'current_time': state['current_time'],
            'unvisited_ids': state['unvisited_ids'].copy(), # Set copy is fast
            # List of dicts: copy the list, and copy each dict
            'vehicle_states': [v.copy() for v in state['vehicle_states']]
        }

    def get_feasible_actions(self, vehicle_idx, state, prune=True):
        # Check Cache
        state_key = self._get_state_key(vehicle_idx, state)
        if state_key in self.action_cache:
            return list(self.action_cache[state_key]) # Return copy of list
        
        v_state = state['vehicle_states'][vehicle_idx]
        curr_loc = self.cust_coords[v_state['loc']]
        curr_time = v_state['time_avail']
        
        candidates = []
        for cid in state['unvisited_ids']:
            cust = self.cust_map[cid]
            if cust['demand'] > v_state['capacity']: continue
            
            dist = euclidean_distance(curr_loc, (cust['x'], cust['y']))
            arrival_est = curr_time + dist
            
            if arrival_est <= cust['L']:
                slack = max(0, cust['L'] - arrival_est)
                # Heuristic Score: Distance + Urgency
                score = dist + (slack * URGENCY_WEIGHT)
                candidates.append((score, cid))
        
        if prune and len(candidates) > MAX_CHILDREN:
            candidates.sort(key=lambda x: x[0]) 
            actions = [c[1] for c in candidates[:MAX_CHILDREN]]
        else:
            actions = [c[1] for c in candidates]
            
        actions.append(0) # Always allow returning to depot
        
        # Store in Cache
        self.action_cache[state_key] = tuple(actions) # Store as tuple to be immutable
        return list(actions)

    def step(self, vehicle_idx, state, action_id, stochastic=False):
        """
        Executes one step. 
        Returns: reward, next_state, finish_time, trace_log
        """
        v_state = state['vehicle_states'][vehicle_idx]
        curr_loc_id = v_state['loc']
        curr_loc = self.cust_coords[curr_loc_id]
        target_loc = self.cust_coords[action_id]
        
        dist = euclidean_distance(curr_loc, target_loc)
        travel_time = StochasticSampler.sample_travel_time(dist) if stochastic else dist
        arrival_time = v_state['time_avail'] + travel_time
        
        transit_cost = dist * TRANSIT_COST_PER_MILE
        wage_billable_min = travel_time 
        
        penalty_cost = 0.0
        service_time = 0.0
        wait_time = 0.0
        outcome = 'SUCCESS'
        
        # Create shallow copies for next state
        next_unvisited = state['unvisited_ids'].copy()
        next_cap = v_state['capacity']
        
        service_start = arrival_time
        
        if action_id == 0:
            # Depot Return
            outcome = 'DEPOT_END'
            # Waiting at depot (0->0) is Unpaid.
            pass 
        else:
            cust = self.cust_map[action_id]
            if arrival_time > cust['L']:
                # Late
                outcome = 'LATE_SKIP'
                penalty_cost = HARD_LATE_PENALTY
                next_unvisited.discard(action_id)
                service_start = arrival_time
                
            elif arrival_time < cust['E']:
                # Early
                wait_time = cust['E'] - arrival_time
                service_start = cust['E']
                
                # Wage Logic: Paid unless staying at depot
                # Here we moved (curr != 0 or action != 0), so wait is PAID.
                wage_billable_min += wait_time
                
                s_time = StochasticSampler.sample_service_time(cust['mean_service_time']) if stochastic else cust['mean_service_time']
                service_time = s_time
                wage_billable_min += s_time
                
                next_cap -= cust['demand']
                next_unvisited.discard(action_id)
            else:
                # On Time
                service_start = arrival_time
                
                s_time = StochasticSampler.sample_service_time(cust['mean_service_time']) if stochastic else cust['mean_service_time']
                service_time = s_time
                wage_billable_min += s_time
                
                next_cap -= cust['demand']
                next_unvisited.discard(action_id)

        wage_cost = wage_billable_min * WAGE_COST_PER_MINUTE
        total_cost = transit_cost + wage_cost + penalty_cost
        reward = -total_cost 
        
        finish_time = service_start + service_time
        
        # Construct Next State (Shallow)
        new_vehicle_states = [v.copy() for v in state['vehicle_states']]
        new_vehicle_states[vehicle_idx] = {
            'loc': action_id, 
            'time_avail': finish_time, 
            'capacity': next_cap
        }
        
        next_state = {
            'current_time': state['current_time'], 
            'unvisited_ids': next_unvisited,
            'vehicle_states': new_vehicle_states
        }
        
        # Trace Log for JSON
        log_data = {
            'node_id': action_id,
            'outcome': outcome,
            'arrival_time': float(arrival_time),
            'service_start': float(service_start),
            'departure_time': float(finish_time),
            'wait_time': float(wait_time),
            'service_duration': float(service_time),
            'transit_cost': float(transit_cost),
            'wage_cost': float(wage_cost),
            'penalty_cost': float(penalty_cost),
            'dist': float(dist)
        }
        
        return reward, next_state, finish_time, log_data

# --- AGENT ---
class MCTSAgent:
    def __init__(self, instance):
        self.engine = MCTSEngine(instance)
        self.instance = instance

    def run_mcts(self, root_state, vehicle_idx):
        root = MCTSNode(state=root_state)
        possible_actions = self.engine.get_feasible_actions(vehicle_idx, root_state, prune=True)
        if not possible_actions: return 0 
        root.untried_actions = possible_actions
        
        for _ in range(MCTS_ITERATIONS):
            node = root
            current_v_idx = vehicle_idx
            
            # Select
            while not node.untried_actions and node.children:
                node = node.best_child()
                # Note: In a real multi-vehicle tree, v_idx would rotate. 
                # Here we simplify assuming MCTS plans for ONE vehicle's next step.
                
            # Expand
            if node.untried_actions:
                action = node.untried_actions.pop()
                # Deterministic Step for Tree Expansion
                reward, next_state, _, _ = self.engine.step(current_v_idx, node.state, action, stochastic=False)
                child_node = MCTSNode(state=next_state, parent=node, action_from_parent=action)
                child_node.untried_actions = self.engine.get_feasible_actions(current_v_idx, next_state, prune=True)
                node.children.append(child_node)
                node = child_node
            else:
                reward = 0
                
            # Rollout
            # Use Manual Shallow Copy for Speed
            temp_state = self.engine.shallow_copy_state(node.state)
            rollout_cum_reward = reward 
            
            for _ in range(ROLLOUT_DEPTH):
                if not temp_state['unvisited_ids']: break
                acts = self.engine.get_feasible_actions(current_v_idx, temp_state, prune=True)
                if not acts: break
                
                # Greedy Rollout Policy
                best_a = 0
                best_score = float('inf')
                curr_v = temp_state['vehicle_states'][current_v_idx]
                curr_loc = self.engine.cust_coords[curr_v['loc']]
                
                for a in acts:
                    if a == 0: continue 
                    cust = self.engine.cust_map[a]
                    d = euclidean_distance(curr_loc, (cust['x'], cust['y']))
                    slack = max(0, cust['L'] - curr_v['time_avail'])
                    score = d + (slack * URGENCY_WEIGHT)
                    if score < best_score:
                        best_score = score
                        best_a = a
                
                # Execute Rollout Step (Stochastic=True reflects environment reality)
                r, temp_state, _, _ = self.engine.step(current_v_idx, temp_state, best_a, stochastic=True)
                rollout_cum_reward += r
                if best_a == 0: break 
            
            # Backprop
            while node is not None:
                node.visits += 1
                node.value += rollout_cum_reward
                node = node.parent
                
        if not root.children: return 0
        return max(root.children, key=lambda c: c.visits).action_from_parent

    def run_episode(self):
        # Initialize State
        state = {
            'current_time': DEPOT_E_TIME,
            'unvisited_ids': set(c['id'] for c in self.instance['customers']),
            'vehicle_states': []
        }
        
        # Initialize Traces
        vehicle_traces = []
        events = []
        
        for v in range(self.instance['num_vehicles']):
            state['vehicle_states'].append({'loc': 0, 'time_avail': DEPOT_E_TIME, 'capacity': self.instance['vehicle_capacity']})
            
            # Initial Depot Trace
            vehicle_traces.append([{
                'node_id': 0,
                'outcome': 'DEPOT_START',
                'arrival_time': DEPOT_E_TIME,
                'service_start': DEPOT_E_TIME,
                'departure_time': DEPOT_E_TIME,
                'wait_time': 0,
                'service_duration': 0,
                'transit_cost': 0, 'wage_cost': 0, 'penalty_cost': 0, 'dist': 0
            }])
            
            heapq.heappush(events, (DEPOT_E_TIME, v))
        
        ep_stats = {
            'total_cost': 0.0, 
            'hard_lates': 0,
            'missed_customers': 0
        }
        
        while events:
            time_now, v_idx = heapq.heappop(events)
            if time_now > DEPOT_L_TIME: break
            
            state['current_time'] = time_now
            state['vehicle_states'][v_idx]['time_avail'] = time_now
            
            # Run MCTS
            action = self.run_mcts(state, v_idx)
            
            # Execute Real Step (Stochastic)
            reward, next_state, finish_time, log = self.engine.step(v_idx, state, action, stochastic=True)
            
            # Record Metrics
            step_cost = log['transit_cost'] + log['wage_cost'] + log['penalty_cost']
            ep_stats['total_cost'] += step_cost
            if log['outcome'] == 'LATE_SKIP': ep_stats['hard_lates'] += 1
            
            vehicle_traces[v_idx].append(log)
            state = next_state
            
            # Schedule Next Event
            if finish_time < DEPOT_L_TIME:
                if action != 0:
                    heapq.heappush(events, (finish_time, v_idx))
                elif action == 0 and state['unvisited_ids']:
                     if time_now + 30 < DEPOT_L_TIME: heapq.heappush(events, (time_now + 30, v_idx))

        ep_stats['missed_customers'] = len(state['unvisited_ids'])
        
        # Full Output Object
        return {
            'total_cost': ep_stats['total_cost'],
            'hard_lates': ep_stats['hard_lates'],
            'missed_customers': ep_stats['missed_customers'],
            'vehicle_traces': vehicle_traces
        }

# --- WORKER ---
def process_instance(filepath):
    try:
        data = load_instance(filepath)
        if isinstance(data['customers'], pd.DataFrame):
            data['customers'] = data['customers'].to_dict(orient='records')
            
        agent = MCTSAgent(data)
        
        daily_logs = []
        
        for day in range(EVAL_SIMULATIONS):
            res = agent.run_episode()
            res['day_index'] = day
            daily_logs.append(res)
            
        # New JSON Style Output
        out_name = os.path.basename(filepath).replace('.json', '') + '_mcts_results.json'
        
        output_data = {
            'instance_file': os.path.basename(filepath),
            'policy_type': 'MCTS_Revised',
            'N': data['num_customers'],
            'V': data['num_vehicles'],
            'daily_simulation_logs': daily_logs
        }
        
        with open(os.path.join(MCTS_RESULTS_DIR, out_name), 'w') as f:
            json.dump(output_data, f, indent=4)
            
        avg_cost = np.mean([l['total_cost'] for l in daily_logs])
        return {
            'file': os.path.basename(filepath),
            'cost': avg_cost
        }
    except Exception as e:
        return {'error': str(e), 'file': os.path.basename(filepath)}

def run_pipeline():
    files = sorted([os.path.join(BASE_DATA_DIR, f) for f in os.listdir(BASE_DATA_DIR) if f.endswith('.json')])
    if not files: return
    
    print(f"--- Starting MCTS Evaluation (Optimized, {EVAL_SIMULATIONS} days/inst) ---")
    print(f"Results will be saved to: {MCTS_RESULTS_DIR}")
    
    num_workers = max(1, os.cpu_count() - 1)
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_instance, f): f for f in files}
        
        with tqdm(total=len(files), desc="Evaluating", unit="inst") as pbar:
            for future in as_completed(futures):
                res = future.result()
                pbar.update(1)
                
    print(f"\nEvaluation Complete.")

if __name__ == '__main__':
    run_pipeline()