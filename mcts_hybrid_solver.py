import numpy as np
import pandas as pd
import os
import json
import copy
import heapq
import math
import time
import torch
import torch.nn as nn
from tqdm import tqdm 
from concurrent.futures import ProcessPoolExecutor, as_completed

from config import (
    WAGE_COST_PER_MINUTE, TRANSIT_COST_PER_MILE,
    DEPOT_E_TIME, DEPOT_L_TIME, HARD_LATE_PENALTY,
    SERVICE_TIME_BASE_MEAN, COORDINATE_BOUNDS
)
from simulator import StochasticSampler
from deterministic_policy_generator import load_instance
from data_generator import euclidean_distance

# --- HYPERPARAMETERS ---
MCTS_ITERATIONS = 250       
EXPLORATION_CONSTANT = 3.0  
MAX_CHILDREN = 10           
EXPANSION_SAMPLES = 3       
GAMMA = 0.99                

# Economic Params
REVENUE_PER_UNIT = 50.0     # Aggressive revenue to incentivize service
RECHARGE_DELAY = 30         # Minutes to wait at depot before becoming active again

# Evaluation
EVAL_SIMULATIONS = 30       

# --- PATHS ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DATA_DIR = os.path.join(SCRIPT_DIR, 'instances', 'data')
SOLUTIONS_DIR = os.path.join(SCRIPT_DIR, 'solutions')
RL_MODEL_PATH = os.path.join(SOLUTIONS_DIR, 'RL', 'vrp_dqn.pth')

SYSTEM_SOLUTIONS_DIR = os.path.join(SOLUTIONS_DIR, 'MCTS_System')
SYSTEM_RESULTS_DIR = os.path.join(SYSTEM_SOLUTIONS_DIR, 'simulation_results')

os.makedirs(SYSTEM_SOLUTIONS_DIR, exist_ok=True)
os.makedirs(SYSTEM_RESULTS_DIR, exist_ok=True)

# --- RL MODEL DEFINITION ---
class VRP_DQN(nn.Module):
    def __init__(self, global_input_dim=2, node_input_dim=6, hidden_dim=128):
        super(VRP_DQN, self).__init__()
        self.global_net = nn.Sequential(nn.Linear(global_input_dim, 32), nn.ReLU(), nn.Linear(32, 32))
        self.node_net = nn.Sequential(nn.Linear(node_input_dim, 32), nn.ReLU(), nn.Linear(32, 32))
        self.scorer = nn.Sequential(nn.Linear(64, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        
    def forward(self, global_feats, node_feats):
        batch_size = global_feats.size(0)
        num_nodes = node_feats.size(1)
        g_emb = self.global_net(global_feats).unsqueeze(1).expand(-1, num_nodes, -1)
        n_emb = self.node_net(node_feats)
        combined = torch.cat([g_emb, n_emb], dim=2)
        return self.scorer(combined).squeeze(2)

# --- RL INTERFACE ---
class RLValuePredictor:
    def __init__(self, model_path, device='cpu'):
        self.device = device
        self.model = VRP_DQN().to(device)
        self.loaded = False
        if os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location=device)
                self.model.load_state_dict(state_dict)
                self.model.eval()
                self.loaded = True
            except Exception as e:
                print(f"RL Load Error: {e}")
        
        self.max_dist = euclidean_distance((0,0), (100,100))
        self.time_horizon = DEPOT_L_TIME - DEPOT_E_TIME

    def get_priors(self, instance, active_v_state, unvisited_ids, global_time):
        """
        Queries RL model for priors. 
        Returns map {node_id: q_value} for feasible nodes.
        """
        if not self.loaded: return {}

        # 1. Global Features
        time_norm = (active_v_state['time'] - DEPOT_E_TIME) / self.time_horizon
        cap_norm = active_v_state['cap'] / instance['vehicle_capacity']
        global_feats = torch.tensor([time_norm, cap_norm], dtype=torch.float, device=self.device).unsqueeze(0)
        
        # 2. Node Features
        limit = 200 
        node_feats_np = np.zeros((limit, 6), dtype=np.float32)
        mask_np = np.zeros(limit, dtype=bool)
        
        curr_loc = instance['cust_coords'][active_v_state['loc']]
        
        # Depot
        dist_0 = euclidean_distance(curr_loc, instance['cust_coords'][0])
        node_feats_np[0] = [dist_0/self.max_dist, 0, 0, 1, 0, 1.0]
        mask_np[0] = True
        
        max_dem = instance.get('max_demand', 1.0)
        
        # Customers
        for cid in instance['cust_map']:
            if cid == 0: continue
            
            cust = instance['cust_map'][cid]
            d = euclidean_distance(curr_loc, instance['cust_coords'][cid])
            is_visited = cid not in unvisited_ids
            
            # Mask
            is_feasible = True
            if is_visited: is_feasible = False
            if cust['demand'] > active_v_state['cap']: is_feasible = False
            arr_est = active_v_state['time'] + d
            if arr_est > cust['L']: is_feasible = False
            
            mask_np[cid] = is_feasible
            
            gap = (cust['L'] - arr_est) / 60.0
            node_feats_np[cid] = [
                d / self.max_dist,
                cust['demand'] / max_dem,
                (cust['E'] - DEPOT_E_TIME) / self.time_horizon,
                (cust['L'] - DEPOT_E_TIME) / self.time_horizon,
                1.0 if is_visited else 0.0,
                gap
            ]
            
        node_tensor = torch.tensor(node_feats_np, dtype=torch.float, device=self.device).unsqueeze(0)
        
        with torch.no_grad():
            q_values = self.model(global_feats, node_tensor).squeeze(0).cpu().numpy()
            
        priors = {}
        for i in range(len(instance['cust_map'])):
            if mask_np[i]: priors[i] = q_values[i]
        return priors

# --- MCTS NODE ---
class MCTSNode:
    def __init__(self, state, parent=None, action=None, prior=0.0):
        self.state = state
        self.parent = parent
        self.action = action 
        self.children = []
        self.visits = 0
        self.value = 0.0 
        self.untried_actions = None 
        self.prior = prior
        self.priors_cache = {} # Cache for children's priors

    def best_child(self, c_param=EXPLORATION_CONSTANT):
        # Normalize priors if map exists in parent
        priors_list = np.array([c.prior for c in self.children])
        
        if len(priors_list) > 0:
            p_min, p_max = priors_list.min(), priors_list.max()
            if p_max > p_min:
                # Softmax-ish or MinMax normalization
                priors_list = (priors_list - p_min) / (p_max - p_min) 
            else:
                priors_list = np.ones_like(priors_list) / len(priors_list)
        
        scores = []
        sqrt_visits = math.sqrt(self.visits)
        
        for i, child in enumerate(self.children):
            q_val = child.value / child.visits if child.visits > 0 else 0
            p_val = priors_list[i]
            u_val = c_param * p_val * sqrt_visits / (1 + child.visits)
            scores.append(q_val + u_val)
            
        return self.children[np.argmax(scores)]

# --- SYSTEM ENGINE ---
class SystemEngine:
    def __init__(self, instance):
        self.instance = instance
        self.cust_map = {c['id']: c for c in instance['customers']}
        self.cust_map[0] = instance['depot']
        self.cust_coords = {c['id']: (c['x'], c['y']) for c in instance['customers']}
        self.cust_coords[0] = (instance['depot']['x'], instance['depot']['y'])
        self.action_cache = {}

    def get_active_vehicle(self, vehicle_states):
        """Returns index of vehicle with minimum availability time."""
        best_v = -1
        min_time = float('inf')
        
        for i, v in enumerate(vehicle_states):
            # Only consider vehicles that haven't timed out
            if v['time'] < min_time and v['time'] < DEPOT_L_TIME:
                min_time = v['time']
                best_v = i
                
        return best_v

    def _get_cache_key(self, v_idx, v_loc, v_time, v_cap, unvisited_mask):
        return (v_idx, v_loc, round(v_time, 1), v_cap, unvisited_mask)

    def get_feasible_actions(self, v_idx, vehicle_states, unvisited_ids):
        v = vehicle_states[v_idx]
        mask = 0
        for cid in unvisited_ids: mask |= (1 << cid)
        key = self._get_cache_key(v_idx, v['loc'], v['time'], v['cap'], mask)
        
        if key in self.action_cache: return list(self.action_cache[key])
        
        curr_loc = self.cust_coords[v['loc']]
        actions = []
        
        for cid in unvisited_ids:
            cust = self.cust_map[cid]
            if cust['demand'] > v['cap']: continue
            dist = euclidean_distance(curr_loc, (cust['x'], cust['y']))
            if v['time'] + dist <= cust['L']:
                actions.append(cid)
                
        actions.append(0) 
        self.action_cache[key] = tuple(actions)
        return list(actions)

    def step(self, v_idx, vehicle_states, unvisited_ids, action, stochastic=False):
        v = vehicle_states[v_idx]
        curr_loc = self.cust_coords[v['loc']]
        target_loc = self.cust_coords[action]
        
        dist = euclidean_distance(curr_loc, target_loc)
        travel_time = StochasticSampler.sample_travel_time(dist) if stochastic else dist
        arrival = v['time'] + travel_time
        
        transit_cost = dist * TRANSIT_COST_PER_MILE
        wage_billable = travel_time
        
        rev = 0; pen = 0; svc = 0; wait = 0; outcome = 'SUCCESS'
        
        next_unvisited = unvisited_ids.copy()
        next_cap = v['cap']
        
        if action == 0:
            outcome = 'DEPOT_END'
            svc_start = arrival
        else:
            cust = self.cust_map[action]
            if arrival > cust['L']:
                outcome = 'LATE_SKIP'
                pen = HARD_LATE_PENALTY
                next_unvisited.discard(action)
                svc_start = arrival
            elif arrival < cust['E']:
                wait = cust['E'] - arrival
                svc_start = cust['E']
                if v['loc'] != 0 or action != 0: wage_billable += wait
                svc = StochasticSampler.sample_service_time(cust['mean_service_time']) if stochastic else cust['mean_service_time']
                wage_billable += svc
                rev = cust['demand'] * REVENUE_PER_UNIT # Dynamic revenue
                next_cap -= cust['demand']
                next_unvisited.discard(action)
            else:
                svc_start = arrival
                svc = StochasticSampler.sample_service_time(cust['mean_service_time']) if stochastic else cust['mean_service_time']
                wage_billable += svc
                rev = cust['demand'] * REVENUE_PER_UNIT
                next_cap -= cust['demand']
                next_unvisited.discard(action)
                
        wage_cost = wage_billable * WAGE_COST_PER_MINUTE
        total_op = transit_cost + wage_cost + pen
        reward = rev - total_op
        finish_time = svc_start + svc
        
        # New State Components
        new_v_states = [vs.copy() for vs in vehicle_states]
        new_v_states[v_idx] = {'loc': action, 'time': finish_time, 'cap': next_cap}
        
        log = {
            'node_id': action, 'outcome': outcome, 'arrival_time': float(arrival),
            'service_start': float(svc_start), 'departure_time': float(finish_time),
            'wait_time': float(wait), 'service_duration': float(svc),
            'transit_cost': float(transit_cost), 'wage_cost': float(wage_cost), 
            'penalty_cost': float(pen), 'dist': float(dist)
        }
        
        return reward, new_v_states, next_unvisited, finish_time, log

# --- SYSTEM AGENT ---
class SystemMCTSAgent:
    def __init__(self, instance, rl_predictor):
        self.engine = SystemEngine(instance)
        self.rl = rl_predictor
        self.instance = instance

    def run_mcts(self, vehicle_states, unvisited_ids):
        # 1. Identify Active Vehicle
        v_idx = self.engine.get_active_vehicle(vehicle_states)
        if v_idx == -1: return None, None
        
        # Root State
        root_state = {
            'vehicle_states': vehicle_states,
            'unvisited_ids': unvisited_ids,
            'active_idx': v_idx
        }
        
        root = MCTSNode(root_state)
        
        # 2. Feasible Actions
        feasible = self.engine.get_feasible_actions(v_idx, vehicle_states, unvisited_ids)
        if not feasible: return None, 0
        
        # 3. RL Priors (ROOT)
        # We query the RL model for the active vehicle at the root state
        priors_map = self.rl.get_priors(self.instance, vehicle_states[v_idx], unvisited_ids, vehicle_states[v_idx]['time'])
        
        scored = []
        for a in feasible:
            val = priors_map.get(a, -1e3)
            scored.append((val, a))
        scored.sort(key=lambda x: x[0], reverse=True)
        
        top_actions = [x[1] for x in scored[:MAX_CHILDREN]]
        root.untried_actions = top_actions
        
        # Cache for Root
        root.priors_cache = {a: v for v, a in scored}

        # --- SEARCH LOOP ---
        for _ in range(MCTS_ITERATIONS):
            node = root
            
            # Select
            while not node.untried_actions and node.children:
                node = node.best_child()
                
            # Expand
            if node.untried_actions:
                action = node.untried_actions.pop()
                
                curr_v_idx = node.state['active_idx']
                curr_v_states = node.state['vehicle_states']
                curr_unvisited = node.state['unvisited_ids']
                
                # Expansion Sampling
                avg_r = 0
                rep_v_states = None
                rep_unvisited = None
                
                for i in range(EXPANSION_SAMPLES):
                    r, n_v_states, n_unvisited, _, _ = self.engine.step(curr_v_idx, curr_v_states, curr_unvisited, action, stochastic=True)
                    avg_r += r
                    if i == 0:
                        rep_v_states = n_v_states
                        rep_unvisited = n_unvisited
                
                avg_r /= EXPANSION_SAMPLES
                
                next_active_idx = self.engine.get_active_vehicle(rep_v_states)
                
                child_state = {
                    'vehicle_states': rep_v_states,
                    'unvisited_ids': rep_unvisited,
                    'active_idx': next_active_idx
                }
                
                # Get Prior for the action leading TO this child (stored in parent)
                prior_to_child = 0.0
                if hasattr(node, 'priors_cache'):
                    prior_to_child = node.priors_cache.get(action, 0.0)
                
                child = MCTSNode(child_state, parent=node, action=action, prior=prior_to_child)
                
                # --- FIX 1: DEEP NEURAL GUIDANCE ---
                # We are expanding this child. We need to prepare IT to select ITS children.
                # Calculate feasible actions for the NEXT active vehicle
                if next_active_idx != -1:
                    child_feasible = self.engine.get_feasible_actions(next_active_idx, rep_v_states, rep_unvisited)
                    
                    # Query RL for the NEXT vehicle
                    child_priors_map = self.rl.get_priors(self.instance, rep_v_states[next_active_idx], rep_unvisited, rep_v_states[next_active_idx]['time'])
                    
                    # Store these priors in the child so it can use them when it becomes a parent
                    child.priors_cache = child_priors_map
                    
                    # Sort/Prune
                    scored_child = []
                    for a_c in child_feasible:
                        val_c = child_priors_map.get(a_c, -1e3)
                        scored_child.append((val_c, a_c))
                    scored_child.sort(key=lambda x: x[0], reverse=True)
                    
                    child.untried_actions = [x[1] for x in scored_child[:MAX_CHILDREN]]
                else:
                    child.untried_actions = []
                
                node.children.append(child)
                node = child
                
                # Value Estimate (Leaf Evaluation)
                leaf_val = avg_r
                if next_active_idx != -1 and hasattr(child, 'priors_cache') and child.priors_cache:
                    # Bootstrapping: Use max Q-value of next state as heuristic
                    leaf_val += GAMMA * max(child.priors_cache.values())
                        
                backprop_val = leaf_val
            else:
                backprop_val = 0
            
            # Backprop
            while node is not None:
                node.visits += 1
                node.value += backprop_val
                node = node.parent
                
        # Final Decision
        if not root.children: return v_idx, 0
        best_child = max(root.children, key=lambda c: c.visits)
        return v_idx, best_child.action

    def run_episode(self):
        vehicle_states = [{'loc': 0, 'time': DEPOT_E_TIME, 'cap': self.instance['vehicle_capacity']} 
                          for _ in range(self.instance['num_vehicles'])]
        unvisited_ids = set(c['id'] for c in self.instance['customers'])
        
        traces = [[] for _ in range(self.instance['num_vehicles'])]
        for v in range(self.instance['num_vehicles']):
            traces[v].append({
                'node_id': 0, 'outcome': 'DEPOT_START', 'arrival_time': DEPOT_E_TIME, 
                'service_start': DEPOT_E_TIME, 'departure_time': DEPOT_E_TIME, 
                'wait_time': 0, 'service_duration': 0, 'transit_cost': 0, 'wage_cost': 0, 'penalty_cost': 0, 'dist': 0
            })
            
        metrics = {'cost': 0, 'hard_late': 0}
        
        while True:
            v_idx = self.engine.get_active_vehicle(vehicle_states)
            if v_idx == -1: break 
            
            _, action = self.run_mcts(vehicle_states, unvisited_ids)
            
            _, next_v_states, next_unvisited, _, log = self.engine.step(v_idx, vehicle_states, unvisited_ids, action, stochastic=True)
            
            metrics['cost'] += (log['transit_cost'] + log['wage_cost'] + log['penalty_cost'])
            if log['outcome'] == 'LATE_SKIP': metrics['hard_late'] += 1
            
            traces[v_idx].append(log)
            vehicle_states = next_v_states
            unvisited_ids = next_unvisited
            
            # --- FIX 2: RECHARGE DELAY ---
            if action == 0:
                # Instead of DEPOT_L_TIME + 1, add delay to simulate restocking/break
                vehicle_states[v_idx]['time'] += RECHARGE_DELAY 
                
        metrics['missed'] = len(unvisited_ids)
        return metrics, traces

# --- WORKER ---
def process_instance(filepath, rl_model_path):
    try:
        data = load_instance(filepath)
        if isinstance(data['customers'], pd.DataFrame):
            data['customers'] = data['customers'].to_dict(orient='records')
            
        data['max_demand'] = max(c['demand'] for c in data['customers'])
        data['cust_map'] = {c['id']: c for c in data['customers']}
        data['cust_map'][0] = data['depot']
        data['cust_coords'] = {c['id']: (c['x'], c['y']) for c in data['customers']}
        data['cust_coords'][0] = (data['depot']['x'], data['depot']['y'])
        
        rl_pred = RLValuePredictor(rl_model_path, device='cpu')
        agent = SystemMCTSAgent(data, rl_pred)
        
        logs = []
        for i in range(EVAL_SIMULATIONS):
            met, tr = agent.run_episode()
            logs.append({
                'day_index': i,
                'total_cost': met['cost'],
                'hard_lates': met['hard_late'],
                'missed_customers': met['missed'],
                'vehicle_traces': tr
            })
            
        res = {
            'instance_file': os.path.basename(filepath),
            'policy_type': 'MCTS_System_AlphaZero',
            'N': data['num_customers'],
            'V': data['num_vehicles'],
            'daily_simulation_logs': logs
        }
        
        out_name = os.path.basename(filepath).replace('.json', '') + '_mcts_system_results.json'
        with open(os.path.join(SYSTEM_RESULTS_DIR, out_name), 'w') as f:
            json.dump(res, f, indent=4)
            
        avg_c = np.mean([l['total_cost'] for l in logs])
        return f"{os.path.basename(filepath)}: ${avg_c:.0f}"
        
    except Exception as e:
        return f"Error {filepath}: {e}"

def run_pipeline():
    if not os.path.exists(RL_MODEL_PATH):
        print("RL Model not found. Aborting.")
        return

    files = sorted([os.path.join(BASE_DATA_DIR, f) for f in os.listdir(BASE_DATA_DIR) if f.endswith('.json')])
    if not files: return
    
    print(f"--- Starting SYSTEM MCTS Evaluation ---")
    workers = max(1, os.cpu_count() - 2)
    
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_instance, f, RL_MODEL_PATH): f for f in files}
        
        with tqdm(total=len(files), unit="inst") as pbar:
            for future in as_completed(futures):
                print(future.result())
                pbar.update(1)

if __name__ == '__main__':
    run_pipeline()