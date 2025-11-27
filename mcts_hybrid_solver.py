import numpy as np
import pandas as pd
import os
import json
import copy
import heapq
import math
import time
import pickle
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
MCTS_ITERATIONS = 200       
EXPLORATION_CONSTANT = 3.0  # Increased: RL priors are strong, need exploration
ROLLOUT_DEPTH = 10          
MAX_CHILDREN = 10           
EXPANSION_SAMPLES = 3       
GAMMA = 0.99                

# Evaluation Settings
EVAL_SIMULATIONS = 30       

# --- PATHS ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DATA_DIR = os.path.join(SCRIPT_DIR, 'instances', 'data')
SOLUTIONS_DIR = os.path.join(SCRIPT_DIR, 'solutions')

# 1. ADP Weights (Value Function)
ADP_WEIGHTS_PATH = os.path.join(SOLUTIONS_DIR, 'ADP', 'strategy', 'adp_master_weights.pkl')

# 2. RL Model (Policy/Pricing)
RL_MODEL_PATH = os.path.join(SOLUTIONS_DIR, 'RL', 'vrp_dqn.pth')

UNIFIED_SOLUTIONS_DIR = os.path.join(SOLUTIONS_DIR, 'MCTS_Hybrid')
UNIFIED_RESULTS_DIR = os.path.join(UNIFIED_SOLUTIONS_DIR, 'simulation_results')

os.makedirs(UNIFIED_SOLUTIONS_DIR, exist_ok=True)
os.makedirs(UNIFIED_RESULTS_DIR, exist_ok=True)

# --- NEURAL NETWORK DEFINITION (Must match trained model) ---
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

# --- RL INTERFACE (The Pricing Engine) ---
class RLValuePredictor:
    def __init__(self, model_path, device='cpu'):
        self.device = device
        self.model = VRP_DQN().to(device)
        try:
            state_dict = torch.load(model_path, map_location=device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            self.loaded = True
        except Exception as e:
            print(f"Warning: Could not load RL model: {e}")
            self.loaded = False
            
        # Global Normalization Constants (Hardcoded from training environment)
        self.max_dist = euclidean_distance((0,0), (100,100))
        self.time_horizon = DEPOT_L_TIME - DEPOT_E_TIME
        # Max demand/cap dynamic per instance, will handle in runtime

    def get_q_priors(self, instance, state, vehicle_idx):
        """
        Returns a dictionary {node_id: Q_Value} for all nodes.
        Used to set the 'Prior' (P) in MCTS selection.
        """
        if not self.loaded: return {}

        v_state = state['vehicle_states'][vehicle_idx]
        curr_loc = instance['cust_coords'][v_state['loc']]
        
        # 1. Global Features
        time_norm = (state['current_time'] - DEPOT_E_TIME) / self.time_horizon
        cap_norm = v_state['capacity'] / instance['vehicle_capacity'] # Approximation
        global_feats = torch.tensor([time_norm, cap_norm], dtype=torch.float, device=self.device).unsqueeze(0)
        
        # 2. Node Features
        # Padding to 200 nodes (Model Standard)
        num_nodes_real = len(instance['customers']) + 1
        limit = 200
        
        node_feats_np = np.zeros((limit, 6), dtype=np.float32)
        mask_np = np.zeros(limit, dtype=bool)
        
        # Depot
        dist_0 = euclidean_distance(curr_loc, instance['cust_coords'][0])
        node_feats_np[0] = [dist_0/self.max_dist, 0, 0, 1, 0, 1.0]
        mask_np[0] = True
        
        # Customers
        max_dem = instance['max_demand']
        
        for cid in range(1, num_nodes_real):
            if cid not in instance['cust_map']: continue
            cust = instance['cust_map'][cid]
            
            d = euclidean_distance(curr_loc, instance['cust_coords'][cid])
            is_visited = cid not in state['unvisited_ids']
            
            # Masking
            is_feasible = True
            if is_visited: is_feasible = False
            if cust['demand'] > v_state['capacity']: is_feasible = False
            arr_est = state['current_time'] + d
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
            
        # Convert to dict for MCTS
        priors = {}
        for i in range(num_nodes_real):
            if mask_np[i]:
                priors[i] = q_values[i]
        return priors

# --- ADP FEATURE EXTRACTION ---
def extract_adp_features(state, vehicle_idx, instance, global_statics):
    v_state = state['vehicle_states'][vehicle_idx]
    curr_loc = instance['cust_coords'][v_state['loc']]

    f_bias = 1.0
    unvisited_demand = 0
    closest_dist = global_statics['max_possible_dist']
    sum_tightness = 0
    
    for cid in state['unvisited_ids']:
        c = instance['cust_map'][cid]
        unvisited_demand += c['demand']
        d = euclidean_distance(curr_loc, (c['x'], c['y']))
        if d < closest_dist: closest_dist = d
        sum_tightness += max(0, c['L'] - state['current_time'])

    if not state['unvisited_ids']: closest_dist = 0.0

    av_vehicles = sum(1 for v in state['vehicle_states'] if v['time_avail'] <= state['current_time'])
    cur_cap = sum(v['capacity'] for v in state['vehicle_states'])

    return np.array([
        f_bias,
        unvisited_demand / max(1, global_statics['max_global_demand']),
        closest_dist / global_statics['max_possible_dist'],
        (DEPOT_L_TIME - state['current_time']) / global_statics['time_horizon'],
        (sum_tightness / max(1, len(state['unvisited_ids']))) / global_statics['time_horizon'],
        av_vehicles / max(1, global_statics['max_global_vehicles']),
        cur_cap / max(1, global_statics['max_global_fleet_capacity'])
    ])

# --- MCTS NODE (With RL Prior) ---
class MCTSNode:
    def __init__(self, state, parent=None, action=None, prior=0.0):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0.0 
        self.untried_actions = None 
        self.prior = prior # Value from RL Network

    def is_fully_expanded(self):
        return self.untried_actions is not None and len(self.untried_actions) == 0

    def best_child(self, c_param=EXPLORATION_CONSTANT):
        # AlphaZero PUCT Formula: Q + U
        # U = c * P * sqrt(N_parent) / (1 + N_child)
        # P comes from RL Q-Value (Softmaxed or normalized)
        
        # Safe normalization of priors for UCB
        priors = np.array([c.prior for c in self.children])
        if len(priors) > 0:
            # Q-values can be large/negative. Normalize to [0, 1] range for P logic
            p_min, p_max = priors.min(), priors.max()
            if p_max > p_min:
                priors = (priors - p_min) / (p_max - p_min)
            else:
                priors = np.ones_like(priors) / len(priors) # Uniform if equal
        
        scores = []
        sqrt_visits = math.sqrt(self.visits)
        
        for i, child in enumerate(self.children):
            q_val = child.value / child.visits if child.visits > 0 else 0
            
            # Prior weight (P) from RL
            p_val = priors[i]
            
            # Exploration Term
            u_val = c_param * p_val * sqrt_visits / (1 + child.visits)
            
            scores.append(q_val + u_val)
            
        return self.children[np.argmax(scores)]

# --- ENGINE ---
class MCTSEngine:
    def __init__(self, instance, adp_weights=None, global_statics=None):
        self.instance = instance
        self.adp_weights = adp_weights
        self.global_statics = global_statics
        self.action_cache = {}

    def _get_key(self, v_idx, state):
        v = state['vehicle_states'][v_idx]
        mask = 0
        for cid in state['unvisited_ids']: mask |= (1 << cid)
        return (v_idx, v['loc'], round(v['time_avail'], 1), mask)

    def shallow_copy(self, state):
        return {
            'current_time': state['current_time'],
            'unvisited_ids': state['unvisited_ids'].copy(),
            'vehicle_states': [v.copy() for v in state['vehicle_states']]
        }

    def get_feasible(self, v_idx, state):
        key = self._get_key(v_idx, state)
        if key in self.action_cache: return list(self.action_cache[key])
        
        v = state['vehicle_states'][v_idx]
        curr_loc = self.instance['cust_coords'][v['loc']]
        
        acts = []
        for cid in state['unvisited_ids']:
            cust = self.instance['cust_map'][cid]
            if cust['demand'] > v['capacity']: continue
            dist = euclidean_distance(curr_loc, (cust['x'], cust['y']))
            if v['time_avail'] + dist <= cust['L']:
                acts.append(cid)
        
        acts.append(0)
        self.action_cache[key] = tuple(acts)
        return list(acts)

    def step(self, v_idx, state, action, stochastic=False):
        v = state['vehicle_states'][v_idx]
        curr_loc = self.instance['cust_coords'][v['loc']]
        target_loc = self.instance['cust_coords'][action]
        
        dist = euclidean_distance(curr_loc, target_loc)
        time_travel = StochasticSampler.sample_travel_time(dist) if stochastic else dist
        arr_time = v['time_avail'] + time_travel
        
        transit = dist * TRANSIT_COST_PER_MILE
        wage_min = time_travel
        
        rev = 0; pen = 0; svc = 0; wait = 0; outcome = 'SUCCESS'
        
        next_ids = state['unvisited_ids'].copy()
        next_cap = v['capacity']
        
        if action == 0:
            outcome = 'DEPOT_END'
            start_svc = arr_time
        else:
            cust = self.instance['cust_map'][action]
            if arr_time > cust['L']:
                outcome = 'LATE_SKIP'
                pen = HARD_LATE_PENALTY
                next_ids.discard(action)
                start_svc = arr_time
            elif arr_time < cust['E']:
                wait = cust['E'] - arr_time
                start_svc = cust['E']
                wage_min += wait
                svc = StochasticSampler.sample_service_time(cust['mean_service_time']) if stochastic else cust['mean_service_time']
                wage_min += svc
                rev = cust['demand'] * 10.0
                next_cap -= cust['demand']
                next_ids.discard(action)
            else:
                start_svc = arr_time
                svc = StochasticSampler.sample_service_time(cust['mean_service_time']) if stochastic else cust['mean_service_time']
                wage_min += svc
                rev = cust['demand'] * 10.0
                next_cap -= cust['demand']
                next_ids.discard(action)
                
        wage = wage_min * WAGE_COST_PER_MINUTE
        contrib = rev - (transit + wage + pen)
        end_time = start_svc + svc
        
        new_vehs = list(state['vehicle_states'])
        new_vehs[v_idx] = {'loc': action, 'time_avail': end_time, 'capacity': next_cap}
        
        next_s = {'current_time': state['current_time'], 'unvisited_ids': next_ids, 'vehicle_states': new_vehs}
        
        log = {
            'node_id': action, 'outcome': outcome, 
            'arrival_time': float(arr_time), 'service_start': float(start_svc), 'departure_time': float(end_time),
            'wait_time': float(wait), 'service_duration': float(svc),
            'transit_cost': float(transit), 'wage_cost': float(wage), 'penalty_cost': float(pen), 'dist': float(dist)
        }
        return contrib, next_s, end_time, log

# --- UNIFIED AGENT ---
class UnifiedAgent:
    def __init__(self, instance, adp_weights, global_statics, rl_predictor):
        self.engine = MCTSEngine(instance, adp_weights, global_statics)
        self.rl_predictor = rl_predictor
        self.instance = instance

    def run_mcts(self, root_state, v_idx):
        root = MCTSNode(root_state)
        
        feasible = self.engine.get_feasible(v_idx, root_state)
        if not feasible: return 0
        
        # --- RL INTEGRATION: SET PRIORS ---
        # Get Q-values from RL Network for the current state
        q_priors = self.rl_predictor.get_q_priors(self.instance, root_state, v_idx)
        
        # Sort feasible actions by RL Q-value (descending) for pruning
        scored_actions = []
        for a in feasible:
            # Default small value if RL didn't return it (e.g. mask mismatch)
            val = q_priors.get(a, -1e5)
            scored_actions.append((val, a))
            
        # Pruning: Only consider top children
        scored_actions.sort(key=lambda x: x[0], reverse=True)
        top_actions = [x[1] for x in scored_actions[:MAX_CHILDREN]]
        
        root.untried_actions = top_actions
        
        # Cache priors in the node wrapper or logic?
        # We actually need to attach priors to children when they are created.
        # For now, we store the map in the root for easy access during expansion
        root.q_map = {a: v for v, a in scored_actions}

        for _ in range(MCTS_ITERATIONS):
            node = root
            
            # Select
            while not node.untried_actions and node.children:
                node = node.best_child()
            
            # Expand
            if node.untried_actions:
                action = node.untried_actions.pop()
                
                # Expansion Sampling (Stochastic Robustness)
                avg_imm = 0
                rep_state = None
                for i in range(EXPANSION_SAMPLES):
                    r, ns, _, _ = self.engine.step(v_idx, node.state, action, stochastic=True)
                    avg_imm += r
                    if i == 0: rep_state = ns
                avg_imm /= EXPANSION_SAMPLES
                
                # Get Prior for this node (for future selection)
                # If this is depth 1, we have it from Root RL. 
                # Deeper nodes: We skip RL inference (too slow) and set uniform priors
                prior = 0.0
                if hasattr(node, 'q_map'):
                    prior = node.q_map.get(action, 0.0)
                
                child = MCTSNode(rep_state, parent=node, action=action, prior=prior)
                child.untried_actions = self.engine.get_feasible(v_idx, rep_state)
                node.children.append(child)
                node = child
                
                # ADP Value Init
                adp_val = 0
                if self.engine.adp_weights is not None:
                    feats = extract_adp_features(rep_state, v_idx, self.instance, self.engine.global_statics)
                    adp_val = np.dot(self.engine.adp_weights, feats)
                
                val = avg_imm + (GAMMA * adp_val)
            else:
                val = 0
                
            # Rollout (ADP Guided)
            temp = self.engine.shallow_copy(node.state)
            roll_rew = val
            
            for _ in range(ROLLOUT_DEPTH):
                if not temp['unvisited_ids']: break
                acts = self.engine.get_feasible(v_idx, temp)
                if not acts: break
                
                # Greedy via ADP
                best_a = 0; best_v = -float('inf')
                for a in acts:
                    imm, nxt, _, _ = self.engine.step(v_idx, temp, a, stochastic=False)
                    v_next = 0
                    if a != 0 and self.engine.adp_weights is not None:
                        f = extract_adp_features(nxt, v_idx, self.instance, self.engine.global_statics)
                        v_next = np.dot(self.engine.adp_weights, f)
                    
                    score = imm + v_next
                    if score > best_v:
                        best_v = score; best_a = a
                
                r, temp, _, _ = self.engine.step(v_idx, temp, best_a, stochastic=True)
                roll_rew += r
                if best_a == 0: break
            
            # Backprop
            while node is not None:
                node.visits += 1
                node.value += roll_rew
                node = node.parent
                
        if not root.children: return 0
        return max(root.children, key=lambda c: c.visits).action

    def run_episode(self):
        state = {
            'current_time': DEPOT_E_TIME,
            'unvisited_ids': set(c['id'] for c in self.instance['customers']),
            'vehicle_states': [{'loc': 0, 'time_avail': DEPOT_E_TIME, 'capacity': self.instance['vehicle_capacity']} 
                               for _ in range(self.instance['num_vehicles'])]
        }
        
        traces = [[] for _ in range(self.instance['num_vehicles'])]
        for v in range(self.instance['num_vehicles']):
            traces[v].append({'node_id': 0, 'outcome': 'DEPOT_START', 'arrival_time': DEPOT_E_TIME, 
                              'service_start': DEPOT_E_TIME, 'departure_time': DEPOT_E_TIME, 
                              'wait_time': 0, 'service_duration': 0, 'transit_cost': 0, 'wage_cost': 0, 'penalty_cost': 0, 'dist': 0})
            
        events = [(DEPOT_E_TIME, v) for v in range(self.instance['num_vehicles'])]
        heapq.heapify(events)
        
        metrics = {'cost': 0, 'missed': 0, 'hard_late': 0}
        
        while events:
            t, v = heapq.heappop(events)
            if t > DEPOT_L_TIME: break
            state['current_time'] = t
            state['vehicle_states'][v]['time_avail'] = t
            
            # Unified Decision
            action = self.run_mcts(state, v)
            
            _, next_s, end_t, log = self.engine.step(v, state, action, stochastic=True)
            
            metrics['cost'] += log['transit_cost'] + log['wage_cost'] + log['penalty_cost']
            if log['outcome'] == 'LATE_SKIP': metrics['hard_late'] += 1
            
            traces[v].append(log)
            state = next_s
            
            if end_t < DEPOT_L_TIME:
                if action != 0:
                    heapq.heappush(events, (end_t, v))
                elif action == 0 and state['unvisited_ids']:
                    if t + 30 < DEPOT_L_TIME: heapq.heappush(events, (t + 30, v))
                    
        metrics['missed'] = len(state['unvisited_ids'])
        return metrics, traces

# --- WORKER ---
def process_instance(filepath, adp_weights, global_statics, rl_model_path):
    try:
        data = load_instance(filepath)
        if isinstance(data['customers'], pd.DataFrame):
            data['customers'] = data['customers'].to_dict(orient='records')
            
        data['cust_map'] = {c['id']: c for c in data['customers']}
        data['cust_map'][0] = data['depot']
        data['cust_coords'] = {c['id']: (c['x'], c['y']) for c in data['customers']}
        data['cust_coords'][0] = (data['depot']['x'], data['depot']['y'])
        
        # Max demand needed for RL normalization
        data['max_demand'] = global_statics['max_global_demand']

        # Load RL Predictor (CPU per worker)
        rl_pred = RLValuePredictor(rl_model_path, device='cpu')
        
        agent = UnifiedAgent(data, adp_weights, global_statics, rl_pred)
        
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
            
        # Save
        res = {
            'instance_file': os.path.basename(filepath),
            'policy_type': 'MCTS_Unified_AlphaZero',
            'N': data['num_customers'],
            'V': data['num_vehicles'],
            'daily_simulation_logs': logs
        }
        out_name = os.path.basename(filepath).replace('.json', '') + '_mcts_unified_results.json'
        with open(os.path.join(UNIFIED_RESULTS_DIR, out_name), 'w') as f:
            json.dump(res, f, indent=4)
            
        avg_c = np.mean([l['total_cost'] for l in logs])
        return f"{os.path.basename(filepath)}: ${avg_c:.0f}"
        
    except Exception as e:
        return f"Error {filepath}: {e}"

def run_pipeline():
    # Load ADP Weights
    adp_weights = None; global_statics = None
    if os.path.exists(ADP_WEIGHTS_PATH):
        with open(ADP_WEIGHTS_PATH, 'rb') as f:
            d = pickle.load(f)
            adp_weights = d['weights']
            global_statics = d['global_statics']
    else:
        print("ADP Weights not found. Aborting.")
        return

    if not os.path.exists(RL_MODEL_PATH):
        print("RL Model not found. Aborting.")
        return

    files = sorted([os.path.join(BASE_DATA_DIR, f) for f in os.listdir(BASE_DATA_DIR) if f.endswith('.json')])
    if not files: return
    
    print(f"--- Starting UNIFIED MCTS Evaluation ---")
    print(f"ADP Weights: Loaded")
    print(f"RL Model: {RL_MODEL_PATH}")
    
    # Parallel Execution
    # Note: RL inference on CPU can be heavy. Reduce workers if memory/cpu issues occur.
    workers = max(1, os.cpu_count() - 2)
    
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_instance, f, adp_weights, global_statics, RL_MODEL_PATH): f for f in files}
        
        with tqdm(total=len(files), unit="inst") as pbar:
            for future in as_completed(futures):
                print(future.result())
                pbar.update(1)

if __name__ == '__main__':
    run_pipeline()