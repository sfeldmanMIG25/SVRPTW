import torch
import numpy as np
import json
import os
import copy
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import glob

# --- IMPORTS ---
from config import (
    WAGE_COST_PER_MINUTE, TRANSIT_COST_PER_MILE, 
    DEPOT_E_TIME, DEPOT_L_TIME, HARD_LATE_PENALTY
)
from simulator import StochasticSampler
from data_generator import euclidean_distance
from deterministic_policy_generator import load_instance

# Import Network Architecture (Robust Fallback)
try:
    from rl_trainer import VRP_DQN
except ImportError:
    import torch.nn as nn
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

# --- HYPERPARAMETERS ---
NUM_ROLLOUTS = 10       
EVAL_SIMULATIONS = 30   
ROLLOUT_DEPTH = 200     
DEVICE = "cpu"          

# --- LIGHTWEIGHT ENGINE ---
class LightweightEngine:
    def __init__(self, instance):
        self.instance = instance
        self.cust_map = {c['id']: c for c in instance['customers']}
        self.cust_map[0] = instance['depot']
        self.coords = {c['id']: (c['x'], c['y']) for c in instance['customers']}
        self.coords[0] = (instance['depot']['x'], instance['depot']['y'])
        self.max_capacity = instance['vehicle_capacity']

    def get_valid_actions(self, v_state, unvisited_ids):
        curr_loc = self.coords[v_state['loc']]
        actions = []
        
        # 1. Try Customers
        for cid in unvisited_ids:
            c = self.cust_map[cid]
            if c['demand'] > v_state['cap']: continue
            
            # Deterministic Time Window Lookahead
            dist = euclidean_distance(curr_loc, (c['x'], c['y']))
            if v_state['time'] + dist > c['L']: continue
            
            actions.append(cid)
            
        # 2. Depot is always an option
        actions.append(0)
        return actions

    def step(self, v_state, unvisited_ids, action, stochastic=True):
        curr_loc = self.coords[v_state['loc']]
        target_loc = self.coords[action]
        dist = euclidean_distance(curr_loc, target_loc)
        
        if stochastic:
            travel_time = StochasticSampler.sample_travel_time(dist)
        else:
            travel_time = dist 
            
        arrival = v_state['time'] + travel_time
        transit_cost = dist * TRANSIT_COST_PER_MILE
        wage_billable = travel_time
        penalty = 0
        
        next_v = v_state.copy()
        next_unvisited = unvisited_ids.copy()
        
        if action == 0:
            next_v['loc'] = 0
            next_v['time'] = arrival
            return (transit_cost + (wage_billable * WAGE_COST_PER_MINUTE)), next_v, next_unvisited, True
        else:
            cust = self.cust_map[action]
            if arrival > cust['L']:
                penalty = HARD_LATE_PENALTY
                next_v['time'] = arrival
                next_unvisited.discard(action)
            else:
                if arrival < cust['E']:
                    wait = cust['E'] - arrival
                    wage_billable += wait
                    start_svc = cust['E']
                else:
                    start_svc = arrival
                
                if stochastic:
                    svc = StochasticSampler.sample_service_time(cust['mean_service_time'])
                else:
                    svc = cust['mean_service_time']
                    
                wage_billable += svc
                next_v['time'] = start_svc + svc
                next_v['cap'] -= cust['demand']
                next_unvisited.discard(action)
            
            next_v['loc'] = action
            cost = transit_cost + (wage_billable * WAGE_COST_PER_MINUTE) + penalty
            return cost, next_v, next_unvisited, False

# --- ROLLOUT AGENT ---
class RolloutAgent:
    def __init__(self, instance, model_path):
        self.instance = instance
        self.engine = LightweightEngine(instance)
        
        self.model = VRP_DQN().to(DEVICE)
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=DEVICE)
            self.model.load_state_dict(state_dict)
            self.model.eval()
        else:
            print(f"WARNING: RL Model not found at {model_path}. Agent will act randomly.")
            
        self.max_dist = euclidean_distance((0,0), (100,100))
        self.time_horizon = DEPOT_L_TIME - DEPOT_E_TIME
        self.max_demand = max([c['demand'] for c in instance['customers']]) if instance['customers'] else 1

    def select_action(self, vehicle_state, unvisited_ids):
        valid_actions = self.engine.get_valid_actions(vehicle_state, unvisited_ids)
        
        if not valid_actions: return 0
        if len(valid_actions) == 1: return valid_actions[0] 
        
        action_costs = []
        
        for action in valid_actions:
            total_cost = 0.0
            
            for _ in range(NUM_ROLLOUTS):
                # A. Simulate Immediate Step
                step_cost, next_v, next_u, done = self.engine.step(vehicle_state, unvisited_ids, action, stochastic=True)
                sim_cost = step_cost
                
                # B. Simulate Future
                if done:
                    # Penalize remaining customers if we quit early
                    sim_cost += (len(next_u) * HARD_LATE_PENALTY)
                else:
                    sim_cost += self._rollout_simulation(next_v, next_u)
                
                total_cost += sim_cost
            
            avg_cost = total_cost / NUM_ROLLOUTS
            action_costs.append((avg_cost, action))
            
        action_costs.sort(key=lambda x: x[0])
        return action_costs[0][1]

    def _rollout_simulation(self, v_state, unvisited_ids):
        curr_v = v_state.copy()
        curr_u = unvisited_ids.copy()
        acc_cost = 0.0
        depth = 0
        
        while depth < ROLLOUT_DEPTH:
            if not curr_u:
                break

            if curr_v['time'] >= DEPOT_L_TIME: 
                acc_cost += (len(curr_u) * HARD_LATE_PENALTY)
                break
                
            action = self._get_dqn_action(curr_v, curr_u)
            
            step_cost, next_v, next_u, done = self.engine.step(curr_v, curr_u, action, stochastic=True)
            acc_cost += step_cost
            
            if done:
                acc_cost += (len(next_u) * HARD_LATE_PENALTY)
                break
            
            curr_v = next_v
            curr_u = next_u
            depth += 1
            
        return acc_cost

    def _get_dqn_action(self, v_state, unvisited_ids):
        valid = self.engine.get_valid_actions(v_state, unvisited_ids)
        if not valid: return 0
        
        t_norm = (v_state['time'] - DEPOT_E_TIME) / self.time_horizon
        c_norm = v_state['cap'] / self.instance['vehicle_capacity']
        g_tens = torch.tensor([[t_norm, c_norm]], dtype=torch.float, device=DEVICE)
        
        n_feats = np.zeros((200, 6), dtype=np.float32)
        mask = np.zeros(200, dtype=bool)
        
        curr_loc = self.engine.coords[v_state['loc']]
        
        # Depot
        dist_0 = euclidean_distance(curr_loc, self.engine.coords[0])
        n_feats[0] = [dist_0/self.max_dist, 0, 0, 1, 0, 1.0]
        mask[0] = True
        
        # Candidates
        for cid in unvisited_ids:
            if cid == 0: continue
            cust = self.engine.cust_map[cid]
            if cust['demand'] > v_state['cap']: continue
            
            d = euclidean_distance(curr_loc, (cust['x'], cust['y']))
            arr_est = v_state['time'] + d
            if arr_est > cust['L']: continue 
            
            gap = (cust['L'] - arr_est) / 60.0
            n_feats[cid] = [
                d / self.max_dist,
                cust['demand'] / self.max_demand,
                (cust['E'] - DEPOT_E_TIME) / self.time_horizon,
                (cust['L'] - DEPOT_E_TIME) / self.time_horizon,
                0.0, 
                gap
            ]
            mask[cid] = True
            
        n_tens = torch.tensor(n_feats, dtype=torch.float, device=DEVICE).unsqueeze(0)
        
        with torch.no_grad():
            q_values = self.model(g_tens, n_tens).squeeze(0).cpu().numpy()
        
        best_a = 0
        best_q = -float('inf')
        
        for a in valid:
            if mask[a]:
                if q_values[a] > best_q:
                    best_q = q_values[a]
                    best_a = a
                    
        return best_a

# --- EVALUATION WORKER ---
def run_simulation_day(agent, day_idx):
    """
    Simulates a full day using the agent.
    Returns dictionary with full trace and day-level summary metrics.
    """
    instance = agent.instance
    num_vehicles = instance['num_vehicles']
    
    vehicles = []
    traces = []
    for i in range(num_vehicles):
        vehicles.append({
            'id': i, 'loc': 0, 'time': DEPOT_E_TIME, 
            'cap': instance['vehicle_capacity'], 'finished': False
        })
        traces.append([]) 
        traces[i].append({
            'node_id': 0, 'outcome': 'DEPOT_START', 'arrival_time': DEPOT_E_TIME,
            'service_start': DEPOT_E_TIME, 'departure_time': DEPOT_E_TIME,
            'wait_time': 0, 'service_duration': 0, 'transit_cost': 0, 
            'wage_cost': 0, 'penalty_cost': 0, 'dist': 0
        })

    unvisited = set(c['id'] for c in instance['customers'])
    
    # Cost Accumulators for this specific day
    total_cost_accum = 0.0
    total_transit_cost = 0.0
    total_wage_cost = 0.0
    hard_lates_count = 0
    
    while True:
        active_v = None
        min_time = float('inf')
        
        for v in vehicles:
            if not v['finished']:
                if v['time'] < min_time:
                    min_time = v['time']
                    active_v = v
        
        if active_v is None: break 
        
        if active_v['time'] >= DEPOT_L_TIME:
            active_v['finished'] = True
            continue
            
        action = agent.select_action(active_v.copy(), unvisited.copy())
        
        v_idx = active_v['id']
        curr_loc = agent.engine.coords[active_v['loc']]
        target_loc = agent.engine.coords[action]
        dist = euclidean_distance(curr_loc, target_loc)
        
        travel_time = StochasticSampler.sample_travel_time(dist)
        arrival = active_v['time'] + travel_time
        
        transit_cost = dist * TRANSIT_COST_PER_MILE
        wage_billable = travel_time
        penalty = 0
        wait = 0
        svc = 0
        outcome = 'SUCCESS'
        
        if action == 0:
            outcome = 'DEPOT_END'
            active_v['finished'] = True
            active_v['loc'] = 0
            active_v['time'] = arrival
            svc_start = arrival
            dept_time = arrival
        else:
            cust = agent.engine.cust_map[action]
            if arrival > cust['L']:
                outcome = 'LATE_SKIP'
                penalty = HARD_LATE_PENALTY
                svc_start = arrival
                dept_time = arrival
                hard_lates_count += 1
                unvisited.discard(action)
            else:
                if arrival < cust['E']:
                    wait = cust['E'] - arrival
                    wage_billable += wait
                    svc_start = cust['E']
                else:
                    svc_start = arrival
                
                svc = StochasticSampler.sample_service_time(cust['mean_service_time'])
                wage_billable += svc
                dept_time = svc_start + svc
                active_v['cap'] -= cust['demand']
                unvisited.discard(action)
                
            active_v['loc'] = action
            active_v['time'] = dept_time

        wage_cost = wage_billable * WAGE_COST_PER_MINUTE
        step_total_cost = transit_cost + wage_cost + penalty
        
        # Accumulate
        total_cost_accum += step_total_cost
        total_transit_cost += transit_cost
        total_wage_cost += wage_cost
        
        traces[v_idx].append({
            'node_id': action,
            'outcome': outcome,
            'arrival_time': round(arrival, 2),
            'service_start': round(svc_start, 2),
            'departure_time': round(dept_time, 2),
            'wait_time': round(wait, 2),
            'service_duration': round(svc, 2),
            'transit_cost': round(transit_cost, 2),
            'wage_cost': round(wage_cost, 2),
            'penalty_cost': round(penalty, 2),
            'dist': round(dist, 2)
        })

    missed_count = len(unvisited)
    missed_penalty = missed_count * HARD_LATE_PENALTY
    total_cost_accum += missed_penalty

    return {
        'day_index': day_idx,
        'total_cost': total_cost_accum,
        'transit_cost': total_transit_cost,
        'wage_cost': total_wage_cost,
        'hard_lates': hard_lates_count,
        'missed_customers': missed_count,
        'vehicle_traces': traces
    }

def process_instance(filepath, rl_model_path):
    try:
        instance = load_instance(filepath)
        if not instance: return None
        if isinstance(instance['customers'], list) is False:
            instance['customers'] = instance['customers'].to_dict(orient='records')
            
        agent = RolloutAgent(instance, rl_model_path)
        
        logs = []
        for d in range(EVAL_SIMULATIONS):
            log = run_simulation_day(agent, d)
            logs.append(log)
            
        return {
            'instance_file': os.path.basename(filepath),
            'policy_type': f'Rollout_Policy_N{NUM_ROLLOUTS}',
            'N': instance['num_customers'],
            'V': instance['num_vehicles'],
            'daily_simulation_logs': logs
        }
    except Exception as e:
        print(f"Error {filepath}: {e}")
        return None

def run_batch_evaluation():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    instance_dir = os.path.join(script_dir, 'instances', 'data')
    rl_model_path = os.path.join(script_dir, 'solutions', 'RL', 'vrp_dqn.pth')
    results_dir = os.path.join(script_dir, 'solutions', 'Rollout', 'simulation_results')
    os.makedirs(results_dir, exist_ok=True)
    
    files = sorted(glob.glob(os.path.join(instance_dir, '*.json')))
    if not files: return

    print(f"--- Starting Rollout Agent Evaluation (K={NUM_ROLLOUTS}) ---")
    print(f"{'Instance':<35} | {'Total Cost':<12} | {'Missed':<8} | {'H.Late':<8} | {'Transit $':<10} | {'Wage $':<10}")
    print("-" * 95)
    
    max_workers = max(1, os.cpu_count() // 2)
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_instance, f, rl_model_path): f for f in files}
        
        for future in as_completed(futures):
            res = future.result()
            if res:
                # Calculate Means
                logs = res['daily_simulation_logs']
                mean_total = np.mean([l['total_cost'] for l in logs])
                mean_missed = np.mean([l['missed_customers'] for l in logs])
                mean_hlates = np.mean([l['hard_lates'] for l in logs])
                mean_transit = np.mean([l.get('transit_cost', 0) for l in logs])
                mean_wage = np.mean([l.get('wage_cost', 0) for l in logs])
                
                # Print Report
                print(f"{res['instance_file']:<35} | ${mean_total:<11,.0f} | {mean_missed:<8.1f} | {mean_hlates:<8.1f} | ${mean_transit:<9,.0f} | ${mean_wage:<9,.0f}")
                
                # Save JSON
                out_name = res['instance_file'].replace('.json', '') + '_rollout_results.json'
                with open(os.path.join(results_dir, out_name), 'w') as f:
                    json.dump(res, f, indent=4)

    print(f"\nEvaluation Complete. Results saved to {results_dir}")

if __name__ == '__main__':
    run_batch_evaluation()