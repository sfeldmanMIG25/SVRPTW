import numpy as np
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import json
import glob
from tqdm import tqdm

# --- IMPORTS ---
from config import (
    WAGE_COST_PER_MINUTE, TRANSIT_COST_PER_MILE, 
    DEPOT_E_TIME, DEPOT_L_TIME, HARD_LATE_PENALTY
)
from simulator import StochasticSampler
from data_generator import euclidean_distance
from deterministic_policy_generator import load_instance
from rollout_agent import RolloutAgent, LightweightEngine # Reuse your single-agent logic

class CooperativeRolloutSolver:
    """
    Implements a Coordinate Descent approach to MMDP.
    Solves the joint action space by serializing agent decisions based on urgency.
    """
    def __init__(self, instance, model_path):
        self.instance = instance
        # We utilize the single-agent planner as a subroutine
        self.sub_agent = RolloutAgent(instance, model_path)
        self.engine = self.sub_agent.engine # Reuse lightweight engine
        
    def get_joint_action(self, vehicles, unvisited_ids):
        """
        Determines actions for ALL vehicles cooperatively.
        
        Strategy:
        1. Identify 'Urgency' (Time Remaining).
        2. Sort vehicles: Most constrained -> Least constrained.
        3. Iterate through sorted vehicles:
           a. Vehicle selects best action given current 'available' map.
           b. 'Lock' that action by removing it from 'available' for subsequent vehicles.
        4. Return map {v_id: action}.
        """
        
        # 1. Sort by Urgency (Time until DEPOT_L_TIME)
        
        # We explicitly copy the state to avoid mutating the real simulation
        working_unvisited = unvisited_ids.copy()
        
        # Sort logic: (DEPOT_L_TIME - v['time']) is time budget. 
        sorted_vehicles = sorted(vehicles, key=lambda v: (DEPOT_L_TIME - v['time']))
        
        joint_actions = {}
        
        for v in sorted_vehicles:
            if v['finished']:
                joint_actions[v['id']] = 0
                continue
                
            # 2. Plan for this agent
            # The 'working_unvisited' set SHRINKS as we iterate.
            # This is the "State Space Reduction" mechanism.
            action = self.sub_agent.select_action(v, working_unvisited)
            
            joint_actions[v['id']] = action
            
            # 3. Lock the Resource (Interaction Handling)
            if action != 0:
                if action in working_unvisited:
                    working_unvisited.remove(action)
                    
        return joint_actions

# --- COOPERATIVE SIMULATION WORKER ---
def run_cooperative_day(solver, day_idx):
    """
    Runs a simulation where ALL vehicles move simultaneously (MMDP transition).
    """
    instance = solver.instance
    num_vehicles = instance['num_vehicles']
    
    # Init State
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
    
    # Global Metrics
    total_cost_accum = 0.0
    total_transit = 0.0
    total_wage = 0.0
    hard_lates_count = 0
    
    # --- JOINT STEP LOOP ---
    while True:
        # Check if ALL finished
        if all(v['finished'] for v in vehicles):
            break
            
        # 1. PLAN JOINT ACTIONS (Cooperative Step)
        # This replaces the single-agent "select_action"
        joint_actions_map = solver.get_joint_action(copy.deepcopy(vehicles), unvisited.copy())
        
        # 2. EXECUTE JOINT ACTIONS (Simultaneous Environment Step)
        # We iterate through vehicles to apply the physics of their assigned actions
        
        step_active = False # Track if anyone actually did something useful
        
        for v in vehicles:
            if v['finished']: continue
            
            action = joint_actions_map.get(v['id'], 0)
            
            # Physics Step (Reusing Logic from rollout_agent for consistency)
            curr_loc = solver.engine.coords[v['loc']]
            target_loc = solver.engine.coords[action]
            dist = euclidean_distance(curr_loc, target_loc)
            
            travel_time = StochasticSampler.sample_travel_time(dist)
            arrival = v['time'] + travel_time
            
            transit_cost = dist * TRANSIT_COST_PER_MILE
            wage_billable = travel_time
            penalty = 0
            wait = 0
            svc = 0
            outcome = 'SUCCESS'
            
            # --- Outcome Logic ---
            if action == 0:
                outcome = 'DEPOT_END'
                v['finished'] = True
                v['loc'] = 0
                v['time'] = arrival
                svc_start = arrival
                dept_time = arrival
            else:
                step_active = True
                cust = solver.engine.cust_map[action]
                
                # Check Late
                if arrival > cust['L']:
                    outcome = 'LATE_SKIP'
                    penalty = HARD_LATE_PENALTY
                    svc_start = arrival
                    dept_time = arrival
                    hard_lates_count += 1
                    # Note: We attempt to remove from unvisited, 
                    # but another agent might have claimed it in this exact timestep
                    # (Though our serial planner prevents this specific collision)
                    if action in unvisited: unvisited.remove(action)
                    
                else:
                    # Early / On Time
                    if arrival < cust['E']:
                        wait = cust['E'] - arrival
                        wage_billable += wait
                        svc_start = cust['E']
                    else:
                        svc_start = arrival
                        
                    svc = StochasticSampler.sample_service_time(cust['mean_service_time'])
                    wage_billable += svc
                    dept_time = svc_start + svc
                    
                    v['cap'] -= cust['demand']
                    if action in unvisited: unvisited.remove(action)
                    
                v['loc'] = action
                v['time'] = dept_time
                
            # Cost Accumulation
            wage_cost = wage_billable * WAGE_COST_PER_MINUTE
            step_cost = transit_cost + wage_cost + penalty
            
            total_cost_accum += step_cost
            total_transit += transit_cost
            total_wage += wage_cost
            
            # Log
            traces[v['id']].append({
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
            
        # Global Time Limit Check (Safety Break)
        # If all active vehicles passed L_TIME, force finish
        if all(v['time'] >= DEPOT_L_TIME for v in vehicles if not v['finished']):
            for v in vehicles: v['finished'] = True

    # End of Day Penalties
    missed_count = len(unvisited)
    total_cost_accum += (missed_count * HARD_LATE_PENALTY)
    
    return {
        'day_index': day_idx,
        'total_cost': total_cost_accum,
        'transit_cost': total_transit,
        'wage_cost': total_wage,
        'hard_lates': hard_lates_count,
        'missed_customers': missed_count,
        'vehicle_traces': traces
    }

# --- WORKER ---
def process_instance(filepath, rl_model_path):
    try:
        instance = load_instance(filepath)
        if not instance: return None
        if isinstance(instance['customers'], list) is False:
            instance['customers'] = instance['customers'].to_dict(orient='records')
            
        # Initialize Cooperative Solver
        solver = CooperativeRolloutSolver(instance, rl_model_path)
        
        logs = []
        # Eval Simulations
        # Note: Reduced to 15 simulations per instance for speed, as MMDP planning is heavier
        for d in range(15): 
            log = run_cooperative_day(solver, d)
            logs.append(log)
            
        return {
            'instance_file': os.path.basename(filepath),
            'policy_type': 'Cooperative_MMDP_Rollout',
            'N': instance['num_customers'],
            'V': instance['num_vehicles'],
            'daily_simulation_logs': logs
        }
    except Exception as e:
        print(f"Error {filepath}: {e}")
        return None

# --- MAIN ---
def run_batch_evaluation():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    instance_dir = os.path.join(script_dir, 'instances', 'data')
    rl_model_path = os.path.join(script_dir, 'solutions', 'RL', 'vrp_dqn.pth')
    results_dir = os.path.join(script_dir, 'solutions', 'Cooperative', 'simulation_results')
    
    os.makedirs(results_dir, exist_ok=True)
    
    files = sorted(glob.glob(os.path.join(instance_dir, '*.json')))
    if not files: return

    print(f"--- Starting Cooperative MMDP Evaluation ---")
    print(f"{'Instance':<35} | {'Total Cost':<12} | {'Missed':<8} | {'H.Late':<8} | {'Transit $':<10} | {'Wage $':<10}")
    print("-" * 95)
    
    # Heavy computation: conservative worker count
    max_workers = max(1, os.cpu_count() // 2)
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_instance, f, rl_model_path): f for f in files}
        
        for future in as_completed(futures):
            res = future.result()
            if res:
                logs = res['daily_simulation_logs']
                mean_total = np.mean([l['total_cost'] for l in logs])
                mean_missed = np.mean([l['missed_customers'] for l in logs])
                mean_hlates = np.mean([l['hard_lates'] for l in logs])
                mean_transit = np.mean([l.get('transit_cost', 0) for l in logs])
                mean_wage = np.mean([l.get('wage_cost', 0) for l in logs])
                
                print(f"{res['instance_file']:<35} | ${mean_total:<11,.0f} | {mean_missed:<8.1f} | {mean_hlates:<8.1f} | ${mean_transit:<9,.0f} | ${mean_wage:<9,.0f}")
                
                out_name = res['instance_file'].replace('.json', '') + '_coop_results.json'
                with open(os.path.join(results_dir, out_name), 'w') as f:
                    json.dump(res, f, indent=4)

    print(f"\nEvaluation Complete. Results saved to {results_dir}")

if __name__ == '__main__':
    run_batch_evaluation()