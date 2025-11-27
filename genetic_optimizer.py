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
    COORDINATE_BOUNDS, SERVICE_TIME_BASE_MEAN
)
from simulator import StochasticSampler
from deterministic_policy_generator import load_instance
from data_generator import euclidean_distance

# --- HYPERPARAMETERS ---
REVENUE_PER_UNIT_DEMAND = 10.0 
GAMMA = 0.9 

# Genetic Algorithm
POPULATION_SIZE = 50
GENERATIONS = 30
ELITISM_COUNT = 5     
MUTATION_RATE = 0.15  
MUTATION_SIGMA = 0.2  
# Using a random subset of instances per generation for speed
TRAIN_BATCH_SIZE = 20 

# Evaluation
EVAL_SIMULATIONS = 30 

# --- PATH CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DATA_DIR = os.path.join(SCRIPT_DIR, 'instances', 'data')
GENETIC_SOLUTIONS_DIR = os.path.join(SCRIPT_DIR, 'solutions', 'Genetic')
GENETIC_RESULTS_DIR = os.path.join(GENETIC_SOLUTIONS_DIR, 'simulation_results')
GENETIC_STRATEGY_DIR = os.path.join(GENETIC_SOLUTIONS_DIR, 'strategy')

os.makedirs(GENETIC_SOLUTIONS_DIR, exist_ok=True)
os.makedirs(GENETIC_RESULTS_DIR, exist_ok=True)
os.makedirs(GENETIC_STRATEGY_DIR, exist_ok=True)

# --- FEATURE NAMES ---
FEATURE_NAMES = [
    "Bias (Intercept)",
    "Norm Demand (Unvisited)",
    "Norm Distance (Closest)",
    "Norm Time Left",
    "Norm Time Window Tightness",
    "Norm Fleet Availability",
    "Norm Fleet Capacity"
]

# --- FEATURE ENGINEERING ---
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

# --- SIMULATION ENGINE ---
class SimulationEngine:
    def __init__(self, global_statics, weights):
        self.global_statics = global_statics
        self.weights = np.array(weights)
        self.current_instance = None
        self.action_cache = {} # Optimization

    def set_instance(self, instance_data):
        self.current_instance = instance_data
        if 'cust_map' not in self.current_instance:
            self.current_instance['cust_map'] = {c['id']: c for c in self.current_instance['customers']}
            self.current_instance['cust_map'][0] = self.current_instance['depot']
            self.current_instance['cust_coords'] = {c['id']: (c['x'], c['y']) for c in self.current_instance['customers']}
            self.current_instance['cust_coords'][0] = (self.current_instance['depot']['x'], self.current_instance['depot']['y'])
        self.action_cache = {} 

    def get_value(self, features):
        return np.dot(self.weights, features)

    def _get_state_key(self, vehicle_idx, state):
        # Cache key for feasible actions
        v_state = state['vehicle_states'][vehicle_idx]
        v_tuple = (v_state['loc'], round(v_state['time_avail'], 1))
        unvisited_mask = 0
        for cid in state['unvisited_ids']: unvisited_mask |= (1 << cid)
        return (vehicle_idx, v_tuple, unvisited_mask)

    def get_feasible_actions(self, vehicle_idx, state):
        # Check cache
        key = self._get_state_key(vehicle_idx, state)
        if key in self.action_cache: return self.action_cache[key]

        v_state = state['vehicle_states'][vehicle_idx]
        actions = [0]
        curr_loc = self.current_instance['cust_coords'][v_state['loc']]
        curr_time = v_state['time_avail']
        
        for cid in state['unvisited_ids']:
            cust = self.current_instance['cust_map'][cid]
            if cust['demand'] > v_state['capacity']: continue
            
            dist = euclidean_distance(curr_loc, (cust['x'], cust['y']))
            # Deterministic Check
            if curr_time + dist <= cust['L']:
                actions.append(cid)
        
        self.action_cache[key] = tuple(actions)
        return actions

    def step(self, vehicle_idx, state, action_id, stochastic=True):
        """Execute one step of simulation (Aligned with Simulator.py logic)."""
        instance = self.current_instance
        v_state = state['vehicle_states'][vehicle_idx]
        curr_loc = instance['cust_coords'][v_state['loc']]
        target_loc = instance['cust_coords'][action_id]
        
        dist = euclidean_distance(curr_loc, target_loc)
        travel_time = StochasticSampler.sample_travel_time(dist) if stochastic else dist
        arrival_time = v_state['time_avail'] + travel_time
        
        wage_billable = travel_time
        transit_cost = dist * TRANSIT_COST_PER_MILE
        
        revenue = 0.0
        penalty = 0.0
        service_time = 0.0
        wait_time = 0.0
        outcome = 'SUCCESS'
        
        next_unvisited = state['unvisited_ids'].copy()
        next_cap = v_state['capacity']
        
        if action_id == 0:
            outcome = 'DEPOT_END'
            service_start = arrival_time
            # Unpaid wait at depot
        else:
            cust = instance['cust_map'][action_id]
            if arrival_time > cust['L']:
                outcome = 'LATE_SKIP'
                penalty = HARD_LATE_PENALTY
                next_unvisited.discard(action_id)
                service_start = arrival_time
            elif arrival_time < cust['E']:
                wait_time = cust['E'] - arrival_time
                service_start = cust['E']
                
                # Paid Wait? Yes, unless 0->0 (which is impossible here since action!=0)
                wage_billable += wait_time
                
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
        
        # Shallow Copy Update
        new_vehicle_states = list(state['vehicle_states'])
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
            'penalty_cost': float(penalty),
            'dist': float(dist)
        }
        
        return contribution, next_state, finish_time, log_data

    def run_episode(self, stochastic=True):
        instance = self.current_instance
        state = {
            'current_time': DEPOT_E_TIME,
            'unvisited_ids': set(c['id'] for c in instance['customers']),
            'vehicle_states': [{'loc': 0, 'time_avail': DEPOT_E_TIME, 'capacity': instance['vehicle_capacity']} 
                               for _ in range(instance['num_vehicles'])]
        }
        
        # Trace Init
        vehicle_traces = []
        for v in range(instance['num_vehicles']):
            vehicle_traces.append([{
                'node_id': 0, 'outcome': 'DEPOT_START', 
                'arrival_time': DEPOT_E_TIME, 'service_start': DEPOT_E_TIME, 
                'departure_time': DEPOT_E_TIME, 'wait_time': 0, 'service_duration': 0,
                'transit_cost': 0, 'wage_cost': 0, 'penalty_cost': 0, 'dist': 0
            }])
        
        events = [(DEPOT_E_TIME, v) for v in range(instance['num_vehicles'])]
        heapq.heapify(events)
        
        # Metrics
        total_profit = 0.0 
        total_cost = 0.0
        hard_lates = 0
        
        while events:
            time_now, v_idx = heapq.heappop(events)
            if time_now > DEPOT_L_TIME: break
            
            state['current_time'] = time_now
            state['vehicle_states'][v_idx]['time_avail'] = time_now
            
            feasible = self.get_feasible_actions(v_idx, state)
            best_val = -float('inf')
            best_action = 0
            
            # Policy Decision
            for action in feasible:
                # Lookahead (Deterministic)
                contrib_det, next_state_det, _, _ = self.step(v_idx, state, action, stochastic=False)
                feats = extract_features(next_state_det, v_idx, instance, self.global_statics)
                val = contrib_det + (GAMMA * self.get_value(feats))
                if val > best_val:
                    best_val = val
                    best_action = action
            
            # Execution (Stochastic)
            real_contrib, next_state, finish_time, log = self.step(v_idx, state, best_action, stochastic=stochastic)
            
            total_profit += real_contrib
            step_cost = log['transit_cost'] + log['wage_cost'] + log['penalty_cost']
            total_cost += step_cost
            if log['outcome'] == 'LATE_SKIP': hard_lates += 1
            
            vehicle_traces[v_idx].append(log)
            state = next_state
            
            if finish_time < DEPOT_L_TIME:
                if best_action != 0:
                    heapq.heappush(events, (finish_time, v_idx))
                elif best_action == 0 and state['unvisited_ids']:
                    next_try = time_now + 30
                    if next_try < DEPOT_L_TIME: heapq.heappush(events, (next_try, v_idx))
        
        missed_count = len(state['unvisited_ids'])
        
        # Penalty for Training Guidance
        fitness = total_profit - (missed_count * HARD_LATE_PENALTY)
        
        return {
            'fitness': fitness,
            'total_cost': total_cost,
            'hard_lates': hard_lates,
            'missed_customers': missed_count,
            'vehicle_traces': vehicle_traces
        }

# --- GENETIC ALGORITHM ---

def evaluate_individual(weights, instances, global_statics):
    """Runs individual on batch of instances."""
    engine = SimulationEngine(global_statics, weights)
    scores = []
    for inst in instances:
        engine.set_instance(inst)
        # Run 1 sim per instance for speed during training
        res = engine.run_episode(stochastic=True)
        scores.append(res['fitness'])
    return np.mean(scores)

class GeneticAlgorithm:
    def __init__(self, num_features):
        self.num_features = num_features
        self.population = [np.random.uniform(-1, 1, num_features) for _ in range(POPULATION_SIZE)]
        
    def evolve(self, all_instances, global_statics):
        print(f"--- Evolving Policy ({GENERATIONS} Gens, {POPULATION_SIZE} Pop) ---")
        max_workers = os.cpu_count()
        
        for gen in range(GENERATIONS):
            start = time.time()
            # Stochastic Batch
            batch = random.sample(all_instances, min(len(all_instances), TRAIN_BATCH_SIZE))
            
            fitnesses = []
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(evaluate_individual, ind, batch, global_statics): i 
                    for i, ind in enumerate(self.population)
                }
                results = [None] * len(self.population)
                for future in as_completed(futures):
                    idx = futures[future]
                    results[idx] = future.result()
                fitnesses = results
            
            # Select
            sorted_indices = np.argsort(fitnesses)[::-1]
            self.population = [self.population[i] for i in sorted_indices]
            best_fit = fitnesses[sorted_indices[0]]
            
            print(f"  Gen {gen+1}/{GENERATIONS} | Best Fitness: ${best_fit:,.0f} | Time: {time.time()-start:.1f}s")
            
            # Breed
            new_pop = self.population[:ELITISM_COUNT]
            while len(new_pop) < POPULATION_SIZE:
                p1 = self.population[random.randint(0, POPULATION_SIZE//2)]
                p2 = self.population[random.randint(0, POPULATION_SIZE//2)]
                child = (p1 + p2) / 2.0
                if random.random() < MUTATION_RATE:
                    child += np.random.normal(0, MUTATION_SIGMA, self.num_features)
                new_pop.append(child)
            self.population = new_pop
            
        return self.population[0]

# --- WORKER FOR FINAL EVALUATION ---
def final_eval_worker(instance, weights, global_statics):
    # Run 30 days per instance
    engine = SimulationEngine(global_statics, weights)
    engine.set_instance(instance)
    
    daily_logs = []
    
    for i in range(EVAL_SIMULATIONS):
        res = engine.run_episode(stochastic=True)
        # Extract logs for JSON
        daily_logs.append({
            'day_index': i,
            'total_cost': res['total_cost'],
            'hard_lates': res['hard_lates'],
            'missed_customers': res['missed_customers'],
            'vehicle_traces': res['vehicle_traces']
        })
        
    # Save Result
    res_data = {
        'instance_file': instance['filename'],
        'policy_type': 'Genetic_Evolution',
        'N': instance['num_customers'],
        'V': instance['num_vehicles'],
        'daily_simulation_logs': daily_logs
    }
    
    base_name = instance['filename'].replace('.json', '')
    out_file = os.path.join(GENETIC_RESULTS_DIR, f"{base_name}_genetic_results.json")
    with open(out_file, 'w') as f:
        json.dump(res_data, f, indent=4)
        
    avg_cost = np.mean([l['total_cost'] for l in daily_logs])
    return avg_cost

# --- MAIN PIPELINE ---
def run_genetic_pipeline():
    # 1. Load Data
    files = sorted([f for f in os.listdir(BASE_DATA_DIR) if f.endswith('.json')])
    if not files: return

    print(f"Loading {len(files)} instances...")
    all_instances = []
    for f in files:
        path = os.path.join(BASE_DATA_DIR, f)
        data = load_instance(path)
        if isinstance(data['customers'], pd.DataFrame):
            data['customers'] = data['customers'].to_dict(orient='records')
        
        # Pre-compute lookups
        data['cust_map'] = {c['id']: c for c in data['customers']}
        data['cust_map'][0] = data['depot']
        data['cust_coords'] = {c['id']: (c['x'], c['y']) for c in data['customers']}
        data['cust_coords'][0] = (data['depot']['x'], data['depot']['y'])
        data['filename'] = f
        all_instances.append(data)
        
    # 2. Statics
    max_demand = max(sum(c['demand'] for c in inst['customers']) for inst in all_instances)
    max_vehicles = max(inst['num_vehicles'] for inst in all_instances)
    max_fleet_cap = max(inst['num_vehicles'] * inst['vehicle_capacity'] for inst in all_instances)
    
    global_statics = {
        'max_global_demand': max_demand,
        'max_possible_dist': euclidean_distance((0,0), (100,100)),
        'max_global_vehicles': max_vehicles,
        'max_global_fleet_capacity': max_fleet_cap,
        'time_horizon': DEPOT_L_TIME - DEPOT_E_TIME
    }

    # 3. Train
    ga = GeneticAlgorithm(num_features=7)
    best_weights = ga.evolve(all_instances, global_statics)
    
    # 4. Save Strategy
    with open(os.path.join(GENETIC_STRATEGY_DIR, 'genetic_weights.pkl'), 'wb') as f:
        pickle.dump({'weights': best_weights, 'global_statics': global_statics}, f)
        
    readable = {'feature_names': FEATURE_NAMES, 'weights': best_weights.tolist()}
    with open(os.path.join(GENETIC_STRATEGY_DIR, 'genetic_weights.json'), 'w') as f:
        json.dump(readable, f, indent=4)
        
    print("\nBest Weights Saved.")
    for n, w in zip(FEATURE_NAMES, best_weights):
        print(f"  {n:<30}: {w:.4f}")

    # 5. Evaluate (Parallel)
    print(f"\n--- Final Evaluation ({EVAL_SIMULATIONS} runs/instance) ---")
    
    costs = []
    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(final_eval_worker, inst, best_weights, global_statics): i 
            for i, inst in enumerate(all_instances)
        }
        
        completed = 0
        for future in as_completed(futures):
            res = future.result()
            costs.append(res)
            completed += 1
            if completed % 10 == 0:
                print(f"  [{completed}/{len(all_instances)}] Avg Cost: ${res:.0f}")
                
    print(f"\nOverall Avg Cost: ${np.mean(costs):,.0f}")
    print(f"Results saved to: {GENETIC_RESULTS_DIR}")

if __name__ == "__main__":
    run_genetic_pipeline()