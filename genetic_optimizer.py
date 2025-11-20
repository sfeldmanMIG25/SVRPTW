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
# Simulation
REVENUE_PER_UNIT_DEMAND = 10.0 
GAMMA = 0.9 

# Genetic Algorithm
POPULATION_SIZE = 50
GENERATIONS = 30
ELITISM_COUNT = 5     
MUTATION_RATE = 0.15  
MUTATION_SIGMA = 0.2  
EPISODES_PER_FITNESS = 5 # Increased slightly for better stability

# Evaluation
EVAL_SIMULATIONS = 100 # Increased target for robustness

# --- PATH CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DATA_DIR = os.path.join(SCRIPT_DIR, 'instances', 'data')
GENETIC_SOLUTIONS_DIR = os.path.join(SCRIPT_DIR, 'solutions', 'Genetic')
GENETIC_RESULTS_DIR = os.path.join(GENETIC_SOLUTIONS_DIR, 'simulation_results')

os.makedirs(GENETIC_SOLUTIONS_DIR, exist_ok=True)
os.makedirs(GENETIC_RESULTS_DIR, exist_ok=True)

# --- FEATURE ENGINEERING ---
def extract_features(state, vehicle_idx, instance, global_statics):
    current_v = state['vehicle_states'][vehicle_idx]
    current_loc_id = current_v['loc']
    
    if current_loc_id == 0:
        curr_coord = (instance['depot']['x'], instance['depot']['y'])
    else:
        cust = instance['cust_map'][current_loc_id]
        curr_coord = (cust['x'], cust['y'])

    # 1. Bias
    f_bias = 1.0
    
    # 2. Demand & 3. Distance & 4. Tightness
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

    # 5. Fleet Availability & 6. Capacity
    available_vehicles = sum(1 for v in state['vehicle_states'] if v['time_avail'] <= state['current_time'])
    total_current_capacity = sum(v['capacity'] for v in state['vehicle_states'])

    # Normalization
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

    def set_instance(self, instance_data):
        self.current_instance = instance_data
        # Check if lookups exist, if not create them
        if 'cust_map' not in self.current_instance:
            self.current_instance['cust_map'] = {c['id']: c for c in self.current_instance['customers']}
            self.current_instance['cust_map'][0] = self.current_instance['depot']
            self.current_instance['cust_coords'] = {c['id']: (c['x'], c['y']) for c in self.current_instance['customers']}
            self.current_instance['cust_coords'][0] = (self.current_instance['depot']['x'], self.current_instance['depot']['y'])

    def get_value(self, features):
        return np.dot(self.weights, features)

    def get_feasible_actions(self, vehicle_idx, state):
        v_state = state['vehicle_states'][vehicle_idx]
        actions = [0]
        curr_loc = self.current_instance['cust_coords'][v_state['loc']]
        curr_time = v_state['time_avail']
        
        for cid in state['unvisited_ids']:
            cust = self.current_instance['cust_map'][cid]
            if cust['demand'] > v_state['capacity']: continue
            
            dist = euclidean_distance(curr_loc, (cust['x'], cust['y']))
            if curr_time + dist <= cust['L']:
                actions.append(cid)
        return actions

    def step(self, vehicle_idx, state, action_id, stochastic=True):
        """Execute one step of simulation."""
        instance = self.current_instance
        v_state = state['vehicle_states'][vehicle_idx]
        curr_loc = instance['cust_coords'][v_state['loc']]
        target_loc = instance['cust_coords'][action_id]
        
        dist = euclidean_distance(curr_loc, target_loc)
        travel_time = StochasticSampler.sample_travel_time(dist) if stochastic else dist
        arrival_time = v_state['time_avail'] + travel_time
        
        wage_min = travel_time
        transit_cost = dist * TRANSIT_COST_PER_MILE
        
        revenue = 0.0
        penalty = 0.0
        service_time = 0.0
        
        next_unvisited = state['unvisited_ids'].copy()
        next_cap = v_state['capacity']
        
        if action_id != 0:
            cust = instance['cust_map'][action_id]
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
                revenue = cust['demand'] * REVENUE_PER_UNIT_DEMAND
                next_cap -= cust['demand']
                next_unvisited.discard(action_id)
            else:
                service_start = arrival_time
                service_time = StochasticSampler.sample_service_time(cust['mean_service_time']) if stochastic else cust['mean_service_time']
                wage_min += service_time
                revenue = cust['demand'] * REVENUE_PER_UNIT_DEMAND
                next_cap -= cust['demand']
                next_unvisited.discard(action_id)
        else:
            service_start = arrival_time

        wage_cost = wage_min * WAGE_COST_PER_MINUTE
        
        # Contribution for Policy (Net Profit)
        contribution = revenue - (transit_cost + wage_cost + penalty)
        
        # Real Cost for Metrics (Operational only)
        real_op_cost = transit_cost + wage_cost + penalty
        
        finish_time = service_start + service_time
        
        # --- OPTIMIZATION: Avoid DeepCopy ---
        # Create shallow copy of vehicle states list
        new_vehicle_states = list(state['vehicle_states'])
        # Create new dict for the specific vehicle (State Update)
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
        
        return contribution, next_state, finish_time, real_op_cost, service_time

    def run_episode(self, stochastic=True):
        """Runs full simulation using current weights."""
        instance = self.current_instance
        state = {
            'current_time': DEPOT_E_TIME,
            'unvisited_ids': set(c['id'] for c in instance['customers']),
            'vehicle_states': [{'loc': 0, 'time_avail': DEPOT_E_TIME, 'capacity': instance['vehicle_capacity']} 
                               for _ in range(instance['num_vehicles'])]
        }
        
        events = [(DEPOT_E_TIME, v) for v in range(instance['num_vehicles'])]
        heapq.heapify(events)
        
        total_profit = 0.0 # Fitness Metric
        total_op_cost = 0.0 # Reporting Metric (Op Cost only)
        total_service_time = 0.0
        hard_late_count = 0
        
        while events:
            time_now, v_idx = heapq.heappop(events)
            if time_now > DEPOT_L_TIME: break
            
            state['current_time'] = time_now
            state['vehicle_states'][v_idx]['time_avail'] = time_now
            
            # DECISION
            feasible = self.get_feasible_actions(v_idx, state)
            best_val = -float('inf')
            best_action = 0
            
            # Policy: Max(Immediate_Contribution + Gamma * VFA(Next_State))
            for action in feasible:
                # Deterministic Lookahead
                contrib_det, next_state_det, _, _, _ = self.step(v_idx, state, action, stochastic=False)
                
                feats = extract_features(next_state_det, v_idx, instance, self.global_statics)
                future_val = self.get_value(feats)
                
                val = contrib_det + (GAMMA * future_val)
                if val > best_val:
                    best_val = val
                    best_action = action
            
            # EXECUTION (Stochastic)
            real_contrib, next_state, finish_time, op_cost, s_time = self.step(v_idx, state, best_action, stochastic=stochastic)
            
            total_profit += real_contrib
            total_op_cost += op_cost
            total_service_time += s_time
            if op_cost >= HARD_LATE_PENALTY:
                 hard_late_count += 1

            state = next_state
            
            if finish_time < DEPOT_L_TIME:
                if best_action != 0:
                    heapq.heappush(events, (finish_time, v_idx))
                elif best_action == 0 and state['unvisited_ids']:
                    next_try = time_now + 30
                    if next_try < DEPOT_L_TIME: heapq.heappush(events, (next_try, v_idx))
        
        # LOGIC:
        # 1. Fitness (Training): Include penalty to guide learning
        unvisited_count = len(state['unvisited_ids'])
        training_penalty = unvisited_count * HARD_LATE_PENALTY
        final_fitness = total_profit - training_penalty
        
        # 2. Reporting: Return stats for aggregator (unvisited count separate from op cost)
        return {
            'fitness': final_fitness,
            'total_cost': total_op_cost, 
            'missed_customers': unvisited_count,
            'hard_late_count': hard_late_count,
            'service_time': total_service_time
        }

# --- GENETIC ALGORITHM LOGIC ---

def evaluate_individual(weights, instances, global_statics):
    """Runs an individual on a set of instances and returns avg fitness."""
    engine = SimulationEngine(global_statics, weights)
    fitness_scores = []
    
    for inst in instances:
        engine.set_instance(inst)
        res = engine.run_episode(stochastic=True)
        fitness_scores.append(res['fitness'])
        
    return np.mean(fitness_scores)

class GeneticAlgorithm:
    def __init__(self, num_features):
        self.num_features = num_features
        self.population = [np.random.uniform(-1, 1, num_features) for _ in range(POPULATION_SIZE)]
        
    def evolve(self, all_instances, global_statics):
        print(f"--- Evolving Policy ({GENERATIONS} Gens, {POPULATION_SIZE} Pop) ---")
        
        # Use all cores
        max_workers = os.cpu_count()
        print(f"Parallelizing with {max_workers} workers.")
        
        for gen in range(GENERATIONS):
            start = time.time()
            batch = random.sample(all_instances, min(len(all_instances), EPISODES_PER_FITNESS))
            
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
            
            # Sort by fitness descending
            sorted_indices = np.argsort(fitnesses)[::-1]
            self.population = [self.population[i] for i in sorted_indices]
            best_fitness = fitnesses[sorted_indices[0]]
            
            print(f"  Gen {gen+1}/{GENERATIONS} | Best Fitness: ${best_fitness:,.2f} | Time: {time.time()-start:.1f}s")
            
            new_pop = []
            new_pop.extend(self.population[:ELITISM_COUNT])
            
            while len(new_pop) < POPULATION_SIZE:
                parent1 = self.population[random.choice(range(POPULATION_SIZE//2))] 
                parent2 = self.population[random.choice(range(POPULATION_SIZE//2))]
                child = (parent1 + parent2) / 2.0
                
                if random.random() < MUTATION_RATE:
                    mutation = np.random.normal(0, MUTATION_SIGMA, self.num_features)
                    child += mutation
                
                new_pop.append(child)
            
            self.population = new_pop
            
        return self.population[0] 

# --- WORKER FOR FINAL EVALUATION ---
def final_eval_worker(instance, weights, global_statics):
    # Worker function to run 100 sims on one instance
    engine = SimulationEngine(global_statics, weights)
    engine.set_instance(instance)
    
    sim_costs = []
    sim_missed = []
    sim_util = []
    
    for _ in range(EVAL_SIMULATIONS):
        res = engine.run_episode(stochastic=True)
        sim_costs.append(res['total_cost'])
        sim_missed.append(res['missed_customers'] + res['hard_late_count'])
        
        capacity = instance['num_vehicles'] * global_statics['time_horizon']
        sim_util.append(res['service_time'] / max(1, capacity))
        
    return {
        'avg_cost': np.mean(sim_costs),
        'avg_fail': np.mean(sim_missed),
        'avg_util': np.mean(sim_util),
        'filename': instance['filename']
    }

# --- MAIN PIPELINE ---
def run_genetic_pipeline():
    # 1. Load Data
    instance_files = sorted([os.path.join(BASE_DATA_DIR, f) for f in os.listdir(BASE_DATA_DIR) if f.endswith('.json')])
    if not instance_files:
        print("No data found.")
        return

    print(f"Loading {len(instance_files)} instances...")
    all_instances = []
    for f in instance_files:
        data = load_instance(f)
        if isinstance(data['customers'], pd.DataFrame):
            data['customers'] = data['customers'].to_dict(orient='records')
        
        # Inject Maps Here for Main Thread
        data['cust_map'] = {c['id']: c for c in data['customers']}
        data['cust_map'][0] = data['depot']
        data['cust_coords'] = {c['id']: (c['x'], c['y']) for c in data['customers']}
        data['cust_coords'][0] = (data['depot']['x'], data['depot']['y'])
        data['filename'] = os.path.basename(f)
        all_instances.append(data)
        
    # 2. Global Statics
    max_demand = max(sum(c['demand'] for c in inst['customers']) for inst in all_instances)
    max_fleet_cap = max(inst['num_vehicles'] * inst['vehicle_capacity'] for inst in all_instances)
    global_statics = {
        'max_global_demand': max_demand,
        'max_possible_dist': euclidean_distance((0,0), (100,100)),
        'max_global_vehicles': max(inst['num_vehicles'] for inst in all_instances),
        'max_global_fleet_capacity': max_fleet_cap,
        'time_horizon': DEPOT_L_TIME - DEPOT_E_TIME
    }

    # 3. Run GA Training
    ga = GeneticAlgorithm(num_features=7)
    best_weights = ga.evolve(all_instances, global_statics)
    
    print("\nBest Weights Found:", best_weights)
    
    # 4. Save Best Policy
    policy_file = os.path.join(GENETIC_SOLUTIONS_DIR, 'genetic_policy_weights.pkl')
    with open(policy_file, 'wb') as f:
        pickle.dump({'weights': best_weights, 'global_statics': global_statics}, f)
    
    # 5. Full Evaluation (Parallelized)
    print(f"\n--- Running Final Evaluation ({EVAL_SIMULATIONS} sims/instance) ---")
    
    summary_metrics = {'cost': [], 'failures': [], 'util': []}
    max_workers = os.cpu_count()
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all instances
        futures = {
            executor.submit(final_eval_worker, inst, best_weights, global_statics): i 
            for i, inst in enumerate(all_instances)
        }
        
        completed = 0
        for future in as_completed(futures):
            res = future.result()
            completed += 1
            
            summary_metrics['cost'].append(res['avg_cost'])
            summary_metrics['failures'].append(res['avg_fail'])
            summary_metrics['util'].append(res['avg_util'])
            
            # Save Result JSON
            result_data = {
                'instance_file': res['filename'],
                'policy_type': 'Genetic_Evolution',
                'metrics': {
                    'mean_total_cost': res['avg_cost'],
                    'mean_missed_customers': res['avg_fail'],
                    'fleet_utilization': res['avg_util']
                }
            }
            
            out_file = os.path.join(GENETIC_RESULTS_DIR, f"{res['filename'].replace('.json', '')}_adp_results.json")
            with open(out_file, 'w') as f:
                json.dump(result_data, f, indent=4)
                
            if completed % 10 == 0:
                print(f"  Processed {completed}/{len(all_instances)}")

    # 6. Summary Report
    print("\n" + "="*40)
    print("GENETIC ALGORITHM SUMMARY")
    print("="*40)
    print(f"Avg Cost:     ${np.mean(summary_metrics['cost']):,.2f}")
    print(f"Avg Failures: {np.mean(summary_metrics['failures']):.2f}")
    print(f"Avg Util:     {np.mean(summary_metrics['util'])*100:.1f}%")
    print("="*40)

if __name__ == "__main__":
    run_genetic_pipeline()