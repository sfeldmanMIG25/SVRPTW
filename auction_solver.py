import numpy as np
import numba as nb
import json
import os
import glob
import heapq
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple, Set, Optional, Deque
from dataclasses import dataclass, field
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
from functools import partial

# --- IMPORTS ---
from config import (
    WAGE_COST_PER_MINUTE, TRANSIT_COST_PER_MILE, 
    DEPOT_E_TIME, DEPOT_L_TIME, HARD_LATE_PENALTY,
    SERVICE_TIME_BASE_MEAN
)
from simulator import StochasticSampler
from data_generator import euclidean_distance
from deterministic_policy_generator import load_instance

# ============================================================================
# OPTIMIZED NUMBA FUNCTIONS
# ============================================================================

@nb.njit(cache=True)
def fast_euclidean(p1: np.ndarray, p2: np.ndarray) -> float:
    """Vectorized Euclidean distance."""
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return np.sqrt(dx * dx + dy * dy)

@nb.njit(cache=True, parallel=True)
def compute_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """Parallelized Distance Matrix Calculation."""
    n = len(coords)
    dist_matrix = np.zeros((n, n))
    for i in nb.prange(n):
        for j in range(i + 1, n):
            d = fast_euclidean(coords[i], coords[j])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d
    return dist_matrix

# ============================================================================
# IMPROVED BAYESIAN ESTIMATOR (Piecewise Linear Risk)
# ============================================================================

class DualBayesianEstimator:
    def __init__(self, config=None):
        # Defaults if config not provided
        cfg = config or {}
        self.travel_alpha = cfg.get('travel_alpha', 2.0)
        self.travel_beta = cfg.get('travel_beta', 2.0)
        
        # Risk Configs
        self.risk_high = cfg.get('risk_buffer_high', 60.0)
        self.risk_mid = cfg.get('risk_buffer_mid', 30.0)
        self.risk_low = cfg.get('risk_buffer_low', 15.0)
        self.ramp_scale = cfg.get('risk_ramp_steepness', 1.0)
        
        # Keep existing service init
        self.service_alpha = 2.0
        self.service_beta = 2.0
        self.travel_risk_factor = 1.0
        
    def update_travel(self, predicted_dist: float, actual_time: float):
        """Update travel time estimator."""
        predicted_time = predicted_dist # Base assumption 1:1
        self.travel_alpha += 1
        error = max(0, actual_time - predicted_time)
        self.travel_beta += error / 5.0
        
        # Adjust risk factor based on systematic error
        if error > 5.0:
            self.travel_risk_factor = min(2.0, self.travel_risk_factor * 1.1)
        elif error < -2.0:
            self.travel_risk_factor = max(0.8, self.travel_risk_factor * 0.95)
            
    def update_service(self, predicted_mean: float, actual_time: float):
        """Update service time estimator."""
        self.service_alpha += 1
        error = max(0, actual_time - predicted_mean)
        self.service_beta += error / 5.0
        
    def get_travel_risk(self, distance: float, time_buffer: float) -> float:
        if time_buffer <= 0: return 100.0
        
        lambda_travel = self.travel_alpha / self.travel_beta
        base_risk = distance * lambda_travel / 100.0
        
        # Parameterized Piecewise Logic
        if time_buffer >= self.risk_high:
            buffer_mult = 1.0
        elif time_buffer >= self.risk_mid:
            # Linear ramp 1.0 -> 1.5
            ratio = (self.risk_high - time_buffer) / (self.risk_high - self.risk_mid)
            buffer_mult = 1.0 + (0.5 * self.ramp_scale) * ratio
        elif time_buffer >= self.risk_low:
             # Linear ramp 1.5 -> 3.0
            ratio = (self.risk_mid - time_buffer) / (self.risk_mid - self.risk_low)
            buffer_mult = 1.5 + (1.5 * self.ramp_scale) * ratio
        else:
            # Linear ramp 3.0 -> 6.0
            ratio = (self.risk_low - time_buffer) / self.risk_low
            buffer_mult = 3.0 + (3.0 * self.ramp_scale) * ratio
            
        return base_risk * buffer_mult * self.travel_risk_factor

# ============================================================================
# DYNAMIC ROUTE BIDDER
# ============================================================================

class OptimizedRouteBidder:
    """
    Bidder with Dynamic Pruning and Route Optimization.
    """
    def __init__(self, vehicle_id: int, instance_data: Dict, 
                 dist_matrix: np.ndarray, estimator: DualBayesianEstimator,
                 pruning_threshold: float):
        self.id = vehicle_id
        self.coords = instance_data['coords']
        self.cust_map = instance_data['customers']
        self.max_capacity = instance_data['vehicle_capacity']
        self.dist_matrix = dist_matrix
        self.estimator = estimator
        self.pruning_threshold = pruning_threshold
        
        # State
        self.current_loc = 0
        self.current_time = DEPOT_E_TIME
        self.remaining_capacity = self.max_capacity
        
        # Plan
        self.route: List[int] = []
        self.route_demand = 0
        self.route_cost_cache: Optional[float] = None
        
    def calculate_insertion_cost(self, customer_id: int) -> Optional[float]:
        """Calculates Marginal Cost of inserting customer into route."""
        cust = self.cust_map[customer_id]
        
        # 1. Capacity Check
        if self.route_demand + cust['demand'] > self.max_capacity:
            return None
            
        # 2. Dynamic Distance Pruning
        # Only consider insertion if close to current location OR close to Depot (start/end)
        # This prevents scanning completely irrelevant vehicles
        dist_from_curr = self.dist_matrix[self.current_loc, customer_id]
        if dist_from_curr > self.pruning_threshold:
             return None
             
        # 3. Cheapest Insertion
        best_cost = float('inf')
        
        # Cache Base Cost
        if self.route_cost_cache is None:
            self.route_cost_cache = self._evaluate_route(self.route)
            if self.route_cost_cache == float('inf'): return None
            
        # Try all positions
        for insert_idx in range(len(self.route) + 1):
            test_route = self.route.copy()
            test_route.insert(insert_idx, customer_id)
            
            cost = self._evaluate_route(test_route)
            if cost < float('inf'):
                marginal = cost - self.route_cost_cache
                if marginal < best_cost:
                    best_cost = marginal
                    
        return best_cost if best_cost < float('inf') else None
        
    def _evaluate_route(self, route: List[int]) -> float:
        """Deterministic evaluation of route cost + risk."""
        if not route:
            dist_home = self.dist_matrix[self.current_loc, 0]
            return dist_home * TRANSIT_COST_PER_MILE
            
        t = self.current_time
        total_cost = 0.0
        prev = self.current_loc
        
        for cid in route:
            cust = self.cust_map[cid]
            
            dist = self.dist_matrix[prev, cid]
            arrival = t + dist
            
            # Feasibility
            if arrival > cust['L']: return float('inf')
            
            wait = max(0, cust['E'] - arrival)
            service_end = max(arrival, cust['E']) + cust['mean_service_time']
            
            transit = dist * TRANSIT_COST_PER_MILE
            wages = (dist + wait + cust['mean_service_time']) * WAGE_COST_PER_MINUTE
            
            # Risk
            time_buffer = cust['L'] - arrival
            risk = self.estimator.get_travel_risk(dist, time_buffer)
            
            total_cost += (transit + wages) * (1.0 + risk)
            
            t = service_end
            prev = cid
            
        # Return to Depot
        dist_home = self.dist_matrix[prev, 0]
        if t + dist_home > DEPOT_L_TIME: return float('inf')
        
        total_cost += dist_home * TRANSIT_COST_PER_MILE
        return total_cost
        
    def insert_customer(self, customer_id: int) -> bool:
        """Commit insertion to best position."""
        best_idx = -1
        best_cost = float('inf')
        
        for insert_idx in range(len(self.route) + 1):
            test_route = self.route.copy()
            test_route.insert(insert_idx, customer_id)
            cost = self._evaluate_route(test_route)
            if cost < best_cost:
                best_cost = cost
                best_idx = insert_idx
                
        if best_idx != -1:
            self.route.insert(best_idx, customer_id)
            self.route_demand += self.cust_map[customer_id]['demand']
            self.route_cost_cache = best_cost
            return True
        return False

# ============================================================================
# HYBRID ASSIGNER (Iterative Hungarian + Greedy)
# ============================================================================

class HybridBatchAssigner:
    def __init__(self, config=None):
        cfg = config or {}
        self.hungarian_rounds = cfg.get('hungarian_rounds', 4)
        self.hungarian_batch_size = cfg.get('hungarian_batch_size', 30)
        
    def assign(self, bidders: List[OptimizedRouteBidder], 
               customer_ids: List[int]) -> List[Tuple[int, int]]:
        
        assignments = [] # List of (vid, cid)
        remaining = set(customer_ids)
        
        # PHASE 1: Iterative Hungarian (The Backbone)
        # We run several rounds to build high-quality route skeletons
        for _ in range(self.hungarian_rounds):
            if not remaining: break
            
            # Take a batch
            current_batch = list(remaining)[:self.hungarian_batch_size]
            
            # Build Cost Matrix
            n_v = len(bidders)
            n_c = len(current_batch)
            cost_matrix = np.full((n_v, n_c), 1e9)
            
            has_feasible = False
            for i, bidder in enumerate(bidders):
                # Simple Load Balancing via Capacity/Cost
                # Heavily loaded vehicles naturally have higher insertion costs
                for j, cid in enumerate(current_batch):
                    cost = bidder.calculate_insertion_cost(cid)
                    if cost is not None:
                        cost_matrix[i, j] = cost
                        has_feasible = True
            
            if not has_feasible:
                break
                
            # Solve
            cost_matrix[np.isinf(cost_matrix)] = 1e12
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            round_assignments = []
            
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < 1e10:
                    vid = r
                    cid = current_batch[c]
                    round_assignments.append((vid, cid))
            
            # Commit
            made_progress = False
            for vid, cid in round_assignments:
                if bidders[vid].insert_customer(cid):
                    assignments.append((vid, cid))
                    remaining.remove(cid)
                    made_progress = True
                    
            if not made_progress: break

        # PHASE 2: Greedy Insertion (The Cleanup)
        # Try to insert remaining customers into ANY vehicle
        if remaining:
            # Sort by urgency (L time)
            sorted_remaining = sorted(list(remaining), 
                                    key=lambda c: bidders[0].cust_map[c]['L'])
            
            for cid in sorted_remaining:
                best_vid = -1
                best_cost = float('inf')
                
                for i, bidder in enumerate(bidders):
                    cost = bidder.calculate_insertion_cost(cid)
                    if cost is not None and cost < best_cost:
                        best_cost = cost
                        best_vid = i
                        
                if best_vid != -1:
                    if bidders[best_vid].insert_customer(cid):
                        assignments.append((best_vid, cid))
        
        return assignments

# ============================================================================
# ROBUST CENTRALIZED CONTROLLER
# ============================================================================

class RobustCentralizedController:
    def __init__(self, instance_data: Dict, config: Dict = None):
        self.instance = instance_data
        self.config = config or {}  # Store config
        
        self._preprocess_data()
        self.dist_matrix = compute_distance_matrix(self.coords_array)
        self.pruning_threshold = self._calculate_pruning_threshold()
        
        # Pass config to components
        self.estimators = [DualBayesianEstimator(self.config) for _ in range(self.instance['num_vehicles'])]
        
        self.bidders = [
            OptimizedRouteBidder(i, self.instance_data, self.dist_matrix, 
                               self.estimators[i], self.pruning_threshold)
            for i in range(self.instance['num_vehicles'])
        ]
        
        self.assigner = HybridBatchAssigner(self.config)
        
        # State
        self.metrics = defaultdict(float)
        self.traces = [[] for _ in range(self.instance['num_vehicles'])]
        self.event_queue = [] 
        
    def _preprocess_data(self):
        custs = self.instance['customers']
        depot = self.instance['depot']
        
        # Handle gaps in IDs by sizing array to max_id + 1
        max_id = max(c['id'] for c in custs) + 1
        self.coords_array = np.zeros((max_id, 2))
        self.coords_array[0] = [depot['x'], depot['y']]
        
        self.cust_map = {}
        for c in custs:
            cid = c['id']
            self.coords_array[cid] = [c['x'], c['y']]
            self.cust_map[cid] = c
            
        self.instance_data = {
            'coords': self.coords_array,
            'customers': self.cust_map,
            'vehicle_capacity': self.instance['vehicle_capacity']
        }
        
    def _calculate_pruning_threshold(self):
        """Dynamic threshold based on average distance."""
        # Sample random pairs to estimate scale
        sample_size = min(100, len(self.instance['customers']))
        dists = []
        for _ in range(sample_size):
            i = np.random.randint(0, len(self.coords_array))
            j = np.random.randint(0, len(self.coords_array))
            if i != j:
                dists.append(fast_euclidean(self.coords_array[i], self.coords_array[j]))
        
        avg_dist = np.mean(dists) if dists else 50.0
        # Allow searching within 2.5x average distance
        multiplier = self.config.get('pruning_multiplier', 2.5)
        return max(50.0, avg_dist * multiplier)

    def run_day(self, day_idx: int) -> Dict:
        self._reset_day_state()
        
        # Initial Hybrid Assignment
        all_cids = list(self.cust_map.keys())
        self.assigner.assign(self.bidders, all_cids)
        
        # Schedule Initials
        for bidder in self.bidders:
            self._schedule_next(bidder.id, DEPOT_E_TIME)
            
        # Sim Loop
        while self.event_queue:
            time, vid, type, cid, planned_dist = heapq.heappop(self.event_queue)
            
            if type == 'ARRIVAL':
                self._process_arrival(vid, cid, time, planned_dist)
            elif type == 'DEPOT_ARRIVAL':
                self._process_depot_arrival(vid, time)
                
        return self._collect_results(day_idx)
        
    def _process_arrival(self, vid, cid, arrival_time, planned_dist):
        bidder = self.bidders[vid]
        cust = self.cust_map[cid]
        
        # Wage Calculation [Fix: Correct Math]
        actual_travel = arrival_time - bidder.current_time
        
        # Update Estimator
        bidder.estimator.update_travel(planned_dist, actual_travel)
        
        # Outcome Logic
        if arrival_time > cust['L']:
            # LATE SKIP
            self.metrics['hard_lates'] += 1
            self.metrics['penalty_cost'] += HARD_LATE_PENALTY
            
            # Log costs incurred to get here
            wage = actual_travel * WAGE_COST_PER_MINUTE
            transit = planned_dist * TRANSIT_COST_PER_MILE
            self.metrics['wage_cost'] += wage
            self.metrics['transit_cost'] += transit
            
            self._log_step(vid, cid, 'LATE_SKIP', arrival_time, arrival_time, arrival_time,
                          0, 0, transit, wage, HARD_LATE_PENALTY)
                          
            # Remove
            if cid in bidder.route:
                bidder.route.remove(cid)
                bidder.route_demand -= cust['demand']
                
            # Update state (Assumed reached)
            bidder.current_time = arrival_time
            
            self._schedule_next(vid, arrival_time)
            return

        # SUCCESS
        wait = max(0, cust['E'] - arrival_time)
        svc_start = max(arrival_time, cust['E'])
        svc_dur = StochasticSampler.sample_service_time(cust['mean_service_time'])
        dept = svc_start + svc_dur
        
        # Update Estimator
        bidder.estimator.update_service(cust['mean_service_time'], svc_dur)
        
        # Metrics
        transit = planned_dist * TRANSIT_COST_PER_MILE
        wage = (actual_travel + wait + svc_dur) * WAGE_COST_PER_MINUTE
        
        self.metrics['transit_cost'] += transit
        self.metrics['wage_cost'] += wage
        
        self._log_step(vid, cid, 'SUCCESS', arrival_time, svc_start, dept,
                      wait, svc_dur, transit, wage, 0)
                      
        # Update State
        bidder.current_loc = cid
        bidder.current_time = dept
        bidder.remaining_capacity -= cust['demand']
        
        if cid in bidder.route:
            bidder.route.remove(cid)
            bidder.route_demand -= cust['demand']
            bidder.route_cost_cache = None
            
        # REPLANNING CHECK [Fix: Smart Trigger]
        if self._should_replan(bidder, cid, arrival_time):
             self._trigger_selective_replanning(bidder)
             
        self._schedule_next(vid, dept)

    def _should_replan(self, bidder, cid, arrival_time):
        cust = self.cust_map[cid]
        window = cust['L'] - cust['E']
        buffer = cust['L'] - arrival_time
        
        ratio = self.config.get('replan_buffer_ratio', 0.2)
        
        if window > 0 and buffer < (ratio * window):
            return True
        return False

    def _trigger_selective_replanning(self, bidder):
        """Keep feasible customers, reassign infeasible ones."""
        feasible = []
        infeasible = []
        
        # Detach route
        current_route = bidder.route
        bidder.route = []
        bidder.route_demand = 0
        bidder.route_cost_cache = None
        
        for cid in current_route:
            # Check feasibility of appending
            if bidder.calculate_insertion_cost(cid) is not None:
                bidder.insert_customer(cid)
                feasible.append(cid)
            else:
                infeasible.append(cid)
                
        # Send infeasible to Hybrid Assigner
        if infeasible:
            self.assigner.assign(self.bidders, infeasible)

    def _schedule_next(self, vid, time):
        bidder = self.bidders[vid]
        
        if not bidder.route:
            # Go Home
            if bidder.current_loc != 0:
                dist = self.dist_matrix[bidder.current_loc, 0]
                travel = StochasticSampler.sample_travel_time(dist)
                arr = time + travel
                heapq.heappush(self.event_queue, (arr, vid, 'DEPOT_ARRIVAL', 0, dist))
            return
            
        next_cid = bidder.route[0]
        dist = self.dist_matrix[bidder.current_loc, next_cid]
        travel = StochasticSampler.sample_travel_time(dist)
        arr = time + travel
        heapq.heappush(self.event_queue, (arr, vid, 'ARRIVAL', next_cid, dist))

    def _process_depot_arrival(self, vid, time):
        bidder = self.bidders[vid]
        actual_travel = time - bidder.current_time
        dist = self.dist_matrix[bidder.current_loc, 0]
        
        transit = dist * TRANSIT_COST_PER_MILE
        wage = actual_travel * WAGE_COST_PER_MINUTE
        
        self.metrics['transit_cost'] += transit
        self.metrics['wage_cost'] += wage
        
        self._log_step(vid, 0, 'DEPOT_END', time, time, time, 0, 0, transit, wage, 0)
        bidder.current_loc = 0
        bidder.current_time = time

    def _log_step(self, vid, nid, out, arr, ss, end, wait, dur, t_cost, w_cost, p_cost):
        self.traces[vid].append({
            'node_id': nid, 'outcome': out, 'arrival_time': round(arr, 2),
            'service_start': round(ss, 2), 'departure_time': round(end, 2),
            'wait_time': round(wait, 2), 'service_duration': round(dur, 2),
            'transit_cost': round(t_cost, 2), 'wage_cost': round(w_cost, 2),
            'penalty_cost': round(p_cost, 2)
        })

    def _reset_day_state(self):
        self.metrics = defaultdict(float)
        self.traces = [[] for _ in range(self.instance['num_vehicles'])]
        self.event_queue = []
        
        for b in self.bidders:
            b.current_loc = 0
            b.current_time = DEPOT_E_TIME
            b.remaining_capacity = self.instance['vehicle_capacity']
            b.route = []
            b.route_demand = 0
            b.route_cost_cache = None
            
            self.traces[b.id].append({
                'node_id': 0, 'outcome': 'DEPOT_START', 'arrival_time': DEPOT_E_TIME,
                'service_start': DEPOT_E_TIME, 'departure_time': DEPOT_E_TIME,
                'wait_time': 0, 'service_duration': 0, 'transit_cost': 0, 
                'wage_cost': 0, 'penalty_cost': 0
            })

    def _collect_results(self, day_idx):
        served = set()
        for t in self.traces:
            for s in t:
                if s['node_id'] != 0 and s['outcome'] == 'SUCCESS':
                    served.add(s['node_id'])
                    
        missed = len(self.cust_map) - len(served)
        penalty = missed * HARD_LATE_PENALTY
        self.metrics['missed_customers'] = missed
        self.metrics['penalty_cost'] += penalty
        self.metrics['total_cost'] = (self.metrics['transit_cost'] + 
                                      self.metrics['wage_cost'] + 
                                      self.metrics['penalty_cost'])
                                      
        return {
            'day_index': day_idx,
            'total_cost': self.metrics['total_cost'],
            'missed_customers': missed,
            'hard_lates': self.metrics['hard_lates'],
            'vehicle_traces': self.traces
        }

# ============================================================================
# MAIN
# ============================================================================

def evaluate_worker(filepath, config=None):
    """
    Worker function to evaluate a single instance with specific config.
    """
    try:
        data = load_instance(filepath)
        # Handle the dataframe vs list quirk
        if isinstance(data['customers'], list) is False:
            data['customers'] = data['customers'].to_dict(orient='records')
            
        # Initialize Controller with the loaded config (if any)
        controller = RobustCentralizedController(data, config=config)
        
        logs = []
        # Run for 30 days (or however many your standard eval requires)
        for d in range(30):
            logs.append(controller.run_day(d))
            
        costs = [l['total_cost'] for l in logs]
        missed = [l['missed_customers'] for l in logs]
        
        return {
            'instance_file': os.path.basename(filepath),
            'policy_type': 'Auction_Solver',
            'mean_cost': np.mean(costs),
            'mean_missed': np.mean(missed),
            'config_used': 'optimized' if config else 'default',
            'daily_simulation_logs': logs
        }
    except Exception as e:
        return {'error': str(e), 'instance_file': os.path.basename(filepath)}

def main():
    # Numba Warmup
    dummy_p = np.array([0.0, 0.0])
    fast_euclidean(dummy_p, dummy_p)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    instance_dir = os.path.join(script_dir, 'instances', 'data')
    results_dir = os.path.join(script_dir, 'solutions', 'AuctionSolver', 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # 1. Load Tuned Parameters if available
    config_path = os.path.join(script_dir, 'best_solver_params.json')
    solver_config = {}
    
    if os.path.exists(config_path):
        print(f"--- Loading Optimized Parameters from {os.path.basename(config_path)} ---")
        with open(config_path, 'r') as f:
            solver_config = json.load(f)
    else:
        print("--- No parameter file found. Using internal defaults. ---")

    files = sorted(glob.glob(os.path.join(instance_dir, '*.json')))
    print(f"--- Final Stochastic VRP Solver (Hybrid Auction) ---")
    print(f"{'Instance':<35} | {'Mean Cost':<12} | {'Missed':<8} | {'Config'}")
    print("-" * 75)
    
    # 2. Bind config to the worker function using partial
    # This ensures every worker gets the same tuned dict
    worker_with_config = partial(evaluate_worker, config=solver_config)
    
    max_workers = max(1, os.cpu_count())
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit the partial function
        futures = {executor.submit(worker_with_config, f): f for f in files}
        
        for future in as_completed(futures):
            res = future.result()
            if 'error' in res:
                print(f"{res['instance_file']:<35} | ERROR: {res['error']}")
                continue
            
            # Print row
            print(f"{res['instance_file']:<35} | "
                  f"${res['mean_cost']:<11,.0f} | "
                  f"{res['mean_missed']:<8.2f} | "
                  f"{res.get('config_used', 'default')}")
            
            # Save results
            out_name = res['instance_file'].replace('.json', '_final_results.json')
            with open(os.path.join(results_dir, out_name), 'w') as f:
                json.dump(res, f, indent=4)

if __name__ == '__main__':
    main()