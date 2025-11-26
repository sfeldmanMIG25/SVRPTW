import gymnasium as gym
import numpy as np
import os
import copy
from gymnasium import spaces

from config import (
    WAGE_COST_PER_MINUTE, TRANSIT_COST_PER_MILE,
    DEPOT_E_TIME, DEPOT_L_TIME, HARD_LATE_PENALTY,
    COORDINATE_BOUNDS
)
from simulator import StochasticSampler
from data_generator import euclidean_distance
from deterministic_policy_generator import load_instance

class VRPEnv(gym.Env):
    """
    Gymnasium Wrapper for SVRPTW.
    Matches the 'High Variance' system dynamics and 'New JSON' reporting style.
    """
    def __init__(self, instance_folder, revenue_per_unit=10.0):
        super(VRPEnv, self).__init__()
        
        self.instance_folder = instance_folder
        self.instance_files = sorted([os.path.join(instance_folder, f) for f in os.listdir(instance_folder) if f.endswith('.json')])
        self.revenue_per_unit = revenue_per_unit
        
        # Pre-load instances
        self.all_instances = []
        # print(f"Loading {len(self.instance_files)} instances for RL...")
        for f in self.instance_files:
            data = load_instance(f)
            if isinstance(data['customers'], list) is False:
                data['customers'] = data['customers'].to_dict(orient='records')
            
            # Inject Lookups
            data['cust_map'] = {c['id']: c for c in data['customers']}
            data['cust_map'][0] = data['depot']
            data['coords'] = {c['id']: (c['x'], c['y']) for c in data['customers']}
            data['coords'][0] = (data['depot']['x'], data['depot']['y'])
            self.all_instances.append(data)
            
        # Normalization Constants
        self.max_demand = max([sum(c['demand'] for c in i['customers']) for i in self.all_instances])
        self.max_dist = euclidean_distance((0,0), (100,100))
        self.time_horizon = DEPOT_L_TIME - DEPOT_E_TIME
        self.max_capacity = max([i['vehicle_capacity'] for i in self.all_instances])

        # Action Space: Node ID (0 to 200)
        self.action_space = spaces.Discrete(200) 
        
        # Observation Space: 
        # Global: [Time_Norm, Cap_Norm]
        # Nodes: [Dist, Demand, Window_Start, Window_End, Is_Visited, Gap_to_Late]
        self.observation_space = spaces.Dict({
            "global": spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32),
            "nodes": spaces.Box(low=-1, high=2, shape=(200, 6), dtype=np.float32), # Expanded range for time diffs
            "mask": spaces.Box(low=0, high=1, shape=(200,), dtype=np.bool_)
        })
        
        self.current_instance = None
        self.sim_state = None
        
    def reset(self, seed=None, options=None, instance_idx=None):
        super().reset(seed=seed)
        
        if instance_idx is None:
            self.current_instance = self.all_instances[np.random.randint(len(self.all_instances))]
        else:
            self.current_instance = self.all_instances[instance_idx]
            
        # Initialize Simulation State
        self.sim_state = {
            'current_time': DEPOT_E_TIME,
            'unvisited_ids': set(c['id'] for c in self.current_instance['customers']),
            'vehicle_queue': [
                {'id': v, 'loc': 0, 'time': DEPOT_E_TIME, 'cap': self.current_instance['vehicle_capacity']}
                for v in range(self.current_instance['num_vehicles'])
            ],
            'active_vehicle': 0, 
            'total_cost': 0,
            'failures': 0, # Missed + Late
            'hard_lates': 0,
            'service_time': 0,
            # Trace Storage for JSON
            'vehicle_traces': [ [] for _ in range(self.current_instance['num_vehicles']) ] 
        }
        
        # Log Start for Vehicle 0
        self._log_step(0, 'DEPOT_START', DEPOT_E_TIME, DEPOT_E_TIME, DEPOT_E_TIME, 0, 0, 0, 0, 0, 0)
        
        return self._get_observation(), {}

    def _log_step(self, v_idx, outcome, arr, svc_start, end, wait, svc_dur, t_cost, w_cost, p_cost, dist):
        """Helper to append to vehicle trace."""
        step = {
            'node_id': self.sim_state['vehicle_queue'][v_idx]['loc'],
            'outcome': outcome,
            'arrival_time': float(arr),
            'service_start': float(svc_start),
            'departure_time': float(end),
            'wait_time': float(wait),
            'service_duration': float(svc_dur),
            'transit_cost': float(t_cost),
            'wage_cost': float(w_cost),
            'penalty_cost': float(p_cost),
            'dist': float(dist)
        }
        self.sim_state['vehicle_traces'][v_idx].append(step)

    def _get_observation(self):
        idx = self.sim_state['active_vehicle']
        if idx >= len(self.sim_state['vehicle_queue']): idx = len(self.sim_state['vehicle_queue']) - 1
             
        inst = self.current_instance
        veh = self.sim_state['vehicle_queue'][idx]
        curr_loc = inst['coords'][veh['loc']]
        
        # 1. Global
        time_norm = (self.sim_state['current_time'] - DEPOT_E_TIME) / self.time_horizon
        cap_norm = veh['cap'] / self.max_capacity
        global_feats = np.array([time_norm, cap_norm], dtype=np.float32)
        
        # 2. Nodes
        num_nodes = len(inst['customers']) + 1 
        node_feats = np.zeros((num_nodes, 6), dtype=np.float32)
        mask = np.zeros(num_nodes, dtype=bool)
        
        # Depot (Node 0)
        dist_0 = euclidean_distance(curr_loc, inst['coords'][0])
        # [Dist, Demand, E, L, Visited, Gap]
        node_feats[0] = [dist_0 / self.max_dist, 0, 0, 1, 0, 1.0] 
        mask[0] = 1 # Always valid to return to depot
        
        # Customers
        for c in inst['customers']:
            cid = c['id']
            d = euclidean_distance(curr_loc, inst['coords'][cid])
            is_visited = cid not in self.sim_state['unvisited_ids']
            
            # Feasibility Mask (Deterministic check)
            is_feasible = True
            if is_visited: is_feasible = False
            if c['demand'] > veh['cap']: is_feasible = False
            
            arrival_est = self.sim_state['current_time'] + d
            if arrival_est > c['L']: is_feasible = False
            
            mask[cid] = 1 if is_feasible else 0
            
            gap_to_late = (c['L'] - arrival_est) / 60.0 # Normalized roughly by hours
            
            node_feats[cid] = [
                d / self.max_dist,
                c['demand'] / self.max_demand,
                (c['E'] - DEPOT_E_TIME) / self.time_horizon,
                (c['L'] - DEPOT_E_TIME) / self.time_horizon,
                1.0 if is_visited else 0.0,
                gap_to_late
            ]
            
        return {
            "global": global_feats,
            "nodes": node_feats, 
            "mask": mask
        }

    def step(self, action):
        inst = self.current_instance
        veh_idx = self.sim_state['active_vehicle']
        
        if veh_idx >= len(self.sim_state['vehicle_queue']):
             return self._get_observation(), 0, True, False, {}

        veh = self.sim_state['vehicle_queue'][veh_idx]
        curr_loc_id = veh['loc']
        curr_loc = inst['coords'][curr_loc_id]
        
        target_loc = inst['coords'][action]
        dist = euclidean_distance(curr_loc, target_loc)
        
        # Stochastic Realization
        travel_time = StochasticSampler.sample_travel_time(dist)
        arrival = veh['time'] + travel_time
        
        transit_cost = dist * TRANSIT_COST_PER_MILE
        wage_billable = travel_time
        
        reward = 0
        penalty = 0
        service_time = 0
        wait_time = 0
        outcome = 'SUCCESS'
        revenue = 0
        
        # --- LOGIC ---
        if action == 0:
            # Depot Return
            outcome = 'DEPOT_END'
            service_start = arrival
            finish_time = arrival
            # Wait at depot is UNPAID
        else:
            cust = inst['cust_map'][action]
            
            if arrival > cust['L']:
                # Late
                outcome = 'LATE_SKIP'
                penalty = HARD_LATE_PENALTY
                self.sim_state['unvisited_ids'].discard(action)
                self.sim_state['hard_lates'] += 1
                self.sim_state['failures'] += 1
                service_start = arrival
                finish_time = arrival
                
            elif arrival < cust['E']:
                # Early
                wait_time = cust['E'] - arrival
                # Paid wait logic: Paid unless staying at depot
                if curr_loc_id != 0 or action != 0:
                    wage_billable += wait_time
                    
                service_start = cust['E']
                service_time = StochasticSampler.sample_service_time(cust['mean_service_time'])
                wage_billable += service_time
                
                revenue = cust['demand'] * self.revenue_per_unit
                self.sim_state['unvisited_ids'].discard(action)
                veh['cap'] -= cust['demand']
                finish_time = service_start + service_time
                
            else:
                # On Time
                service_start = arrival
                service_time = StochasticSampler.sample_service_time(cust['mean_service_time'])
                wage_billable += service_time
                
                revenue = cust['demand'] * self.revenue_per_unit
                self.sim_state['unvisited_ids'].discard(action)
                veh['cap'] -= cust['demand']
                finish_time = service_start + service_time
        
        wage_cost = wage_billable * WAGE_COST_PER_MINUTE
        op_cost = transit_cost + wage_cost + penalty
        
        # Reward = Revenue - Cost
        reward = revenue - op_cost
        
        # Update Global State
        self.sim_state['total_cost'] += op_cost
        self.sim_state['service_time'] += service_time
        
        # Log Step
        self._log_step(veh_idx, outcome, arrival, service_start, finish_time, wait_time, service_time, 
                       transit_cost, wage_cost, penalty, dist)
        
        # Update Vehicle
        veh['loc'] = action
        veh['time'] = finish_time
        self.sim_state['current_time'] = finish_time
        
        terminated = False
        
        # Check Vehicle Completion
        if action == 0 or finish_time >= DEPOT_L_TIME:
            # Move to next vehicle
            self.sim_state['active_vehicle'] += 1
            if self.sim_state['active_vehicle'] >= len(self.sim_state['vehicle_queue']):
                terminated = True
                
                # End of Episode Penalty for Missed Customers
                missed = len(self.sim_state['unvisited_ids'])
                if missed > 0:
                    reward -= (missed * HARD_LATE_PENALTY)
                    self.sim_state['failures'] += missed
            else:
                # Setup next vehicle
                self.sim_state['current_time'] = DEPOT_E_TIME
                # Log Start for Next Vehicle
                self._log_step(self.sim_state['active_vehicle'], 'DEPOT_START', DEPOT_E_TIME, DEPOT_E_TIME, DEPOT_E_TIME, 0, 0, 0, 0, 0, 0)

        return self._get_observation(), reward, terminated, False, {}