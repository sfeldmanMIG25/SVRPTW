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
    Gymnasium Wrapper for the SVRPTW Simulator.
    State Representation:
        - Global: [Current_Time_Norm, Vehicle_Capacity_Norm]
        - Per Node: [Dist_From_Current, Demand_Norm, Time_Window_Start, Time_Window_End, Is_Visited]
    """
    def __init__(self, instance_folder, revenue_per_unit=10.0):
        super(VRPEnv, self).__init__()
        
        self.instance_folder = instance_folder
        self.instance_files = sorted([os.path.join(instance_folder, f) for f in os.listdir(instance_folder) if f.endswith('.json')])
        self.revenue_per_unit = revenue_per_unit
        
        # Pre-load all instances to memory for speed
        self.all_instances = []
        print(f"Loading {len(self.instance_files)} instances for RL Environment...")
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
            
        # Calculate Global Normalization Constants
        self.max_demand = max([sum(c['demand'] for c in i['customers']) for i in self.all_instances])
        self.max_dist = euclidean_distance((0,0), (100,100))
        self.time_horizon = DEPOT_L_TIME - DEPOT_E_TIME
        self.max_capacity = max([i['vehicle_capacity'] for i in self.all_instances])

        # Action Space: Node Index (0 to Max_Nodes). 
        self.action_space = spaces.Discrete(200) 
        
        # Observation Space
        self.observation_space = spaces.Dict({
            "global": spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32),
            "nodes": spaces.Box(low=0, high=1, shape=(200, 5), dtype=np.float32),
            "mask": spaces.Box(low=0, high=1, shape=(200,), dtype=np.bool_)
        })
        
        self.current_instance = None
        self.sim_state = None
        
    def reset(self, seed=None, options=None, instance_idx=None):
        """Resets to a new random instance (or specific one if provided)."""
        super().reset(seed=seed)
        
        if instance_idx is None:
            # Use numpy random if seed not provided, else use self.np_random
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
            'active_vehicle': 0, # Index of current vehicle being controlled
            'total_profit': 0,
            'total_cost': 0,
            'failures': 0,
            'service_time': 0
        }
        
        return self._get_observation(), {}

    def _get_observation(self):
        # FIX: Handle case where all vehicles are finished (done state)
        idx = self.sim_state['active_vehicle']
        if idx >= len(self.sim_state['vehicle_queue']):
             idx = len(self.sim_state['vehicle_queue']) - 1
             
        inst = self.current_instance
        veh = self.sim_state['vehicle_queue'][idx]
        curr_loc = inst['coords'][veh['loc']]
        
        # 1. Global Features: [Time_Norm, Capacity_Norm]
        time_norm = (self.sim_state['current_time'] - DEPOT_E_TIME) / self.time_horizon
        cap_norm = veh['cap'] / self.max_capacity
        global_feats = np.array([time_norm, cap_norm], dtype=np.float32)
        
        # 2. Node Features & Mask
        num_nodes = len(inst['customers']) + 1 # +1 for Depot
        node_feats = np.zeros((num_nodes, 5), dtype=np.float32)
        mask = np.zeros(num_nodes, dtype=bool)
        
        # Depot (Node 0)
        dist_0 = euclidean_distance(curr_loc, inst['coords'][0])
        node_feats[0] = [dist_0 / self.max_dist, 0, 0, 1, 0] # Depot always there
        mask[0] = 1 # Returning to depot is valid (ends shift)
        
        # Customers
        for c in inst['customers']:
            cid = c['id']
            # Feats: [Dist, Demand, Window_Start, Window_End, Is_Visited]
            d = euclidean_distance(curr_loc, inst['coords'][cid])
            
            is_visited = cid not in self.sim_state['unvisited_ids']
            
            # Mask Logic
            is_feasible = True
            if is_visited: is_feasible = False
            if c['demand'] > veh['cap']: is_feasible = False
            
            # Time feasibility (Deterministic check for mask)
            arrival = self.sim_state['current_time'] + d
            if arrival > c['L']: is_feasible = False
            
            mask[cid] = 1 if is_feasible else 0
            
            node_feats[cid] = [
                d / self.max_dist,
                c['demand'] / self.max_demand,
                (c['E'] - DEPOT_E_TIME) / self.time_horizon,
                (c['L'] - DEPOT_E_TIME) / self.time_horizon,
                1.0 if is_visited else 0.0
            ]
            
        return {
            "global": global_feats,
            "nodes": node_feats, # Shape (N, 5)
            "mask": mask,        # Shape (N,)
            "num_nodes": num_nodes # Helper to slice
        }

    def step(self, action):
        """
        Executes action (Node ID). 
        """
        inst = self.current_instance
        veh_idx = self.sim_state['active_vehicle']
        
        # Safety check if step called after done
        if veh_idx >= len(self.sim_state['vehicle_queue']):
             return self._get_observation(), 0, True, False, {}

        veh = self.sim_state['vehicle_queue'][veh_idx]
        terminated = False
        truncated = False
        
        # Validate Action
        if action == 0:
            # Return to Depot -> End Shift for this vehicle
            # Transit Cost
            curr_loc = inst['coords'][veh['loc']]
            target_loc = inst['coords'][0]
            dist = euclidean_distance(curr_loc, target_loc)
            
            travel_time = StochasticSampler.sample_travel_time(dist)
            arrival = veh['time'] + travel_time
            
            cost = (dist * TRANSIT_COST_PER_MILE) + (travel_time * WAGE_COST_PER_MINUTE)
            
            # Update State
            self.sim_state['total_cost'] += cost
            
            # Switch to next vehicle
            self.sim_state['active_vehicle'] += 1
            
            # Check Episode Done
            if self.sim_state['active_vehicle'] >= len(self.sim_state['vehicle_queue']):
                # All vehicles done
                terminated = True
                reward = 0 # Terminal step
                
                # Apply heavy penalty for any unvisited customers at very end
                missed_pen = len(self.sim_state['unvisited_ids']) * HARD_LATE_PENALTY
                
                # FIX: DO NOT ADD PENALTY TO REPORTED COST (as per user instruction)
                # self.sim_state['total_cost'] += missed_pen 
                self.sim_state['failures'] += len(self.sim_state['unvisited_ids'])
                
                # Reward = Revenue - Cost (for RL signal, we KEEP the penalty to guide learning)
                reward -= missed_pen
                
            else:
                # Next vehicle starts fresh at depot
                terminated = False
                reward = 0
                self.sim_state['current_time'] = DEPOT_E_TIME
                
        else:
            # Visit Customer
            cust = inst['cust_map'][action]
            curr_loc = inst['coords'][veh['loc']]
            target_loc = inst['coords'][action]
            dist = euclidean_distance(curr_loc, target_loc)
            
            # Stochastic Realization
            travel_time = StochasticSampler.sample_travel_time(dist)
            arrival = veh['time'] + travel_time
            
            # Costs
            transit_cost = dist * TRANSIT_COST_PER_MILE
            wage_min = travel_time
            revenue = 0
            penalty = 0
            service_time = 0
            
            # Logic
            if arrival > cust['L']:
                # Late Failure
                penalty = HARD_LATE_PENALTY
                self.sim_state['unvisited_ids'].discard(action) # Failed/Dropped
                self.sim_state['failures'] += 1
                finish_time = arrival
                
            elif arrival < cust['E']:
                # Early Wait
                wait = cust['E'] - arrival
                wage_min += wait
                service_start = cust['E']
                service_time = StochasticSampler.sample_service_time(cust['mean_service_time'])
                wage_min += service_time
                
                revenue = cust['demand'] * self.revenue_per_unit
                self.sim_state['unvisited_ids'].discard(action)
                veh['cap'] -= cust['demand']
                finish_time = service_start + service_time
                
            else:
                # On Time
                service_start = arrival
                service_time = StochasticSampler.sample_service_time(cust['mean_service_time'])
                wage_min += service_time
                
                revenue = cust['demand'] * self.revenue_per_unit
                self.sim_state['unvisited_ids'].discard(action)
                veh['cap'] -= cust['demand']
                finish_time = service_start + service_time
                
            op_cost = transit_cost + (wage_min * WAGE_COST_PER_MINUTE) + penalty
            
            # RL Reward = Revenue - Operational Cost
            reward = revenue - op_cost
            
            # Update Vehicle State
            veh['loc'] = action
            veh['time'] = finish_time
            self.sim_state['current_time'] = finish_time
            self.sim_state['total_cost'] += op_cost
            self.sim_state['total_profit'] += reward
            self.sim_state['service_time'] += service_time
            
            terminated = False
            
            # Auto-terminate if time limit exceeded for this vehicle
            if finish_time >= DEPOT_L_TIME:
                # In future logic, could force return to depot here
                pass

        return self._get_observation(), reward, terminated, truncated, {}