import numpy as np
import pandas as pd
from scipy.stats import lognorm, norm
from data_generator import euclidean_distance
from config import (
    WAGE_COST_PER_MINUTE, TRANSIT_COST_PER_MILE, 
    EARLY_WAITING_PENALTY_PER_MINUTE, HARD_LATE_PENALTY,
    TRAVEL_TIME_LN_SIGMA, SERVICE_TIME_SIGMA,
    DEPOT_E_TIME, DEPOT_L_TIME
)

# --- UTILITY CLASS FOR STOCHASTIC REALIZATIONS ---
class StochasticSampler:
    """Handles realization of random variables for simulation."""

    @staticmethod
    def sample_travel_time(distance_mean_mi):
        if distance_mean_mi <= 0: return 0.0
        sigma_sq = TRAVEL_TIME_LN_SIGMA**2
        mu_log = np.log(distance_mean_mi) - (sigma_sq / 2)
        scale = np.exp(mu_log)
        return lognorm.rvs(s=TRAVEL_TIME_LN_SIGMA, loc=0, scale=scale)

    @staticmethod
    def sample_service_time(mean_time_min):
        return max(0, norm.rvs(loc=mean_time_min, scale=SERVICE_TIME_SIGMA))


# --- CORE SIMULATOR CLASS ---
class SVRPTW_Simulator:
    def __init__(self, instance_data):
        self.instance = instance_data
        self.depot = instance_data['depot']
        
        self.WAGE_COST_PER_MINUTE = WAGE_COST_PER_MINUTE
        self.TRANSIT_COST_PER_MILE = TRANSIT_COST_PER_MILE
        self.HARD_LATE_PENALTY = HARD_LATE_PENALTY
        
        self.customer_map = {c['id']: c for c in instance_data['customers']}
        self.customer_map[0] = self.depot 
        self.coordinates = {
            id: (node['x'], node['y']) for id, node in self.customer_map.items()
        }
        
    def _get_node_data(self, node_id):
        return self.customer_map.get(node_id, None)

    def run_vehicle_route(self, route_plan):
        """
        Simulates a single vehicle and records the full event trace.
        """
        if not route_plan or len(route_plan) < 2:
            return self._empty_result()
            
        current_time = self.depot['E']
        
        # Metrics
        total_distance = 0.0
        total_transit_time = 0.0
        total_service_time = 0.0
        total_billable_wait_time = 0.0
        total_unpaid_wait_time = 0.0
        hard_late_penalty_count = 0
        
        # Trace recorder
        vehicle_trace = []
        
        # Start Log
        current_node_id = route_plan[0]['node_id']
        vehicle_trace.append({
            'node_id': current_node_id,
            'type': 'DEPOT_START',
            'arrival_time': current_time,
            'service_start': current_time,
            'departure_time': current_time,
            'wait_time': 0,
            'service_duration': 0
        })
        
        # --- Simulate Sequence ---
        for next_step in route_plan[1:]:
            next_node_id = next_step['node_id']
            next_node_data = self._get_node_data(next_node_id)
            
            # 1. Travel
            p_current = self.coordinates[current_node_id]
            p_next = self.coordinates[next_node_id]
            mean_distance = euclidean_distance(p_current, p_next)
            
            realized_travel_time = StochasticSampler.sample_travel_time(mean_distance)
            
            total_distance += mean_distance
            total_transit_time += realized_travel_time
            time_of_arrival = current_time + realized_travel_time
            
            # 2. Arrival Logic
            is_customer = (next_node_id != 0)
            service_start_time = time_of_arrival
            realized_service_time = 0.0
            wait_time = 0.0
            outcome = 'SUCCESS'
            
            # CASE A: LATE
            if is_customer and time_of_arrival > next_node_data['L']:
                hard_late_penalty_count += 1
                outcome = 'LATE_SKIP'
                # Vehicle moves on immediately, no service, no wait.
                
            # CASE B: EARLY
            elif time_of_arrival < next_node_data['E']:
                wait_time = next_node_data['E'] - time_of_arrival
                service_start_time = next_node_data['E']
                
                if current_node_id == 0 and next_node_id == 0:
                     total_unpaid_wait_time += wait_time
                else:
                     total_billable_wait_time += wait_time
                
                if is_customer:
                    realized_service_time = StochasticSampler.sample_service_time(next_node_data['mean_service_time'])
            
            # CASE C: ON TIME
            else:
                if is_customer:
                    realized_service_time = StochasticSampler.sample_service_time(next_node_data['mean_service_time'])

            total_service_time += realized_service_time
            
            departure_time = service_start_time + realized_service_time
            
            # 3. Log Step
            vehicle_trace.append({
                'node_id': next_node_id,
                'type': 'CUSTOMER' if is_customer else 'DEPOT_END',
                'arrival_time': round(time_of_arrival, 2),
                'service_start': round(service_start_time, 2),
                'departure_time': round(departure_time, 2),
                'wait_time': round(wait_time, 2),
                'service_duration': round(realized_service_time, 2),
                'outcome': outcome
            })

            # 4. Update State
            current_time = departure_time
            current_node_id = next_node_id
            
        # --- Final Costs ---
        billable_minutes = total_transit_time + total_service_time + total_billable_wait_time
        wage_cost = billable_minutes * self.WAGE_COST_PER_MINUTE
        transit_cost = total_distance * self.TRANSIT_COST_PER_MILE
        penalty_cost = hard_late_penalty_count * self.HARD_LATE_PENALTY
        total_cost = wage_cost + transit_cost + penalty_cost
        
        return {
            'total_cost': total_cost,
            'total_distance_mi': total_distance,
            'total_transit_min': total_transit_time,
            'total_billable_wait_min': total_billable_wait_time,
            'total_service_min': total_service_time,
            'hard_late_penalty_count': hard_late_penalty_count,
            'trace': vehicle_trace
        }

    def _empty_result(self):
        return {
            'total_cost': 0.0, 'total_distance_mi': 0.0, 
            'total_transit_min': 0.0, 'total_billable_wait_min': 0.0,
            'total_service_min': 0.0, 'hard_late_penalty_count': 0, 
            'trace': []
        }

    def run_policy_for_day(self, policy_routes):
        """
        Runs all vehicle routes for a specific stochastic realization (one day).
        Returns aggregated stats AND the full list of vehicle traces.
        """
        day_results = []
        
        for route_data in policy_routes:
            # Check format of route_data
            if len(route_data) > 0 and isinstance(route_data[0], dict):
                 node_sequence = [{'node_id': step['node_id']} for step in route_data]
            else:
                 # Fallback if just IDs passed
                 node_sequence = [{'node_id': nid} for nid in route_data]
                 
            vehicle_result = self.run_vehicle_route(node_sequence)
            day_results.append(vehicle_result)
        
        # Aggregate
        aggregated = {
            'total_cost': sum(r['total_cost'] for r in day_results),
            'total_distance_mi': sum(r['total_distance_mi'] for r in day_results),
            'hard_late_penalty_count': sum(r['hard_late_penalty_count'] for r in day_results),
            'vehicle_traces': [r['trace'] for r in day_results] # The Full Sequence
        }
        
        return aggregated