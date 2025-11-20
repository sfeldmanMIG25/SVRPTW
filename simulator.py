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
        """Samples travel time based on distance using a Log-Normal distribution."""
        if distance_mean_mi <= 0:
            return 0.0
            
        # Adjusting the scale parameter to account for variance (E[X] = exp(mu + sigma^2/2))
        sigma_sq = TRAVEL_TIME_LN_SIGMA**2
        mu_log = np.log(distance_mean_mi) - (sigma_sq / 2)
        scale = np.exp(mu_log)
        
        #
        return lognorm.rvs(s=TRAVEL_TIME_LN_SIGMA, loc=0, scale=scale)

    @staticmethod
    def sample_service_time(mean_time_min):
        """Samples service time using a Normal distribution."""
        #
        return max(0, norm.rvs(loc=mean_time_min, scale=SERVICE_TIME_SIGMA))


# --- CORE SIMULATOR CLASS ---
class SVRPTW_Simulator:
    """
    Executes a predefined route/policy for a single vehicle across one stochastic day.
    Updates: 
    1. Wage costs are only incurred when the vehicle is 'active' (not waiting at depot).
    2. Handles discrete events.
    """
    def __init__(self, instance_data):
        self.instance = instance_data
        self.depot = instance_data['depot']
        
        self.WAGE_COST_PER_MINUTE = WAGE_COST_PER_MINUTE
        self.TRANSIT_COST_PER_MILE = TRANSIT_COST_PER_MILE
        self.HARD_LATE_PENALTY = HARD_LATE_PENALTY
        
        # Map IDs to data
        self.customer_map = {c['id']: c for c in instance_data['customers']}
        self.customer_map[0] = self.depot 
        self.coordinates = {
            id: (node['x'], node['y']) for id, node in self.customer_map.items()
        }
        
    def _get_node_data(self, node_id):
        return self.customer_map.get(node_id, None)

    def run_vehicle_route(self, route_plan):
        """
        Simulates a single vehicle.
        Wage Cost Logic:
        - Transit Time: Always Paid.
        - Service Time: Always Paid.
        - Waiting Time: Paid ONLY if current_node != Depot.
        """
        if not route_plan or len(route_plan) < 2:
            return self._empty_result()
            
        current_time = self.depot['E'] # Simulation starts at 8:00 AM
        
        # Metrics
        total_distance = 0.0
        
        # Time Buckets
        total_transit_time = 0.0
        total_service_time = 0.0
        total_billable_wait_time = 0.0 # Waiting at customers
        total_unpaid_wait_time = 0.0   # Waiting at depot
        
        hard_late_penalty_count = 0
        
        # Start state
        current_node_id = route_plan[0]['node_id']
        
        # --- Simulate Sequence ---
        for next_step in route_plan[1:]:
            next_node_id = next_step['node_id']
            next_node_data = self._get_node_data(next_node_id)
            
            # 1. Distance & Transit
            p_current = self.coordinates[current_node_id]
            p_next = self.coordinates[next_node_id]
            mean_distance = euclidean_distance(p_current, p_next)
            
            realized_travel_time = StochasticSampler.sample_travel_time(mean_distance)
            
            total_distance += mean_distance
            total_transit_time += realized_travel_time
            
            time_of_arrival = current_time + realized_travel_time
            
            # 2. Check Time Windows
            is_customer = (next_node_id != 0)
            service_start_time = time_of_arrival
            realized_service_time = 0.0
            wait_time = 0.0
            
            # CASE A: LATE (Hard Penalty)
            if is_customer and time_of_arrival > next_node_data['L']:
                hard_late_penalty_count += 1
                # Service denied. Vehicle arrives, realizes it's late, moves on.
                # No service time, no wait time.
                pass
                
            # CASE B: EARLY (Wait)
            elif time_of_arrival < next_node_data['E']:
                wait_time = next_node_data['E'] - time_of_arrival
                service_start_time = next_node_data['E']
                
                # WAGE LOGIC: Is this wait billable?
                # If we are waiting at the customer's location, yes.
                # If we travelled from Depot -> Customer and arrived early, 
                # that wait happens at the customer site (or just outside). It is billable.
                # NOTE: The prompt implies "No penalty for vehicles waiting AT the depot".
                # Once a vehicle leaves the depot (current_node=0 -> next_node=X), 
                # any subsequent wait is "in the field".
                
                # However, if we are at Depot (0) and going to Depot (0) (staying put), 
                # that is unpaid.
                if current_node_id == 0 and next_node_id == 0:
                     total_unpaid_wait_time += wait_time
                else:
                     # Even if we came FROM depot, if we arrive early at customer, 
                     # we are sitting in the truck at the customer site. That is billable.
                     total_billable_wait_time += wait_time
                
                # Realize Service
                if is_customer:
                    realized_service_time = StochasticSampler.sample_service_time(next_node_data['mean_service_time'])
            
            # CASE C: ON TIME
            else:
                if is_customer:
                    realized_service_time = StochasticSampler.sample_service_time(next_node_data['mean_service_time'])

            total_service_time += realized_service_time
            
            # 3. Update State
            current_time = service_start_time + realized_service_time
            current_node_id = next_node_id
            
            # Special Check: If we just returned to depot, we stop processing if it's the end of plan
            # But the loop handles that naturally. 
            # Any time spent "waiting" after returning to depot (if simulated) would be unpaid.

        # --- 4. Calculate Final Costs ---
        
        # Wage Cost: We pay for Transit + Service + Billable Waiting
        # We DO NOT pay for Unpaid Wait (sitting at depot)
        billable_minutes = total_transit_time + total_service_time + total_billable_wait_time
        
        wage_cost = billable_minutes * self.WAGE_COST_PER_MINUTE
        transit_cost = total_distance * self.TRANSIT_COST_PER_MILE
        penalty_cost = hard_late_penalty_count * self.HARD_LATE_PENALTY
        
        total_cost = wage_cost + transit_cost + penalty_cost
        
        # Total simulation time (for metrics only, not cost)
        total_sim_time = total_transit_time + total_service_time + total_billable_wait_time + total_unpaid_wait_time

        return {
            'total_cost': total_cost,
            'total_distance_mi': total_distance,
            'total_time_min': total_sim_time,
            'billable_time_min': billable_minutes,
            'total_transit_min': total_transit_time,
            'total_billable_wait_min': total_billable_wait_time,
            'total_unpaid_wait_min': total_unpaid_wait_time,
            'total_service_min': total_service_time,
            'hard_late_penalty_count': hard_late_penalty_count,
            'service_efficiency': total_service_time / max(1, total_transit_time)
        }

    def _empty_result(self):
        return {
            'total_cost': 0.0, 'total_distance_mi': 0.0, 
            'total_time_min': 0.0, 'billable_time_min': 0.0,
            'total_transit_min': 0.0, 'total_billable_wait_min': 0.0,
            'total_unpaid_wait_min': 0.0, 'total_service_min': 0.0,
            'hard_late_penalty_count': 0, 'service_efficiency': 0.0
        }

    def run_policy_for_day(self, policy_routes):
        """
        Runs all vehicle routes defined by the policy.
        """
        day_results = []
        
        for route_data in policy_routes:
            node_sequence = [step['node_id'] for step in route_data]
            vehicle_result = self.run_vehicle_route([{'node_id': id} for id in node_sequence])
            day_results.append(vehicle_result)
        
        # Aggregate
        aggregated = {
            'total_cost': sum(r['total_cost'] for r in day_results),
            'total_distance_mi': sum(r['total_distance_mi'] for r in day_results),
            'total_time_min': sum(r['total_time_min'] for r in day_results),
            'billable_time_min': sum(r['billable_time_min'] for r in day_results),
            'total_transit_min': sum(r['total_transit_min'] for r in day_results),
            'total_billable_wait_min': sum(r['total_billable_wait_min'] for r in day_results),
            'total_service_min': sum(r['total_service_min'] for r in day_results),
            'hard_late_penalty_count': sum(r['hard_late_penalty_count'] for r in day_results),
            'total_service_efficiency': sum(r['total_service_min'] for r in day_results) / max(1, sum(r['total_transit_min'] for r in day_results))
        }
        
        return aggregated

if __name__ == '__main__':
    # Simple test block to verify wage logic
    print("Simulator loaded. Run stochastic_evaluator.py or greedy_evaluator.py to test.")