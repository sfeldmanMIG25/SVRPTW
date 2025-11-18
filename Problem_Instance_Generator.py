import numpy as np
import pandas as pd
from scipy.stats import lognorm, norm
from config import (
    COORDINATE_BOUNDS, DEPOT_COORDINATE,
    SERVICE_TIME_BASE_MEAN, SERVICE_TIME_SIGMA
)
from math import sqrt, ceil

def euclidean_distance(p1, p2):
    """Calculates the Euclidean distance between two points."""
    return sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def generate_instance(num_customers, vehicle_factor=0.5, demand_range=(10, 25), time_window_range=(60, 180)):
    """
    Generates a feasible SVRPTW problem instance.
    
    Args:
        num_customers (int): The number of customers (N).
        vehicle_factor (float): V will be set to ceil(N * vehicle_factor). Default is 0.5.
        demand_range (tuple): Min and Max demand for customers.
        time_window_range (tuple): Min and Max duration of a customer's time window (in minutes).

    Returns:
        dict: A dictionary containing all instance data.
    """
    
    # 1. Determine Vehicle Count
    num_vehicles = max(1, ceil(num_customers * vehicle_factor))
    
    # 2. Generate Customer Data
    customer_data = []
    total_demand = 0
    
    # Randomly generate customer points and demands
    for i in range(1, num_customers + 1):
        x = np.random.uniform(*COORDINATE_BOUNDS)
        y = np.random.uniform(*COORDINATE_BOUNDS)
        demand = np.random.randint(*demand_range)
        total_demand += demand
        
        # Time Window Generation
        # Assume a standard operating day starts at 480 minutes (8:00 AM) and ends at 960 minutes (4:00 PM)
        # Time in minutes from midnight.
        earliest_start = 480 
        latest_end = 960
        
        # Calculate max duration for time window
        max_duration = np.random.uniform(*time_window_range) 
        
        # Randomly place the time window start (E_i) within the operating day
        E_i = np.random.uniform(earliest_start, latest_end - max_duration)
        L_i = E_i + max_duration
        
        # Mean service time is small fraction of the window duration
        mean_service_time = SERVICE_TIME_BASE_MEAN 
        
        customer_data.append({
            'id': i,
            'x': x,
            'y': y,
            'demand': demand,
            'E': E_i, # Earliest time
            'L': L_i, # Latest time
            'mean_service_time': mean_service_time
        })
        
    # 3. Determine Vehicle Capacity (Tight but Feasible)
    # Capacity Q = (Total Demand / Num Vehicles) * Buffer Factor
    # Buffer factor > 1 ensures feasibility, but a small factor ensures high utilization.
    buffer_factor = 1.25  # 25% buffer over the theoretical minimum
    vehicle_capacity = ceil((total_demand / num_vehicles) * buffer_factor)
    
    # 4. Create Instance Dictionary
    instance = {
        'num_customers': num_customers,
        'num_vehicles': num_vehicles,
        'vehicle_capacity': vehicle_capacity,
        'depot': {
            'id': 0,
            'x': DEPOT_COORDINATE[0],
            'y': DEPOT_COORDINATE[1],
            'E': earliest_start, # Depot available time
            'L': latest_end,   # Depot return time
        },
        'customers': pd.DataFrame(customer_data)
    }
    
    return instance

def generate_all_instances(n_instances=100, customer_counts=[20, 50, 100]):
    """Generates a list of all required problem instances."""
    instances = []
    for _ in range(n_instances):
        # Select a random number of customers from the target range
        N = np.random.choice(customer_counts)
        # Apply the proportional vehicle factor (e.g., 1/4, 1/2, 3/4)
        factor = np.random.choice([0.25, 0.5, 0.75], p=[0.2, 0.6, 0.2]) # Bias towards 1/2
        
        instance = generate_instance(num_customers=N, vehicle_factor=factor)
        instances.append(instance)
        
    return instances

if __name__ == '__main__':
    # Example usage: Generate 100 instances with N=20, 50, 100
    all_instances = generate_all_instances(n_instances=100)
    print(f"Generated {len(all_instances)} problem instances.")
    
    # Display details of the first instance for verification
    first_instance = all_instances[0]
    print("\n--- First Instance Details ---")
    print(f"Customers (N): {first_instance['num_customers']}")
    print(f"Vehicles (V): {first_instance['num_vehicles']}")
    print(f"Capacity (Q): {first_instance['vehicle_capacity']}")
    print(f"Total Demand: {first_instance['customers']['demand'].sum()}")
    print("\nCustomer Time Window Sample:")
    print(first_instance['customers'][['id', 'E', 'L', 'demand']].head(3))