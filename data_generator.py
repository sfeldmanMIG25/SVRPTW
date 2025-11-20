import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
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
        # Coordinates will be NumPy floats, which is generally fine
        x = np.random.uniform(*COORDINATE_BOUNDS)
        y = np.random.uniform(*COORDINATE_BOUNDS)
        # Demand is a NumPy int
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
            'demand': int(demand), # Explicitly convert demand to standard Python int
            'E': E_i, # Earliest time
            'L': L_i, # Latest time
            'mean_service_time': mean_service_time
        })
        
    # 3. Determine Vehicle Capacity (Tight but Feasible)
    # Capacity Q = (Total Demand / Num Vehicles) * Buffer Factor
    # Buffer factor > 1 ensures feasibility, but a small factor ensures high utilization.
    buffer_factor = 1.25  # 25% buffer over the theoretical minimum
    vehicle_capacity = int(ceil((total_demand / num_vehicles) * buffer_factor)) # Ensure capacity is standard int
    
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

def create_visualization(instance, file_path):
    """Creates and saves a scatter plot visualization of the VRP instance."""
    df = instance['customers']
    
    # Calculate time window duration and normalize for color coding (tightness)
    df['duration'] = df['L'] - df['E']
    min_dur = df['duration'].min()
    max_dur = df['duration'].max()
    
    # Color map: Shorter duration (tighter window) = darker/hotter color
    colors = df['duration']
    cmap = plt.cm.plasma_r 
    
    plt.figure(figsize=(10, 10))
    
    # Plot Customers
    scatter = plt.scatter(df['x'], df['y'], 
                          c=colors, cmap=cmap, 
                          s=df['demand'] * 3, # Marker size proportional to demand
                          edgecolor='black', 
                          alpha=0.8)
    
    # Plot Depot (starting point)
    plt.scatter(instance['depot']['x'], instance['depot']['y'], 
                marker='s', color='red', s=300, label='Depot (Start/End)', zorder=5)
    
    # Add labels and title
    plt.title(f"SVRPTW Instance (N={instance['num_customers']}, V={instance['num_vehicles']}, Q={instance['vehicle_capacity']})", fontsize=14)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.xlim(*COORDINATE_BOUNDS)
    plt.ylim(*COORDINATE_BOUNDS)
    
    # Add Color Bar for Time Window Duration (Tightness)
    cbar = plt.colorbar(scatter)
    cbar.set_label('Time Window Duration (min) - [Tighter $\leftarrow$ Looser]', rotation=270, labelpad=20)

    # Add Legend for Demand (Size)
    handles, labels = scatter.legend_elements(prop="sizes", alpha=0.6, num=5)
    # Filter and format legend labels
    raw_sizes = [int(s/3) for s in scatter.properties()['sizes']]
    unique_sizes = sorted(list(set(raw_sizes)))
    
    # Find the indices corresponding to the min and max unique sizes to select handles
    legend_handles = []
    legend_labels = []
    
    # Create custom legend for demands
    for d in unique_sizes:
        # Create a proxy artist (circle) for the legend
        legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                         markerfacecolor='gray', markersize=sqrt(d*3), 
                                         markeredgecolor='black', linestyle=''))
        legend_labels.append(f"Demand: {d}")

    plt.legend(legend_handles, legend_labels, loc='lower left', title="Demand Size", scatterpoints=1)


    plt.grid(True, linestyle='--', alpha=0.6)
    plt.gca().set_aspect('equal', adjustable='box')
    
    # Save the figure
    plt.savefig(file_path)
    plt.close()

def generate_all_instances(n_instances=100, customer_counts=[20, 50, 100]):
    """Generates a list of all required problem instances and saves them."""
    
    # Define directory structure
    base_dir = 'instances'
    data_dir = os.path.join(base_dir, 'data')
    visuals_dir = os.path.join(base_dir, 'visuals')
    
    # Create directories if they don't exist
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(visuals_dir, exist_ok=True)
    
    print(f"Saving {n_instances} instances and visuals to: '{base_dir}/'")

    instances = []
    for i in range(1, n_instances + 1):
        # Select N and factor (biased towards 1/2)
        N = np.random.choice(customer_counts)
        factor = np.random.choice([0.25, 0.5, 0.75], p=[0.2, 0.6, 0.2]) 
        
        instance = generate_instance(num_customers=N, vehicle_factor=factor)
        instances.append(instance)
        
        # --- File Naming Convention ---
        # Example: N100_V050_I001.json
        N_str = str(instance['num_customers']).zfill(3)
        V_str = str(instance['num_vehicles']).zfill(3)
        I_str = str(i).zfill(3)
        filename_base = f"N{N_str}_V{V_str}_I{I_str}"
        
        # 1. Save Data (JSON)
        data_filepath = os.path.join(data_dir, f"{filename_base}.json")
        
        # Prepare data for JSON saving
        data_to_save = instance.copy()
        # Convert the customers DataFrame to a list of records (dicts)
        data_to_save['customers'] = data_to_save['customers'].to_dict(orient='records')

        # Convert remaining numpy floats/ints in the main dictionary to standard Python types
        for key in ['num_customers', 'num_vehicles', 'vehicle_capacity']:
            if isinstance(data_to_save[key], np.integer):
                data_to_save[key] = int(data_to_save[key])
            elif isinstance(data_to_save[key], np.floating):
                data_to_save[key] = float(data_to_save[key])

        # Recursively convert numpy types in the depot dictionary
        for key, value in data_to_save['depot'].items():
             if isinstance(value, (np.integer, np.floating)):
                data_to_save['depot'][key] = value.item() # .item() converts numpy scalar to standard Python scalar

        
        with open(data_filepath, 'w') as f:
            json.dump(data_to_save, f, indent=4)
        
        # 2. Save Visualization (PNG)
        visual_filepath = os.path.join(visuals_dir, f"{filename_base}.png")
        create_visualization(instance, visual_filepath)
        
        # print progress update
        if i % 10 == 0 or i == n_instances:
            print(f"  > Generated and saved instance {i}/{n_instances}: {filename_base}")

    return instances

if __name__ == '__main__':
    # The main execution block now calls the saving function
    all_instances = generate_all_instances(n_instances=100)
    
    # Display details of the first instance for verification
    first_instance = all_instances[0]
    print("\n--- Verification Sample (Instance 1) ---")
    print(f"Customers (N): {first_instance['num_customers']}")
    print(f"Vehicles (V): {first_instance['num_vehicles']}")
    print(f"Capacity (Q): {first_instance['vehicle_capacity']}")
    print(f"Total Demand: {first_instance['customers']['demand'].sum()}")
    print("Check the 'instances/data/' and 'instances/visuals/' folders for the saved files.")