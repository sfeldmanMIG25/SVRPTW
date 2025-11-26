import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import time
from math import sqrt, ceil
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from config import (
    COORDINATE_BOUNDS, DEPOT_COORDINATE,
    SERVICE_TIME_BASE_MEAN
)

def euclidean_distance(p1, p2):
    return sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def generate_instance(num_customers):
    """
    Generates an EXTREME-Variance Stress Test Instance.
    - Tight Windows (30-90m)
    - Long Routes (Low Vehicle Factor)
    - High Capacity (Failures are purely time-based)
    """
    
    # --- STRESS TEST CONFIGURATION ---
    # 0.30 means ~3 vehicles for 10 customers. 
    # Long routes maximize the accumulation of delays.
    vehicle_factor = 0.30
    
    # 40% Capacity Buffer -> We almost never fail on Load.
    # Failures will be driven by Time Variance.
    capacity_buffer = 1.40 
    
    # EXTREME TIGHTNESS: 30 to 90 minutes
    time_window_range = (30, 90)

    # --- GENERATION LOGIC ---
    num_vehicles = max(2, int(ceil(num_customers * vehicle_factor)))
    
    customer_data = []
    total_demand = 0
    
    day_start = 480
    day_end = 960
    
    for i in range(1, num_customers + 1):
        x = np.random.uniform(*COORDINATE_BOUNDS)
        y = np.random.uniform(*COORDINATE_BOUNDS)
        
        demand = np.random.randint(10, 25)
        total_demand += demand
        
        # Time Window
        max_duration = np.random.uniform(*time_window_range) 
        
        dist_from_depot = euclidean_distance((x, y), DEPOT_COORDINATE)
        # Increased padding factor (2.0) to account for the heavy-tail travel times
        travel_padding = dist_from_depot * 2.0 
        
        min_E = day_start + travel_padding
        max_E = day_end - max_duration - travel_padding - SERVICE_TIME_BASE_MEAN
        
        if max_E < min_E:
            E_i = (day_start + day_end) / 2 - (max_duration/2)
        else:
            E_i = np.random.uniform(min_E, max_E)
            
        L_i = E_i + max_duration
        
        customer_data.append({
            'id': i,
            'x': x,
            'y': y,
            'demand': int(demand),
            'E': E_i, 
            'L': L_i, 
            'mean_service_time': SERVICE_TIME_BASE_MEAN
        })
        
    avg_demand_per_vehicle = total_demand / num_vehicles
    vehicle_capacity = int(ceil(avg_demand_per_vehicle * capacity_buffer))
    
    instance = {
        'num_customers': num_customers,
        'num_vehicles': num_vehicles,
        'vehicle_capacity': vehicle_capacity,
        'depot': {
            'id': 0,
            'x': DEPOT_COORDINATE[0],
            'y': DEPOT_COORDINATE[1],
            'E': day_start,
            'L': day_end,
        },
        'customers': pd.DataFrame(customer_data)
    }
    return instance

def create_visualization(instance, file_path):
    plt.switch_backend('Agg') 
    
    df = instance['customers']
    plt.figure(figsize=(8, 8))
    
    durations = df['L'] - df['E']
    plt.scatter(df['x'], df['y'], c=durations, cmap='viridis', s=df['demand']*4, edgecolors='black', alpha=0.8)
    plt.colorbar(label='Window Duration (Darker = Shorter)')
    
    plt.scatter(instance['depot']['x'], instance['depot']['y'], marker='s', color='red', s=200, label='Depot')
    plt.title(f"Extreme Instance: N={instance['num_customers']} V={instance['num_vehicles']}")
    plt.xlim(0, 100); plt.ylim(0, 100)
    plt.legend()
    plt.savefig(file_path)
    plt.close()

def _worker_generate_single_instance(i, scenarios, data_dir, visuals_dir):
    try:
        np.random.seed(int(time.time() * 1000) % 2**32 + i)
        
        N = np.random.choice(scenarios)
        instance = generate_instance(num_customers=N)
        
        filename = f"N{str(N).zfill(3)}_V{str(instance['num_vehicles']).zfill(3)}_I{str(i).zfill(3)}"
        
        data_to_save = instance.copy()
        data_to_save['customers'] = data_to_save['customers'].to_dict(orient='records')
        
        for k in ['num_customers', 'num_vehicles', 'vehicle_capacity']:
            data_to_save[k] = int(data_to_save[k])
        for k, v in data_to_save['depot'].items():
            if hasattr(v, 'item'): data_to_save['depot'][k] = v.item()

        json_path = os.path.join(data_dir, f"{filename}.json")
        with open(json_path, 'w') as f:
            json.dump(data_to_save, f, indent=4)
            
        visual_path = os.path.join(visuals_dir, f"{filename}.png")
        create_visualization(instance, visual_path)
        
        return f"Generated {filename}"
    except Exception as e:
        return f"Error on index {i}: {e}"

def generate_all_instances(n_instances=100):
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'instances')
    data_dir = os.path.join(base_dir, 'data')
    visuals_dir = os.path.join(base_dir, 'visuals')
    
    print("Cleaning old instances...")
    if os.path.exists(data_dir):
        for f in os.listdir(data_dir):
            if f.endswith('.json'): os.remove(os.path.join(data_dir, f))
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(visuals_dir, exist_ok=True)
    
    print(f"Generating {n_instances} Extreme-Variance instances (Parallel)...")
    
    scenarios = [20, 50, 100] 
    
    max_workers = os.cpu_count()
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_worker_generate_single_instance, i, scenarios, data_dir, visuals_dir): i
            for i in range(1, n_instances + 1)
        }
        
        results = []
        for future in tqdm(as_completed(futures), total=n_instances, desc="Generating", unit="inst"):
            results.append(future.result())

    print("Generation Complete.")

if __name__ == '__main__':
    generate_all_instances(n_instances=100)