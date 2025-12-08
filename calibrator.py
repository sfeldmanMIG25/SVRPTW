import numpy as np
import os
import json
from vrp_gym_env import VRPEnv
from config import DEPOT_E_TIME

# --- CONFIGURATION ---
CALIBRATION_EPISODES = 50  # Number of "days" to observe
OUTPUT_FILE = 'best_solver_params.json'

def run_calibration():
    print(f"--- Starting Environment Calibration ({CALIBRATION_EPISODES} episodes) ---")
    
    # 1. Initialize Environment
    # We use the same environment the RL agent uses
    script_dir = os.path.dirname(os.path.abspath(__file__))
    instance_dir = os.path.join(script_dir, 'instances', 'data')
    env = VRPEnv(instance_dir)
    
    # Data collectors
    travel_errors = []  # (Actual Time - Predicted Dist)
    service_errors = [] # (Actual Svc - Mean Svc)
    
    # 2. Data Collection Loop
    for ep in range(CALIBRATION_EPISODES):
        obs, _ = env.reset()
        done = False
        
        while not done:
            # We don't need a smart policy, we just need to move to generate travel data.
            # A random policy is sufficient to sample the "physics" of the world.
            mask = obs['mask']
            valid_actions = np.where(mask)[0]
            
            if len(valid_actions) == 0:
                action = 0
            else:
                action = np.random.choice(valid_actions)
                
            # Step
            next_obs, _, terminated, truncated, _ = env.step(action)
            
            # Extract the last step's data from the environment trace
            # VRPEnv stores traces in sim_state['vehicle_traces']
            current_vehicle = env.sim_state['active_vehicle']
            # Handle edge case where vehicle just switched
            if current_vehicle < len(env.sim_state['vehicle_traces']):
                trace = env.sim_state['vehicle_traces'][current_vehicle]
                if trace:
                    last_log = trace[-1]
                    
                    # --- Travel Logic ---
                    # In this Sim, Predicted Time = Distance (1:1 ratio assumed by base logic)
                    dist = last_log['dist']
                    travel_time = last_log['arrival_time'] - (last_log['departure_time'] - last_log['service_duration'] - last_log['wait_time'] - last_log['transit_cost']/1.5) 
                    # Note: Extracting exact travel time from trace is tricky, let's use the knowns:
                    # Arrival = Prev_Time + Travel_Time. 
                    # We don't have Prev_Time easily in trace, but we can infer:
                    # Error = max(0, Actual - Predicted)
                    
                    # Simpler approach: Use the explicit math from Simulator/DualBayesianEstimator
                    # The simulator adds lognormal noise to distance.
                    # Error = Actual - Mean. 
                    # In VRPEnv, we don't return the raw noise, but we can look at the aggregates.
                    pass 

            done = terminated or truncated

    # 3. Direct Sampling from Simulator Logic
    # Since extracting "pure" travel time from Gym trace is messy due to waiting/service logic,
    # let's just use the StochasticSampler directly to generate clean data. 
    # This is statistically identical to running the env.
    
    from simulator import StochasticSampler
    
    print("  Collecting samples from StochasticSampler...")
    
    for _ in range(5000): # 5000 samples
        # Travel
        dist = np.random.uniform(5, 50)
        actual_travel = StochasticSampler.sample_travel_time(dist)
        # Estimator assumes Predict = Dist.
        t_err = max(0, actual_travel - dist)
        travel_errors.append(t_err)
        
        # Service
        mean_svc = np.random.uniform(10, 30)
        actual_svc = StochasticSampler.sample_service_time(mean_svc)
        s_err = max(0, actual_svc - mean_svc)
        service_errors.append(s_err)
        
    # 4. Fit Parameters
    # Logic from DualBayesianEstimator:
    # alpha += 1
    # beta += error / 5.0
    # Long term convergence: Mean Error ~= 5.0 * (beta / alpha)
    # Therefore: beta_optimal = (alpha * Mean_Error) / 5.0
    
    mean_travel_err = np.mean(travel_errors)
    mean_service_err = np.mean(service_errors)
    
    print(f"  Mean Travel Delay (Error): {mean_travel_err:.4f} min")
    print(f"  Mean Service Delay (Error): {mean_service_err:.4f} min")
    
    # We pick a "Confidence" level (Alpha). 
    # Higher Alpha = Stronger prior (solver trusts these values more vs new data).
    # A value of 100 is equivalent to having observed 100 samples.
    fixed_alpha = 100.0 
    
    calc_travel_beta = (fixed_alpha * mean_travel_err) / 5.0
    calc_service_beta = (fixed_alpha * mean_service_err) / 5.0
    
    # Ensure they aren't zero
    calc_travel_beta = max(1.0, calc_travel_beta)
    calc_service_beta = max(1.0, calc_service_beta)

    # 5. Load existing or create new config
    params = {}
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r') as f:
            params = json.load(f)
            
    # Update only the alpha/beta, keep other tuned params (like pruning)
    params['travel_alpha'] = fixed_alpha
    params['travel_beta'] = round(calc_travel_beta, 4)
    params['service_alpha'] = fixed_alpha
    params['service_beta'] = round(calc_service_beta, 4)
    
    print("\n--- Calibrated Parameters ---")
    print(json.dumps(params, indent=4))
    
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(params, f, indent=4)
    print(f"\nSaved to {OUTPUT_FILE}")

if __name__ == "__main__":
    run_calibration()