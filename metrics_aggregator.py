import os
import json
import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
DEPOT_OPEN_DURATION = 480 
HARD_LATE_PENALTY = 1000.0 # Define penalty here for standardization

def find_results_files(base_dir):
    """
    Locates result files for ORTools, Greedy, ADP, Genetic, MCTS (all variants), and RL.
    """
    paths = {
        'ORTools': [],
        'Greedy': [],
        'ADP': [],
        'Genetic': [],
        'MCTS': [],
        'MCTS_Improved': [],
        'MCTS_Hybrid': [],
        'RL': []
    }
    
    # 1. ORTools Paths
    or_path = os.path.join(base_dir, 'solutions', 'ORTools', 'simulation_results')
    if os.path.exists(or_path):
        paths['ORTools'] = glob.glob(os.path.join(or_path, '*.json'))
        
    # 2. Greedy Paths
    greedy_path = os.path.join(base_dir, 'solutions', 'Greedy', 'simulation_results')
    if os.path.exists(greedy_path):
        paths['Greedy'] = glob.glob(os.path.join(greedy_path, '*.json'))
        
    # 3. ADP Paths
    adp_path_1 = os.path.join(base_dir, 'solutions', 'ADP', 'simulation_results')
    adp_path_2 = os.path.join(base_dir, 'instances', 'solutions', 'ADP', 'simulation_results')
    if os.path.exists(adp_path_1): paths['ADP'] += glob.glob(os.path.join(adp_path_1, '*.json'))
    if os.path.exists(adp_path_2): paths['ADP'] += glob.glob(os.path.join(adp_path_2, '*.json'))
    paths['ADP'] = list(set(paths['ADP']))

    # 4. Genetic Paths
    genetic_path = os.path.join(base_dir, 'solutions', 'Genetic', 'simulation_results')
    if os.path.exists(genetic_path):
        paths['Genetic'] = glob.glob(os.path.join(genetic_path, '*.json'))

    # 5. MCTS (Basic) Paths
    mcts_path = os.path.join(base_dir, 'solutions', 'MCTS', 'simulation_results')
    if os.path.exists(mcts_path):
        paths['MCTS'] = glob.glob(os.path.join(mcts_path, '*.json'))

    # 6. MCTS Improved Paths
    mcts_imp_path = os.path.join(base_dir, 'solutions', 'MCTS_Improved', 'simulation_results')
    if os.path.exists(mcts_imp_path):
        paths['MCTS_Improved'] = glob.glob(os.path.join(mcts_imp_path, '*.json'))

    # 7. MCTS Hybrid Paths
    mcts_hyb_path = os.path.join(base_dir, 'solutions', 'MCTS_Hybrid', 'simulation_results')
    if os.path.exists(mcts_hyb_path):
        paths['MCTS_Hybrid'] = glob.glob(os.path.join(mcts_hyb_path, '*.json'))

    # 8. RL Paths
    rl_path = os.path.join(base_dir, 'solutions', 'RL', 'simulation_results') 
    if os.path.exists(rl_path):
        paths['RL'] = glob.glob(os.path.join(rl_path, '*.json'))
    
    return paths

def parse_results(method_name, file_list):
    records = []
    print(f"Parsing {len(file_list)} files for {method_name}...")
    
    for filepath in file_list:
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            metrics = data.get('metrics', {})
            
            # --- Standardize Fields ---
            fname = os.path.basename(filepath)
            instance_id = fname.replace('.json', '').replace('_stochastic_results', '')\
                               .replace('_greedy_results', '').replace('_adp_results', '')\
                               .replace('_mcts_results', '').replace('_mcts_improved_results', '')\
                               .replace('_mcts_hybrid_results', '').replace('_rl_results', '')
            
            # Raw Operational Cost (Fuel + Wages + Late Fines)
            raw_cost = metrics.get('mean_stochastic_cost', metrics.get('mean_total_cost', 0.0))
            
            # Failures
            missed = metrics.get('mean_missed_customers', 0.0)
            late = metrics.get('mean_hard_late_penalties', 0.0)
            total_failures = missed + late
            
            # --- FIX: STANDARDIZED TRUE COST CALCULATION ---
            # True Cost = Operational Cost + (Missed Customers * $1000)
            # We assume 'raw_cost' does NOT include the missed penalty (based on your last instruction).
            # We add it here to make the comparison fair across all methods.
            true_total_cost = raw_cost + (missed * HARD_LATE_PENALTY)
            
            # Utilization
            utilization = 0.0
            if 'fleet_utilization' in metrics:
                utilization = metrics['fleet_utilization']
            else:
                efficiency = metrics.get('mean_service_efficiency', 0.0)
                transit = metrics.get('mean_transit_min', 0.0)
                est_service_time = efficiency * transit
                num_vehicles = data.get('V', 0)
                if num_vehicles > 0:
                    capacity_time = num_vehicles * DEPOT_OPEN_DURATION
                    utilization = est_service_time / capacity_time
            
            records.append({
                'Instance': instance_id,
                'Method': method_name,
                'Avg_Cost': true_total_cost, # Use the corrected cost
                'Avg_Failures': total_failures,
                'Utilization_Pct': utilization * 100.0, 
                'N_Customers': data.get('N', 0),
                'N_Vehicles': data.get('V', 0)
            })
            
        except Exception as e:
            print(f"Error parsing {filepath}: {e}")
            
    return records

def generate_report():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Scanning for results in: {script_dir}")
    file_paths = find_results_files(script_dir)
    
    all_records = []
    for method, files in file_paths.items():
        if files:
            all_records.extend(parse_results(method, files))
        else:
            print(f"Warning: No result files found for {method}")
            
    if not all_records:
        print("No results found to aggregate.")
        return

    df = pd.DataFrame(all_records)
    
    # --- Summary Table ---
    summary = df.groupby('Method').agg({
        'Avg_Cost': 'mean',
        'Avg_Failures': 'mean',
        'Utilization_Pct': 'mean',
        'Instance': 'count'
    }).rename(columns={'Instance': 'Count'}).reset_index()
    
    print("\n" + "="*60)
    print("AGGREGATE PERFORMANCE SUMMARY (Adjusted for Missed Penalties)")
    print("="*60)
    print(summary.to_string(index=False, float_format=lambda x: "{:.2f}".format(x)))
    print("="*60)
    
    output_dir = os.path.join(script_dir, 'solutions')
    os.makedirs(output_dir, exist_ok=True)
    
    summary.to_csv(os.path.join(output_dir, 'benchmark_method_summary.csv'), index=False)
    df.to_csv(os.path.join(output_dir, 'benchmark_full_details.csv'), index=False)
    
    generate_charts(df, output_dir)

def generate_charts(df, output_dir):
    plt.style.use('bmh') 
    
    methods = df['Method'].unique()
    colors = {
        'ORTools': '#1f77b4',      # Blue
        'Greedy': '#ff7f0e',       # Orange
        'ADP': '#2ca02c',          # Green
        'Genetic': '#9467bd',      # Purple
        'MCTS': '#d62728',         # Red
        'MCTS_Improved': '#e377c2',# Pink
        'MCTS_Hybrid': '#bcbd22',  # Olive/Gold
        'RL': '#17becf'            # Cyan
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Chart 1: Average Cost
    avg_cost = df.groupby('Method')['Avg_Cost'].mean()
    axes[0].bar(avg_cost.index, avg_cost.values, color=[colors.get(m, 'gray') for m in avg_cost.index])
    axes[0].set_title('True Avg Cost ($)\n(Ops + Missed Penalty)')
    axes[0].set_ylabel('Cost')
    plt.setp(axes[0].get_xticklabels(), rotation=45, ha='right')
    
    # Chart 2: Failures
    avg_fail = df.groupby('Method')['Avg_Failures'].mean()
    axes[1].bar(avg_fail.index, avg_fail.values, color=[colors.get(m, 'gray') for m in avg_fail.index])
    axes[1].set_title('Avg Failures (Missed + Late)')
    axes[1].set_ylabel('Count')
    plt.setp(axes[1].get_xticklabels(), rotation=45, ha='right')
    
    # Chart 3: Utilization
    avg_util = df.groupby('Method')['Utilization_Pct'].mean()
    axes[2].bar(avg_util.index, avg_util.values, color=[colors.get(m, 'gray') for m in avg_util.index])
    axes[2].set_title('Fleet Utilization (%)')
    axes[2].set_ylabel('Percentage')
    plt.setp(axes[2].get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'benchmark_charts.png'))
    plt.close()
    print(f"Charts saved to: {os.path.join(output_dir, 'benchmark_charts.png')}")

if __name__ == "__main__":
    generate_report()