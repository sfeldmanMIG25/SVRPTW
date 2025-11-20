import os
import json
import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
# Constants for utilization calculation
DEPOT_OPEN_DURATION = 480 # 8:00 AM to 4:00 PM (Minutes)

def find_results_files(base_dir):
    """
    Locates result files for ORTools, Greedy, and ADP.
    Handles potential path variations from previous scripts.
    """
    paths = {
        'ORTools': [],
        'Greedy': [],
        'ADP': []
    }
    
    # 1. ORTools Paths
    or_path = os.path.join(base_dir, 'solutions', 'ORTools', 'simulation_results')
    if os.path.exists(or_path):
        paths['ORTools'] = glob.glob(os.path.join(or_path, '*.json'))
        
    # 2. Greedy Paths
    greedy_path = os.path.join(base_dir, 'solutions', 'Greedy', 'simulation_results')
    if os.path.exists(greedy_path):
        paths['Greedy'] = glob.glob(os.path.join(greedy_path, '*.json'))
        
    # 3. ADP Paths (Checking both root 'solutions' and 'instances/solutions' pattern)
    adp_path_1 = os.path.join(base_dir, 'solutions', 'ADP', 'simulation_results')
    adp_path_2 = os.path.join(base_dir, 'instances', 'solutions', 'ADP', 'simulation_results')
    
    if os.path.exists(adp_path_1):
        paths['ADP'] += glob.glob(os.path.join(adp_path_1, '*.json'))
    if os.path.exists(adp_path_2):
        paths['ADP'] += glob.glob(os.path.join(adp_path_2, '*.json'))
        
    # Remove duplicates if any
    paths['ADP'] = list(set(paths['ADP']))
    
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
            
            # 1. Instance Name
            # Remove method suffixes to get the clean Nxx_Vxx_Ixx ID
            fname = os.path.basename(filepath)
            instance_id = fname.replace('_stochastic_results.json', '')\
                               .replace('_greedy_results.json', '')\
                               .replace('_adp_results.json', '')
            
            # 2. Cost
            # ORTools/Greedy use 'mean_stochastic_cost', ADP uses 'mean_total_cost'
            cost = metrics.get('mean_stochastic_cost', metrics.get('mean_total_cost', 0.0))
            
            # 3. Missed Customers / Failures
            # ORTools: failures are 'hard_late_penalties' (dropped nodes or late arrivals)
            # ADP/Greedy: failures are 'missed_customers' + 'hard_late_penalties'
            missed = metrics.get('mean_missed_customers', 0.0)
            late = metrics.get('mean_hard_late_penalties', 0.0)
            total_failures = missed + late
            
            # 4. Utilization %
            # ADP calculates 'fleet_utilization' directly.
            # ORTools/Greedy provide 'mean_service_efficiency' (Service/Transit ratio) and 'mean_transit_min'.
            # We derive Service Time = Efficiency * Transit.
            # Then Utilization = Service Time / (Vehicles * 480).
            
            utilization = 0.0
            if 'fleet_utilization' in metrics:
                utilization = metrics['fleet_utilization']
            else:
                # Derivative calculation
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
                'Avg_Cost': cost,
                'Avg_Failures': total_failures,
                'Utilization_Pct': utilization * 100.0, # Store as percentage
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
    
    # --- 1. Summary Table by Method ---
    summary = df.groupby('Method').agg({
        'Avg_Cost': 'mean',
        'Avg_Failures': 'mean',
        'Utilization_Pct': 'mean',
        'Instance': 'count'
    }).rename(columns={'Instance': 'Count'}).reset_index()
    
    print("\n" + "="*60)
    print("AGGREGATE PERFORMANCE SUMMARY")
    print("="*60)
    print(summary.to_string(index=False, float_format=lambda x: "{:.2f}".format(x)))
    print("="*60)
    
    # --- 2. Detailed Comparison (Pivot) ---
    # We want to see head-to-head performance on the same instances
    pivot_df = df.pivot_table(index=['Instance', 'N_Customers'], 
                              columns='Method', 
                              values=['Avg_Cost', 'Avg_Failures'])
    
    # Flatten columns
    pivot_df.columns = [f"{col[1]}_{col[0]}" for col in pivot_df.columns]
    pivot_df = pivot_df.reset_index()
    
    # Save CSVs
    output_dir = os.path.join(script_dir, 'solutions')
    os.makedirs(output_dir, exist_ok=True)
    
    summary_csv = os.path.join(output_dir, 'benchmark_method_summary.csv')
    summary.to_csv(summary_csv, index=False)
    
    detailed_csv = os.path.join(output_dir, 'benchmark_instance_comparison.csv')
    pivot_df.to_csv(detailed_csv, index=False)
    
    print(f"\nReports saved to:\n1. {summary_csv}\n2. {detailed_csv}")

    # --- 3. Visualizations ---
    generate_charts(df, output_dir)

def generate_charts(df, output_dir):
    # Set plot style
    plt.style.use('bmh') # clean style
    
    methods = df['Method'].unique()
    colors = {'ORTools': '#1f77b4', 'Greedy': '#ff7f0e', 'ADP': '#2ca02c'}
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Chart 1: Average Cost
    avg_cost = df.groupby('Method')['Avg_Cost'].mean()
    axes[0].bar(avg_cost.index, avg_cost.values, color=[colors.get(m, 'gray') for m in avg_cost.index])
    axes[0].set_title('Average Operational Cost ($)')
    axes[0].set_ylabel('Cost')
    
    # Chart 2: Failures (Missed + Late)
    avg_fail = df.groupby('Method')['Avg_Failures'].mean()
    axes[1].bar(avg_fail.index, avg_fail.values, color=[colors.get(m, 'gray') for m in avg_fail.index])
    axes[1].set_title('Avg Failures (Missed + Late)')
    axes[1].set_ylabel('Count')
    
    # Chart 3: Utilization
    avg_util = df.groupby('Method')['Utilization_Pct'].mean()
    axes[2].bar(avg_util.index, avg_util.values, color=[colors.get(m, 'gray') for m in avg_util.index])
    axes[2].set_title('Fleet Utilization (%)')
    axes[2].set_ylabel('Percentage')
    axes[2].set_ylim(0, 100)
    
    plt.tight_layout()
    chart_path = os.path.join(output_dir, 'benchmark_charts.png')
    plt.savefig(chart_path)
    print(f"Charts saved to: {chart_path}")
    plt.close()

if __name__ == "__main__":
    generate_report()