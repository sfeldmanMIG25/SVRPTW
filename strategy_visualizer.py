import json
import os
import re
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np

# --- HELPER FUNCTIONS ---

def get_node_coordinates(customers, node_id, depot):
    if node_id == 0:
        return depot['x'], depot['y']
    
    # Handle list of dicts (standard) or dict of dicts
    if isinstance(customers, list):
        for customer in customers:
            if customer['id'] == node_id:
                return customer['x'], customer['y']
    elif isinstance(customers, dict):
        if node_id in customers:
             return customers[node_id]['x'], customers[node_id]['y']
             
    raise ValueError(f"Node ID {node_id} not found.")

def calculate_node_reliability(instance, logs):
    total_days = len(logs)
    node_failures = {c['id']: 0 for c in instance['customers']}
    
    for log in logs:
        serviced_nodes = set()
        if 'vehicle_traces' in log:
            for trace in log['vehicle_traces']:
                for step in trace:
                    if step['node_id'] != 0:
                        if step.get('outcome', 'SUCCESS') != 'LATE_SKIP':
                            serviced_nodes.add(step['node_id'])
        
        for cid in node_failures.keys():
            if cid not in serviced_nodes:
                node_failures[cid] += 1
                
    return {cid: count / max(1, total_days) for cid, count in node_failures.items()}

def visualize_simulation_variance(instance, simulation_result, output_dir):
    logs = simulation_result.get('daily_simulation_logs', [])
    if not logs: return

    num_vehicles = instance.get('num_vehicles', 1)
    depot = instance['depot']
    customers = instance['customers']
    failure_rates = calculate_node_reliability(instance, logs)
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # PANEL 1: Route Variance
    ax1 = axes[0]
    ax1.set_title(f"Route Variance (30 Days) - {simulation_result.get('policy_type', 'Unknown')}", fontsize=14)
    ax1.set_xlabel("X Coordinate")
    ax1.set_ylabel("Y Coordinate")
    ax1.scatter(depot['x'], depot['y'], marker='s', color='black', s=300, zorder=10, label='Depot')
    
    # Plot Customers
    cust_x, cust_y = zip(*[(c['x'], c['y']) for c in customers])
    ax1.scatter(cust_x, cust_y, c='lightgray', edgecolors='gray', s=80, zorder=5)
    
    # Plot Traces (High Transparency)
    vehicle_colors = plt.cm.nipy_spectral(np.linspace(0, 0.9, num_vehicles))
    
    for log in logs:
        traces = log.get('vehicle_traces', [])
        for v_idx, trace in enumerate(traces):
            if not trace: continue
            xs, ys = [depot['x']], [depot['y']]
            for step in trace:
                x, y = get_node_coordinates(customers, step['node_id'], depot)
                xs.append(x)
                ys.append(y)
            ax1.plot(xs, ys, color=vehicle_colors[v_idx % num_vehicles], linewidth=1.5, alpha=0.08)

    ax1.set_xlim(0, 100); ax1.set_ylim(0, 100)
    ax1.grid(True, alpha=0.3)
    
    # PANEL 2: Reliability
    ax2 = axes[1]
    ax2.set_title("Customer Reliability (Failure Rate)", fontsize=14)
    ax2.scatter(depot['x'], depot['y'], marker='s', color='black', s=300, zorder=10)
    
    cmap = plt.cm.RdYlBu_r 
    norm = mcolors.Normalize(vmin=0, vmax=1.0)
    
    coords_x, coords_y, colors, sizes = [], [], [], []
    for c in customers:
        coords_x.append(c['x'])
        coords_y.append(c['y'])
        rate = failure_rates.get(c['id'], 0.0)
        colors.append(rate)
        sizes.append(c['demand'] * 5)
        
    sc = ax2.scatter(coords_x, coords_y, c=colors, cmap=cmap, norm=norm, s=sizes, edgecolors='black', alpha=0.9, zorder=5)
    
    for cid, rate in failure_rates.items():
        if rate > 0.1:
            x, y = get_node_coordinates(customers, cid, depot)
            ax2.text(x, y+1, f"{rate*100:.0f}%", fontsize=9, ha='center', fontweight='bold')

    ax2.set_xlim(0, 100); ax2.set_ylim(0, 100); ax2.grid(True, alpha=0.3)
    plt.colorbar(sc, ax=ax2, label='Failure Rate')
    
    out_name = os.path.basename(simulation_result['instance_file']).replace('.json', '') + "_variance.png"
    plt.savefig(os.path.join(output_dir, out_name), dpi=100)
    plt.close()

if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    INSTANCE_DIR = os.path.join(SCRIPT_DIR, 'instances', 'data')
    SOLUTIONS_DIR = os.path.join(SCRIPT_DIR, 'solutions')
    
    print("--- Strategy Visualizer (Batch) ---")
    
    for root, dirs, files in os.walk(SOLUTIONS_DIR):
        if 'visuals' in root: continue
        
        # Check if this folder has results
        results = [f for f in files if f.endswith('_results.json')]
        if not results: continue
            
        # Create sibling visuals folder
        visual_dir = root.replace('simulation_results', 'visuals')
        os.makedirs(visual_dir, exist_ok=True)
        
        for file in results:
            try:
                with open(os.path.join(root, file), 'r') as f:
                    res_data = json.load(f)
                
                # Find Instance
                inst_name = res_data.get('instance_file')
                if not inst_name: 
                    # Try regex from filename
                    base = file.replace('_stochastic_results.json', '').replace('_greedy_results.json', '').replace('_mcts_results.json', '')
                    inst_name = base + ".json"

                inst_path = os.path.join(INSTANCE_DIR, inst_name)
                if not os.path.exists(inst_path): continue
                    
                with open(inst_path, 'r') as f:
                    inst_data = json.load(f)
                if isinstance(inst_data['customers'], dict):
                    inst_data['customers'] = list(inst_data['customers'].values())

                visualize_simulation_variance(inst_data, res_data, visual_dir)
                print(f"Generated visual for: {file}")
                
            except Exception as e:
                print(f"Skipped {file}: {e}")