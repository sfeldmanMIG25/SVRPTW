import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import json
import random
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from vrp_gym_env import VRPEnv

# --- CONFIG ---
BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 5000
TARGET_UPDATE = 10
MEMORY_SIZE = 50000
LR = 1e-4
NUM_EPISODES = 2000
EVAL_FREQ = 250
EVAL_SIMS = 30 # Number of stochastic days per instance

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'instances', 'data')
SAVE_DIR = os.path.join(SCRIPT_DIR, 'solutions', 'RL')
RESULTS_DIR = os.path.join(SAVE_DIR, 'simulation_results')
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Q-NETWORK ---
class VRP_DQN(nn.Module):
    def __init__(self, global_input_dim=2, node_input_dim=6, hidden_dim=128):
        super(VRP_DQN, self).__init__()
        
        self.global_net = nn.Sequential(
            nn.Linear(global_input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        
        self.node_net = nn.Sequential(
            nn.Linear(node_input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        
        self.scorer = nn.Sequential(
            nn.Linear(32 + 32, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1) 
        )
        
    def forward(self, global_feats, node_feats):
        # global: (Batch, 2)
        # nodes: (Batch, N, 6)
        batch_size = global_feats.size(0)
        num_nodes = node_feats.size(1)
        
        g_emb = self.global_net(global_feats)
        n_emb = self.node_net(node_feats)
        
        g_emb_expanded = g_emb.unsqueeze(1).expand(-1, num_nodes, -1)
        combined = torch.cat([g_emb_expanded, n_emb], dim=2)
        
        q_values = self.scorer(combined)
        return q_values.squeeze(2)

# --- WORKER FOR PARALLEL EVALUATION ---
def evaluate_worker(instance_idx, model_state, data_dir):
    """
    Runs 30 episodes for a single instance using a CPU-loaded model.
    """
    try:
        # Re-init env and model locally to avoid pickling issues
        env = VRPEnv(data_dir)
        local_net = VRP_DQN().to('cpu')
        local_net.load_state_dict(model_state)
        local_net.eval()
        
        logs = []
        
        for day in range(EVAL_SIMS):
            obs, _ = env.reset(instance_idx=instance_idx)
            
            done = False
            while not done:
                # Prepare Inputs
                raw_nodes = obs['nodes']
                raw_mask = obs['mask']
                
                # Pad
                limit = 200
                p_nodes = np.zeros((limit, 6), dtype=np.float32)
                p_mask = np.zeros(limit, dtype=bool)
                p_nodes[:raw_nodes.shape[0], :] = raw_nodes
                p_mask[:raw_mask.shape[0]] = raw_mask
                
                # Inference
                g_tens = torch.tensor(obs['global'], dtype=torch.float).unsqueeze(0)
                n_tens = torch.tensor(p_nodes, dtype=torch.float).unsqueeze(0)
                m_tens = torch.tensor(p_mask, dtype=torch.bool).unsqueeze(0)
                
                with torch.no_grad():
                    q_vals = local_net(g_tens, n_tens)
                    q_vals[~m_tens] = -float('inf')
                    action = q_vals.max(1)[1].item()
                    
                obs, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
            # Capture Log
            s = env.sim_state
            logs.append({
                'day_index': day,
                'total_cost': s['total_cost'],
                'hard_lates': s['hard_lates'],
                'missed_customers': len(s['unvisited_ids']),
                'vehicle_traces': s['vehicle_traces']
            })
            
        # Instance Meta
        inst_file = os.path.basename(env.instance_files[instance_idx])
        num_c = env.all_instances[instance_idx]['num_customers']
        num_v = env.all_instances[instance_idx]['num_vehicles']
        
        return {
            'instance_file': inst_file,
            'N': num_c,
            'V': num_v,
            'daily_simulation_logs': logs
        }
        
    except Exception as e:
        return {'error': str(e)}

# --- TRAINER ---
def train():
    env = VRPEnv(DATA_DIR)
    
    policy_net = VRP_DQN().to(device)
    target_net = VRP_DQN().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    
    # Simple deque memory (can be optimized, but fine for now)
    memory = deque(maxlen=MEMORY_SIZE)
    
    print(f"--- Starting DQN Training ({NUM_EPISODES} episodes) ---")
    
    steps_done = 0
    
    for i_episode in range(NUM_EPISODES):
        obs, _ = env.reset()
        
        # Preprocessing function for padding
        def pad_state(obs):
            rn = obs['nodes']
            rm = obs['mask']
            p_n = np.zeros((200, 6), dtype=np.float32)
            p_m = np.zeros(200, dtype=bool)
            p_n[:rn.shape[0], :] = rn
            p_m[:rm.shape[0]] = rm
            return obs['global'], p_n, p_m

        cur_g, cur_n, cur_m = pad_state(obs)
        
        total_reward = 0
        
        while True:
            # Epsilon Greedy
            eps = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * steps_done / EPS_DECAY)
            steps_done += 1
            
            valid_indices = np.nonzero(cur_m)[0]
            
            if random.random() < eps:
                if len(valid_indices) == 0: action = 0
                else: action = np.random.choice(valid_indices)
            else:
                t_g = torch.tensor(cur_g, dtype=torch.float, device=device).unsqueeze(0)
                t_n = torch.tensor(cur_n, dtype=torch.float, device=device).unsqueeze(0)
                t_m = torch.tensor(cur_m, dtype=torch.bool, device=device).unsqueeze(0)
                with torch.no_grad():
                    q = policy_net(t_g, t_n)
                    q[~t_m] = -float('inf')
                    action = q.max(1)[1].item()
            
            next_obs, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            total_reward += reward
            
            nex_g, nex_n, nex_m = pad_state(next_obs)
            
            # Store
            memory.append((cur_g, cur_n, cur_m, action, reward, nex_g, nex_n, nex_m, done))
            
            cur_g, cur_n, cur_m = nex_g, nex_n, nex_m
            
            # Optimize
            if len(memory) > BATCH_SIZE:
                batch = random.sample(memory, BATCH_SIZE)
                
                # Unpack
                b_g, b_n, b_m, b_a, b_r, b_ng, b_nn, b_nm, b_d = zip(*batch)
                
                t_g = torch.tensor(np.array(b_g), dtype=torch.float, device=device)
                t_n = torch.tensor(np.array(b_n), dtype=torch.float, device=device)
                t_a = torch.tensor(b_a, dtype=torch.long, device=device).unsqueeze(1)
                t_r = torch.tensor(b_r, dtype=torch.float, device=device).unsqueeze(1)
                t_ng = torch.tensor(np.array(b_ng), dtype=torch.float, device=device)
                t_nn = torch.tensor(np.array(b_nn), dtype=torch.float, device=device)
                t_nm = torch.tensor(np.array(b_nm), dtype=torch.bool, device=device)
                t_d = torch.tensor(b_d, dtype=torch.float, device=device).unsqueeze(1)
                
                curr_q = policy_net(t_g, t_n).gather(1, t_a)
                
                with torch.no_grad():
                    next_q = target_net(t_ng, t_nn)
                    next_q[~t_nm] = -float('inf')
                    max_next = next_q.max(1)[0].unsqueeze(1)
                    target = t_r + (GAMMA * max_next * (1 - t_d))
                
                loss = nn.MSELoss()(curr_q, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            if done: break
        
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
            
        if (i_episode+1) % 100 == 0:
            print(f"  Episode {i_episode+1} | Last Reward: {total_reward:.1f} | Eps: {eps:.2f}")

        if (i_episode+1) % EVAL_FREQ == 0:
            # Save Checkpoint
            torch.save(policy_net.state_dict(), os.path.join(SAVE_DIR, 'vrp_dqn.pth'))
            # Run Parallel Evaluation
            run_parallel_evaluation(policy_net)
            
    # Final Save
    torch.save(policy_net.state_dict(), os.path.join(SAVE_DIR, 'vrp_dqn.pth'))
    print("Training Complete.")
    
    # Final Eval
    run_parallel_evaluation(policy_net)

def run_parallel_evaluation(policy_net):
    print("\n--- Running Parallel Evaluation ---")
    
    # Get model state for CPU workers
    cpu_state = {k: v.cpu() for k, v in policy_net.state_dict().items()}
    
    # Get file list
    instance_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.json')])
    if not instance_files: return
    
    total_costs = []
    
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {
            executor.submit(evaluate_worker, i, cpu_state, DATA_DIR): i 
            for i in range(len(instance_files))
        }
        
        for future in as_completed(futures):
            res = future.result()
            if 'error' in res:
                print(f"Eval Error: {res['error']}")
                continue
            
            # Save JSON
            fname = res['instance_file'].replace('.json', '')
            out_path = os.path.join(RESULTS_DIR, f"{fname}_rl_results.json")
            
            # Construct Final Output
            final_json = {
                'instance_file': res['instance_file'],
                'policy_type': 'RL_DQN',
                'N': res['N'],
                'V': res['V'],
                'daily_simulation_logs': res['daily_simulation_logs']
            }
            
            with open(out_path, 'w') as f:
                json.dump(final_json, f, indent=4)
            
            # Track avg cost
            avg_c = np.mean([l['total_cost'] for l in res['daily_simulation_logs']])
            total_costs.append(avg_c)
            
    print(f"Evaluation Complete. Avg Cost across all instances: ${np.mean(total_costs):,.0f}")
    print(f"Results saved to: {RESULTS_DIR}\n")

if __name__ == "__main__":
    train()