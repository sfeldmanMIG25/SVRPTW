import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import json
import random
import time
from collections import deque
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
EVAL_FREQ = 100

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'instances', 'data')
SAVE_DIR = os.path.join(SCRIPT_DIR, 'solutions', 'RL')
os.makedirs(SAVE_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"RL Trainer using device: {device}")

# --- Q-NETWORK ---
class VRP_DQN(nn.Module):
    def __init__(self, global_input_dim=2, node_input_dim=5, hidden_dim=128):
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
        batch_size = global_feats.size(0)
        num_nodes = node_feats.size(1)
        
        g_emb = self.global_net(global_feats)
        n_emb = self.node_net(node_feats)
        
        g_emb_expanded = g_emb.unsqueeze(1).expand(-1, num_nodes, -1)
        combined = torch.cat([g_emb_expanded, n_emb], dim=2)
        
        q_values = self.scorer(combined)
        return q_values.squeeze(2)

# --- REPLAY MEMORY ---
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, global_s, node_s, mask, action, reward, global_next, node_next, mask_next, done):
        self.memory.append((global_s, node_s, mask, action, reward, global_next, node_next, mask_next, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# --- TRAINER ---
def train():
    env = VRPEnv(DATA_DIR)
    
    policy_net = VRP_DQN().to(device)
    target_net = VRP_DQN().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = ReplayMemory(MEMORY_SIZE)
    
    steps_done = 0
    
    print("--- Starting Training ---")
    
    for i_episode in range(NUM_EPISODES):
        # Gym reset returns (obs, info)
        obs, _ = env.reset()
        
        state_g = torch.tensor(obs['global'], dtype=torch.float, device=device).unsqueeze(0)
        
        raw_nodes = obs['nodes']
        raw_mask = obs['mask']
        
        def pad_to_fixed(nodes, mask, limit=200):
            p_nodes = np.zeros((limit, 5), dtype=np.float32)
            p_mask = np.zeros(limit, dtype=bool)
            p_nodes[:nodes.shape[0], :] = nodes
            p_mask[:mask.shape[0]] = mask
            return p_nodes, p_mask

        p_nodes, p_mask = pad_to_fixed(raw_nodes, raw_mask)
        state_n = torch.tensor(p_nodes, dtype=torch.float, device=device).unsqueeze(0)
        state_m = torch.tensor(p_mask, dtype=torch.bool, device=device).unsqueeze(0)

        total_reward = 0
        
        while True:
            # SELECT ACTION
            eps = EPS_END + (EPS_START - EPS_END) * \
                  np.exp(-1. * steps_done / EPS_DECAY)
            steps_done += 1
            
            valid_indices = torch.nonzero(state_m[0]).flatten()
            
            if random.random() < eps:
                if len(valid_indices) == 0: action = 0
                else: action = valid_indices[random.randint(0, len(valid_indices)-1)].item()
            else:
                with torch.no_grad():
                    q_vals = policy_net(state_g, state_n)
                    q_vals[~state_m] = -float('inf')
                    action = q_vals.max(1)[1].item()
            
            # STEP (Gymnasium style)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            total_reward += reward
            
            n_raw_nodes = next_obs['nodes']
            n_raw_mask = next_obs['mask']
            pn_nodes, pn_mask = pad_to_fixed(n_raw_nodes, n_raw_mask)
            
            memory.push(
                obs['global'], p_nodes, p_mask, 
                action, reward, 
                next_obs['global'], pn_nodes, pn_mask, 
                done
            )
            
            obs = next_obs
            p_nodes, p_mask = pn_nodes, pn_mask
            state_g = torch.tensor(obs['global'], dtype=torch.float, device=device).unsqueeze(0)
            state_n = torch.tensor(p_nodes, dtype=torch.float, device=device).unsqueeze(0)
            state_m = torch.tensor(p_mask, dtype=torch.bool, device=device).unsqueeze(0)
            
            # OPTIMIZE
            if len(memory) > BATCH_SIZE:
                transitions = memory.sample(BATCH_SIZE)
                batch = list(zip(*transitions))
                
                b_glo = torch.tensor(np.array(batch[0]), dtype=torch.float, device=device)
                b_nod = torch.tensor(np.array(batch[1]), dtype=torch.float, device=device)
                b_msk = torch.tensor(np.array(batch[2]), dtype=torch.bool, device=device)
                b_act = torch.tensor(batch[3], dtype=torch.long, device=device).unsqueeze(1)
                b_rew = torch.tensor(batch[4], dtype=torch.float, device=device).unsqueeze(1)
                b_glo_next = torch.tensor(np.array(batch[5]), dtype=torch.float, device=device)
                b_nod_next = torch.tensor(np.array(batch[6]), dtype=torch.float, device=device)
                b_msk_next = torch.tensor(np.array(batch[7]), dtype=torch.bool, device=device)
                b_done = torch.tensor(batch[8], dtype=torch.float, device=device).unsqueeze(1)
                
                curr_q = policy_net(b_glo, b_nod).gather(1, b_act)
                
                with torch.no_grad():
                    next_q_vals = target_net(b_glo_next, b_nod_next)
                    # Don't choose invalid actions for next state max
                    next_q_vals[~b_msk_next] = -float('inf')
                    max_next_q = next_q_vals.max(1)[0].unsqueeze(1)
                    expected_q = b_rew + (GAMMA * max_next_q * (1 - b_done))
                
                loss = nn.MSELoss()(curr_q, expected_q)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            if done:
                break
                
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
            
        if i_episode % EVAL_FREQ == 0:
            print(f"Episode {i_episode} | Reward: {total_reward:.1f} | Epsilon: {eps:.2f}")
            
    torch.save(policy_net.state_dict(), os.path.join(SAVE_DIR, 'vrp_dqn.pth'))
    print("Training Complete. Model saved.")
    
    evaluate(policy_net, env)

def evaluate(model, env):
    print("\n--- Evaluating RL Policy on All Instances ---")
    model.eval()
    
    # FIX: Create results directory for JSONs
    results_dir = os.path.join(SAVE_DIR, 'simulation_results')
    os.makedirs(results_dir, exist_ok=True)

    summary = {'cost': [], 'missed': [], 'util': []}
    
    for i in range(len(env.all_instances)):
        instance_costs = []
        instance_missed = []
        instance_util = []
        
        for _ in range(10):
            obs, _ = env.reset(instance_idx=i)
            
            def pad_to_fixed(nodes, mask, limit=200):
                p_nodes = np.zeros((limit, 5), dtype=np.float32)
                p_mask = np.zeros(limit, dtype=bool)
                p_nodes[:nodes.shape[0], :] = nodes
                p_mask[:mask.shape[0]] = mask
                return p_nodes, p_mask

            p_nodes, p_mask = pad_to_fixed(obs['nodes'], obs['mask'])
            
            state_g = torch.tensor(obs['global'], dtype=torch.float, device=device).unsqueeze(0)
            state_n = torch.tensor(p_nodes, dtype=torch.float, device=device).unsqueeze(0)
            state_m = torch.tensor(p_mask, dtype=torch.bool, device=device).unsqueeze(0)
            
            while True:
                with torch.no_grad():
                    q_vals = model(state_g, state_n)
                    q_vals[~state_m] = -float('inf')
                    action = q_vals.max(1)[1].item()
                
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                if done:
                    break
                
                p_nodes, p_mask = pad_to_fixed(obs['nodes'], obs['mask'])
                state_g = torch.tensor(obs['global'], dtype=torch.float, device=device).unsqueeze(0)
                state_n = torch.tensor(p_nodes, dtype=torch.float, device=device).unsqueeze(0)
                state_m = torch.tensor(p_mask, dtype=torch.bool, device=device).unsqueeze(0)
            
            s = env.sim_state
            cap = len(s['vehicle_queue']) * env.time_horizon
            
            instance_costs.append(s['total_cost'])
            instance_missed.append(s['failures'])
            instance_util.append(s['service_time'] / max(1, cap))

        # Calculate means for this instance
        mean_cost = np.mean(instance_costs)
        mean_missed = np.mean(instance_missed)
        mean_util = np.mean(instance_util)
        
        # Add to global summary
        summary['cost'].append(mean_cost)
        summary['missed'].append(mean_missed)
        summary['util'].append(mean_util)
        
        # FIX: Save JSON for Aggregator
        instance_filename = os.path.basename(env.instance_files[i])
        base_name = instance_filename.replace('.json', '')
        
        result_data = {
            'instance_file': instance_filename,
            'policy_type': 'RL_DQN',
            'metrics': {
                'mean_total_cost': mean_cost,
                'mean_missed_customers': mean_missed,
                'fleet_utilization': mean_util
            }
        }
        
        with open(os.path.join(results_dir, f"{base_name}_rl_results.json"), 'w') as f:
            json.dump(result_data, f, indent=4)

    print(f"Avg Cost: ${np.mean(summary['cost']):.2f}")
    print(f"Avg Missed: {np.mean(summary['missed']):.2f}")
    print(f"Avg Util: {np.mean(summary['util'])*100:.1f}%")
    print(f"Results saved to: {results_dir}")

if __name__ == "__main__":
    train()