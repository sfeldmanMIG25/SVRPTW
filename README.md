# Learning to Route Under Uncertainty: The SVRPTW

**Author:** Stephen Feldman  
**Affiliation:** Decision Making Under Uncertainty, Rensselaer Polytechnic Institute

## Overview
This repository contains the implementation and experimentation framework for the **Stochastic Vehicle Routing Problem with Time Windows (SVRPTW)**. This research investigates how different decision-making frameworks handle high-variance urban delivery environments. Unlike traditional VRP setups which assume perfect information, this environment models heavy-tailed Log-Normal travel delays and stochastic service times, forcing agents to balance operational efficiency (cost) against reliability (missed deliveries).

The project benchmarks four distinct strategies:
1. **Greedy Heuristic:** A myopic baseline.
2. **Deterministic Optimization:** Google OR-Tools (Constraint Programming).
3. **Deep Reinforcement Learning:** A Deep Q-Network (DQN) learning risk-aware policies.
4. **Bayesian Auction Mechanism:** A decentralized solver using dynamic risk buffering.

### Key Features
* **Extreme Variance Environment:** A custom generator creating topologies with tight time windows (30-90 min) and volatile travel times.
* **Stochastic Simulation:** A unified simulator that evaluates static plans against dynamic reality over repeated trials (30 days/instance).
* **Multi-Solver Benchmark:** A standardized pipeline to compare Operations Research solvers against Learning-based agents.
* **Visual Analysis:** Tools to visualize route variance and node failure rates across simulation days.

---
## Installation & Requirements
The project requires Python 3.8+ and the following dependencies:

pip install numpy pandas matplotlib torch gymnasium ortools scipy optuna tqdm
Repository Structure
text
1. Core Environment
    config.py: Central configuration for physical constants, costs ($14.50/hr wages, $0.50/mile), and stochastic distributions.
    simulator.py: The stochastic transition engine. It executes routes and injects random delays based on Log-Normal (travel) and Normal (service) distributions.
    vrp_gym_env.py: A Gymnasium wrapper converting the SVRPTW into a standard RL environment.

2. Data Generation
    data_generator.py: Generates the "Extreme" instance dataset. It creates instances (N=20 to N=100) with tight time windows designed to stress-test reliability.

3. Solvers & Policies
    Baselines:
        greedy_evaluator.py: Implements the Nearest Neighbor heuristic (Lower Bound).
        deterministic_policy_generator.py: Uses Google OR-Tools to solve the mean-value problem (static planning baseline).
        Stochastic_Evaluator.py: Evaluates the static OR-Tools plans against the stochastic simulator.
    Reinforcement Learning:
        rl_trainer.py: Trains a Deep Q-Network (DQN) to learn risk-aware routing policies using a dual-stream architecture (Global Context + Node Features).
    Auction Mechanism:
        auction_solver.py: A decentralized, market-based solver. Vehicles bid on customers using a "Risk Cost" derived from a Dual Bayesian Estimator.
        tuner.py: Uses Optuna to optimize the Bayesian priors and risk buffer thresholds.
    Experimental:
        mcts_solver.py / mcts_hybrid_solver.py: Monte Carlo Tree Search implementations (found to be computationally expensive for this state space).
        rollout_agent.py: A lookahead rollout policy.

4. Analysis
    metrics_aggregator.py: Scrapes result JSONs to compute aggregate statistics (Operational Cost, Failure Rates, Utilization).
    strategy_visualizer.py: Generates heatmaps of route variance and customer reliability.
Execution Pipeline
To reproduce the results presented in the paper, follow this execution flow:

Step 1: Generate Data
Generate the dataset of 100 "Extreme" topology instances.

bash
python data_generator.py
Outputs to: instances/data/

Step 2: Run Baselines
Greedy Heuristic:
Runs the myopic nearest-neighbor policy.

bash
python greedy_evaluator.py
Deterministic Solver (OR-Tools):
First, generate the static "optimal" plans based on expected values.

bash
python deterministic_policy_generator.py
Then, simulate these static plans against the stochastic reality (30 days per instance).

bash
python Stochastic_Evaluator.py
Step 3: Train & Run Deep Q-Network (DQN)
Train the neural network to approximate Q-values for vehicle routing.

bash
python rl_trainer.py
Note: The script automatically runs an evaluation pass after training completes.

Step 4: Run Auction Solver
Execute the decentralized bidding mechanism. This solver uses a Dual Bayesian Estimator to learn traffic noise over time.

bash
# Optional: Tune hyperparameters first (warning: computationally expensive)
# python tuner.py

# Run the solver using optimized parameters
python auction_solver.py
Step 5: Analyze Results
Aggregate Metrics:
Generates the summary table (Mean Op Cost, Failure Rates) comparing all strategies.

bash
python metrics_aggregator.py
Visualize Strategies:
Creates visual overlays showing where specific policies fail (e.g., missed customers marked in red).

bash
python strategy_visualizer.py
Outputs to: solutions/[SOLVER]/visuals/

