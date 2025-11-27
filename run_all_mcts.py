import time
import sys
import os

# Ensure the current directory is in the python path so imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    print("\n" + "="*60)
    print("STARTING BATCH MCTS EXECUTION")
    print("Targets: Standard MCTS + Unified Hybrid (AlphaZero-Style)")
    print("="*60 + "\n")
    
    total_start = time.time()
    
    # --- 1. Standard MCTS (Optimized Baseline) ---
    print("\n>>> [1/2] Running Standard MCTS (mcts_solver.py)...")
    try:
        # Import dynamically to ensure we pick up the latest version
        import mcts_solver
        mcts_solver.run_pipeline()
        print(">>> [SUCCESS] Standard MCTS finished.")
    except ImportError:
        print(">>> [ERROR] Could not import 'mcts_solver.py'. Check filename.")
    except Exception as e:
        print(f">>> [FAILURE] Standard MCTS crashed: {e}")

    # --- 2. Unified Hybrid MCTS (The "AlphaZero" Solver) ---
    print("\n>>> [2/2] Running Unified Hybrid MCTS (mcts_hybrid_solver.py)...")
    try:
        import mcts_hybrid_solver
        mcts_hybrid_solver.run_pipeline()
        print(">>> [SUCCESS] Unified Hybrid MCTS finished.")
    except ImportError:
        print(">>> [ERROR] Could not import 'mcts_hybrid_solver.py'. Ensure the previous code block was saved with this name.")
    except Exception as e:
        print(f">>> [FAILURE] Unified Hybrid MCTS crashed: {e}")

    total_elapsed = time.time() - total_start
    
    print(f"\n{'#'*60}")
    print(f"BATCH COMPLETE")
    print(f"Total Execution Time: {total_elapsed/60:.2f} minutes")
    print(f"{'#'*60}")

if __name__ == "__main__":
    main()