import time
import sys
import os

# Ensure the current directory is in the python path so imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    print("\n" + "="*60)
    print("STARTING BATCH MCTS EXECUTION (Direct Import Mode)")
    print("="*60 + "\n")
    
    total_start = time.time()
    
    # --- 1. Basic MCTS ---
    print("\n>>> [1/3] Running Basic MCTS...")
    try:
        from mcts_solver import run_mcts_pipeline as run_basic
        run_basic()
        print(">>> [SUCCESS] Basic MCTS finished.")
    except Exception as e:
        print(f">>> [FAILURE] Basic MCTS crashed: {e}")

    # --- 2. Improved MCTS ---
    print("\n>>> [2/3] Running Improved MCTS...")
    try:
        # We import inside the function or loop to ensure fresh loading if needed, 
        # but standard import at top is fine too. 
        # Aliasing prevents name collision.
        from mcts_improved_solver import run_mcts_pipeline as run_improved
        run_improved()
        print(">>> [SUCCESS] Improved MCTS finished.")
    except Exception as e:
        print(f">>> [FAILURE] Improved MCTS crashed: {e}")

    # --- 3. Hybrid MCTS ---
    print("\n>>> [3/3] Running Hybrid MCTS (ADP-Guided)...")
    try:
        from mcts_hybrid_solver import run_mcts_pipeline as run_hybrid
        run_hybrid()
        print(">>> [SUCCESS] Hybrid MCTS finished.")
    except Exception as e:
        print(f">>> [FAILURE] Hybrid MCTS crashed: {e}")

    total_elapsed = time.time() - total_start
    
    print(f"\n{'#'*60}")
    print(f"BATCH COMPLETE")
    print(f"Total Execution Time: {total_elapsed/60:.2f} minutes")
    print(f"{'#'*60}")

if __name__ == "__main__":
    main()