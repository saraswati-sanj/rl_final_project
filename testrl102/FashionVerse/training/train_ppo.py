"""
FashionVerse — Top-level PPO Training Script
Usage:
    python training/train_ppo.py --timesteps 50000
    python training/train_ppo.py --quick         # 5000 steps for quick test
"""
import os, sys, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.rl.train import run_all_experiments

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FashionVerse PPO agent")
    parser.add_argument("--timesteps", type=int, default=50000,
                        help="Total training timesteps per experiment")
    parser.add_argument("--quick", action="store_true",
                        help="Quick 5000-step test run")
    args = parser.parse_args()

    ts = 5000 if args.quick else args.timesteps
    print(f"[train_ppo] Running with timesteps={ts}")
    run_all_experiments(timesteps=ts)
