"""
FashionVerse — Training Pipeline
Full training script: PPO + DQN + Baselines + Experiments 1-4.
Run: python backend/rl/train.py
"""

import os, sys, json, time, random
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from backend.rl.fashion_env import FashionEnv
from backend.rl.ppo_agent import FashionPPOAgent
from backend.rl.dqn_baseline import FashionDQNAgent
from backend.rl.reward import RewardConfig
from backend.fashion.constraints import FashionConstraints
from backend.fashion.outfit_generator import OutfitGenerator

SEED = 42
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "experiments", "results")
MODELS_DIR  = os.path.join(os.path.dirname(__file__), "..", "..", "models")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


# ── Random Baseline ───────────────────────────────────────────────────────────

def evaluate_random_baseline(n_episodes=200, seed=SEED):
    """Uniform random valid-action selection — lower bound baseline."""
    env = FashionEnv(seed=seed, max_steps=8)
    rewards, feedbacks = [], []
    rng = random.Random(seed)
    for ep in range(n_episodes):
        env.reset()
        done = False
        ep_r, ep_f = 0, "neutral"
        while not done:
            action = env.action_space_.sample_valid_action(rng)
            _, r, terminated, truncated, info = env.step(action)
            ep_r += r
            ep_f = info.get("feedback", ep_f)
            done = terminated or truncated
        rewards.append(ep_r)
        feedbacks.append(ep_f)

    n = len(feedbacks)
    pos = sum(1 for f in feedbacks if f in ("love", "like", "save", "purchase"))
    result = {
        "agent": "random",
        "mean_reward": float(np.mean(rewards)),
        "std_reward":  float(np.std(rewards)),
        "acceptance_rate": pos / n,
        "feedback_distribution": {
            f: feedbacks.count(f) for f in
            ["love","like","neutral","skip","dislike","save","purchase"]
        },
    }
    print(f"[Random]   mean_reward={result['mean_reward']:.3f} | "
          f"acceptance={result['acceptance_rate']:.3f}")
    return result, rewards


# ── Rule-Based Baseline ───────────────────────────────────────────────────────

def evaluate_rule_based_baseline(n_episodes=200, seed=SEED):
    """
    Rule-based outfit generator as baseline.
    Uses greedy compatibility selection — no learning involved.
    """
    env = FashionEnv(seed=seed, max_steps=8)
    gen = OutfitGenerator()
    rewards, feedbacks = [], []

    rng = random.Random(seed)
    for ep in range(n_episodes):
        obs, info = env.reset()
        c = env._constraints
        outfit = gen.generate_rule_based(c, seed=ep)

        # Simulate presenting the outfit to the user
        if env._user and outfit:
            feedback, satisfaction = env._user.evaluate_outfit(outfit, c.occasion)
            from backend.rl.reward import RewardCalculator
            calc = RewardCalculator()
            r, _ = calc.terminal_reward(
                feedback, outfit, c,
                env._profile.recently_recommended if env._profile else [],
                c.occasion,
            )
        else:
            feedback, r = "skip", -2.0

        rewards.append(r)
        feedbacks.append(feedback)

    n = len(feedbacks)
    pos = sum(1 for f in feedbacks if f in ("love", "like", "save", "purchase"))
    result = {
        "agent": "rule_based",
        "mean_reward": float(np.mean(rewards)),
        "std_reward":  float(np.std(rewards)),
        "acceptance_rate": pos / n,
        "feedback_distribution": {
            f: feedbacks.count(f) for f in
            ["love","like","neutral","skip","dislike","save","purchase"]
        },
    }
    print(f"[RuleBased] mean_reward={result['mean_reward']:.3f} | "
          f"acceptance={result['acceptance_rate']:.3f}")
    return result, rewards


# ── Popularity Baseline ───────────────────────────────────────────────────────

def evaluate_popularity_baseline(n_episodes=200, seed=SEED):
    """Popularity-based selection — picks highest-scored items."""
    env = FashionEnv(seed=seed, max_steps=8)
    gen = OutfitGenerator()
    rewards, feedbacks = [], []

    for ep in range(n_episodes):
        obs, info = env.reset()
        c = env._constraints
        outfit = gen.generate_popular(c)

        if env._user and outfit:
            feedback, _ = env._user.evaluate_outfit(outfit, c.occasion)
            from backend.rl.reward import RewardCalculator
            calc = RewardCalculator()
            r, _ = calc.terminal_reward(
                feedback, outfit, c,
                env._profile.recently_recommended if env._profile else [],
                c.occasion,
            )
        else:
            feedback, r = "skip", -2.0

        rewards.append(r)
        feedbacks.append(feedback)

    n = len(feedbacks)
    pos = sum(1 for f in feedbacks if f in ("love", "like", "save", "purchase"))
    result = {
        "agent": "popularity",
        "mean_reward": float(np.mean(rewards)),
        "std_reward":  float(np.std(rewards)),
        "acceptance_rate": pos / n,
        "feedback_distribution": {
            f: feedbacks.count(f) for f in
            ["love","like","neutral","skip","dislike","save","purchase"]
        },
    }
    print(f"[Popular]  mean_reward={result['mean_reward']:.3f} | "
          f"acceptance={result['acceptance_rate']:.3f}")
    return result, rewards


# ── PPO Training ──────────────────────────────────────────────────────────────

def train_ppo(total_timesteps=30000, seed=SEED, ent_coef=0.01,
              personality=None, enable_drift=False, label="ppo"):
    agent = FashionPPOAgent(
        seed=seed, ent_coef=ent_coef,
        personality=personality, verbose=0,
    )
    # Swap env if drift needed
    if enable_drift:
        from stable_baselines3.common.vec_env import DummyVecEnv
        from stable_baselines3.common.monitor import Monitor
        agent.model.set_env(DummyVecEnv([
            lambda: Monitor(FashionEnv(seed=seed, enable_drift=True, drift_every_n=50))
        ]))

    t0 = time.time()
    training_data = agent.train(total_timesteps=total_timesteps)
    elapsed = time.time() - t0
    print(f"[PPO/{label}] Training time: {elapsed:.1f}s")

    # Evaluate
    eval_env = FashionEnv(seed=seed+999, max_steps=8)
    eval_result = agent.evaluate(n_episodes=100, env=eval_env)
    eval_result["agent"] = label
    eval_result["training_time_s"] = elapsed
    eval_result["total_timesteps"] = total_timesteps

    # Save model
    model_path = os.path.join(MODELS_DIR, f"{label}.zip")
    agent.save(model_path)

    return agent, training_data, eval_result


# ── DQN Training ─────────────────────────────────────────────────────────────

def train_dqn(total_timesteps=30000, seed=SEED, label="dqn"):
    agent = FashionDQNAgent(seed=seed, verbose=0)
    t0 = time.time()
    training_data = agent.train(total_timesteps=total_timesteps)
    elapsed = time.time() - t0
    print(f"[DQN/{label}] Training time: {elapsed:.1f}s")

    eval_env = FashionEnv(seed=seed+999, max_steps=8)
    eval_result = agent.evaluate(n_episodes=100, env=eval_env)
    eval_result["agent"] = label
    eval_result["training_time_s"] = elapsed

    model_path = os.path.join(MODELS_DIR, f"{label}.zip")
    agent.save(model_path)
    return agent, training_data, eval_result


# ── Experiment Runner ─────────────────────────────────────────────────────────

def to_serializable(val):
    if isinstance(val, (np.floating, float)):
        return float(val)
    elif isinstance(val, (np.integer, int)):
        return int(val)
    elif isinstance(val, (np.ndarray, list, tuple)):
        return [to_serializable(x) for x in val]
    elif isinstance(val, dict):
        return {k: to_serializable(v) for k, v in val.items()}
    return val


def save_result(name, data):
    path = os.path.join(RESULTS_DIR, f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_serializable(data), f, indent=2)
    print(f"  -> Saved: {path}")


def run_all_experiments(timesteps=30000):
    print("\n" + "="*60)
    print("  FashionVerse RL Training -- All Experiments")
    print("="*60)

    all_results = {}

    # ── Exp 1: Baseline Comparison ────────────────────────────────────────
    print("\n[Exp 1] Random vs Rule-Based vs Popularity vs PPO vs DQN")

    rand_result, rand_rewards     = evaluate_random_baseline(n_episodes=200)
    rule_result, rule_rewards     = evaluate_rule_based_baseline(n_episodes=200)
    pop_result, pop_rewards       = evaluate_popularity_baseline(n_episodes=200)
    ppo_agent, ppo_train, ppo_res = train_ppo(total_timesteps=timesteps, label="ppo_exp1")
    dqn_agent, dqn_train, dqn_res = train_dqn(total_timesteps=timesteps, label="dqn_exp1")

    exp1 = {
        "random":     rand_result,
        "rule_based": rule_result,
        "popularity": pop_result,
        "ppo":        ppo_res,
        "dqn":        dqn_res,
        "ppo_training_curve": {
            "timesteps": ppo_train["timesteps_at_log"],
            "mean_rewards": ppo_train["mean_rewards_log"],
            "episode_rewards": ppo_train["episode_rewards"],
        },
        "dqn_training_curve": {
            "timesteps": dqn_train["timesteps_at_log"],
            "mean_rewards": dqn_train["mean_rewards_log"],
        },
        "random_episode_rewards":   rand_rewards,
        "rule_based_episode_rewards": rule_rewards,
    }
    save_result("exp1_baseline_comparison", exp1)
    all_results["exp1"] = exp1

    # ── Exp 2: Static vs Drifting Preferences ────────────────────────────
    print("\n[Exp 2] PPO: Static vs Drifting User Preferences")

    _, ppo_static_train, ppo_static_res = train_ppo(
        total_timesteps=timesteps, label="ppo_static", enable_drift=False)
    _, ppo_drift_train, ppo_drift_res = train_ppo(
        total_timesteps=timesteps, label="ppo_drift", enable_drift=True)

    exp2 = {
        "static_prefs": {
            "eval": ppo_static_res,
            "curve": {
                "timesteps": ppo_static_train["timesteps_at_log"],
                "mean_rewards": ppo_static_train["mean_rewards_log"],
            },
        },
        "drifting_prefs": {
            "eval": ppo_drift_res,
            "curve": {
                "timesteps": ppo_drift_train["timesteps_at_log"],
                "mean_rewards": ppo_drift_train["mean_rewards_log"],
            },
        },
    }
    save_result("exp2_preference_drift", exp2)
    all_results["exp2"] = exp2

    # ── Exp 3: Reward Ablation (with/without diversity) ───────────────────
    print("\n[Exp 3] Reward Ablation: with vs without Diversity component")

    cfg_with_div    = RewardConfig(use_diversity=True)
    cfg_without_div = RewardConfig(use_diversity=False)

    ppo_with    = FashionPPOAgent(reward_config=cfg_with_div,    seed=SEED, verbose=0)
    ppo_without = FashionPPOAgent(reward_config=cfg_without_div, seed=SEED, verbose=0)

    train_with    = ppo_with.train(total_timesteps=timesteps)
    train_without = ppo_without.train(total_timesteps=timesteps)

    eval_with    = ppo_with.evaluate(n_episodes=100)
    eval_without = ppo_without.evaluate(n_episodes=100)

    exp3 = {
        "with_diversity": {
            "eval": eval_with,
            "curve": {"timesteps": train_with["timesteps_at_log"],
                       "mean_rewards": train_with["mean_rewards_log"]},
        },
        "without_diversity": {
            "eval": eval_without,
            "curve": {"timesteps": train_without["timesteps_at_log"],
                       "mean_rewards": train_without["mean_rewards_log"]},
        },
    }
    save_result("exp3_reward_ablation", exp3)
    all_results["exp3"] = exp3

    # ── Exp 4: Exploration vs Exploitation ────────────────────────────────
    print("\n[Exp 4] Exploration vs Exploitation (entropy coefficient)")

    results_exp4 = {}
    for label, ent in [("high_exploration", 0.05),
                        ("balanced",         0.01),
                        ("exploitation",     0.001)]:
        agent = FashionPPOAgent(seed=SEED, ent_coef=ent, verbose=0)
        td = agent.train(total_timesteps=timesteps)
        ev = agent.evaluate(n_episodes=100)
        results_exp4[label] = {
            "ent_coef": ent,
            "eval": ev,
            "curve": {"timesteps": td["timesteps_at_log"],
                       "mean_rewards": td["mean_rewards_log"]},
        }
        print(f"  ent_coef={ent}: acceptance={ev['acceptance_rate']:.3f}")

    save_result("exp4_exploration", results_exp4)
    all_results["exp4"] = results_exp4

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  EXPERIMENT SUMMARY")
    print("="*60)
    print(f"  Random     : {rand_result['mean_reward']:.3f} reward | "
          f"{rand_result['acceptance_rate']:.3f} acceptance")
    print(f"  RuleBased  : {rule_result['mean_reward']:.3f} reward | "
          f"{rule_result['acceptance_rate']:.3f} acceptance")
    print(f"  Popularity : {pop_result['mean_reward']:.3f} reward | "
          f"{pop_result['acceptance_rate']:.3f} acceptance")
    print(f"  DQN        : {dqn_res['mean_reward']:.3f} reward | "
          f"{dqn_res['acceptance_rate']:.3f} acceptance")
    print(f"  PPO        : {ppo_res['mean_reward']:.3f} reward | "
          f"{ppo_res['acceptance_rate']:.3f} acceptance")

    save_result("all_results_summary", all_results)
    return all_results, ppo_agent


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=30000)
    parser.add_argument("--quick",     action="store_true",
                        help="Quick run with 5000 timesteps for testing")
    args = parser.parse_args()

    ts = 5000 if args.quick else args.timesteps
    run_all_experiments(timesteps=ts)
