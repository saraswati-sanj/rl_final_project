"""
FashionVerse — Evaluation & Plot Generation
Loads experiment results and generates all 10 required plots.
Run: python training/evaluate.py
"""

import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "experiments", "results")
PLOTS_DIR   = os.path.join(os.path.dirname(__file__), "..", "experiments", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Style
sns.set_theme(style="darkgrid", palette="muted")
COLORS = {"ppo": "#6C63FF", "dqn": "#FF6584", "rule_based": "#43B89C",
          "random": "#A0A0A0", "popularity": "#F5A623",
          "with_diversity": "#6C63FF", "without_diversity": "#FF6584",
          "high_exploration": "#FF6584", "balanced": "#6C63FF",
          "exploitation": "#43B89C"}


def load(name):
    path = os.path.join(RESULTS_DIR, f"{name}.json")
    if not os.path.exists(path):
        print(f"  [WARN] Missing: {path}")
        return None
    with open(path) as f:
        return json.load(f)


def moving_avg(data, window=10):
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window)/window, mode="valid").tolist()


def save_plot(fig, name):
    path = os.path.join(PLOTS_DIR, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Plot] Saved: {path}")
    return path


# ── Plot 1 & 2: Reward vs Episode + Moving Average ────────────────────────────
def plot_reward_curves(exp1):
    if not exp1:
        return
    ppo_curve = exp1.get("ppo_training_curve", {})
    rewards   = ppo_curve.get("episode_rewards", [])
    if not rewards:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("PPO Training Reward", fontsize=14, fontweight="bold")

    # Raw reward per episode
    axes[0].plot(rewards, alpha=0.4, color=COLORS["ppo"], linewidth=0.8)
    axes[0].set_title("Reward per Episode")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Total Episode Reward")

    # Moving average
    ma = moving_avg(rewards, window=20)
    x  = list(range(len(ma)))
    axes[1].plot(x, ma, color=COLORS["ppo"], linewidth=2)
    axes[1].fill_between(x, [v - 1 for v in ma], [v + 1 for v in ma],
                          alpha=0.2, color=COLORS["ppo"])
    axes[1].set_title("Moving Average Reward (window=20)")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Avg Reward")

    plt.tight_layout()
    save_plot(fig, "01_02_reward_curves")


# ── Plot 3: PPO vs Baselines ──────────────────────────────────────────────────
def plot_baseline_comparison(exp1):
    if not exp1:
        return
    agents  = ["random", "rule_based", "popularity", "dqn", "ppo"]
    labels  = ["Random", "Rule-Based", "Popularity", "DQN", "PPO"]
    rewards = [exp1.get(a, {}).get("mean_reward", 0) for a in agents]
    accepts = [exp1.get(a, {}).get("acceptance_rate", 0) for a in agents]
    colors  = [COLORS.get(a, "#999") for a in agents]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Algorithm Comparison", fontsize=14, fontweight="bold")

    bars1 = axes[0].bar(labels, rewards, color=colors, edgecolor="white", linewidth=1.2)
    axes[0].set_title("Mean Episode Reward")
    axes[0].set_ylabel("Reward")
    for bar, val in zip(bars1, rewards):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                     f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    bars2 = axes[1].bar(labels, [a*100 for a in accepts], color=colors,
                         edgecolor="white", linewidth=1.2)
    axes[1].set_title("Acceptance Rate (%)")
    axes[1].set_ylabel("Acceptance Rate (%)")
    axes[1].set_ylim(0, 100)
    for bar, val in zip(bars2, accepts):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f"{val*100:.1f}%", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    save_plot(fig, "03_baseline_comparison")


# ── Plot 4: Acceptance Rate ───────────────────────────────────────────────────
def plot_acceptance_rate(exp1):
    if not exp1:
        return
    agents = ["random", "rule_based", "popularity", "dqn", "ppo"]
    labels = ["Random", "Rule-Based", "Popularity", "DQN", "PPO (Ours)"]
    rates  = [exp1.get(a, {}).get("acceptance_rate", 0)*100 for a in agents]
    colors = [COLORS.get(a, "#999") for a in agents]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_title("Acceptance Rate by Algorithm", fontsize=13, fontweight="bold")
    bars = ax.barh(labels, rates, color=colors, edgecolor="white", height=0.6)
    ax.set_xlabel("Acceptance Rate (%)")
    ax.set_xlim(0, 105)
    for bar, val in zip(bars, rates):
        ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}%", va="center", fontsize=10)
    plt.tight_layout()
    save_plot(fig, "04_acceptance_rate")


# ── Plot 5: User Satisfaction (Feedback Distribution) ─────────────────────────
def plot_user_satisfaction(exp1):
    if not exp1:
        return
    ppo_dist = exp1.get("ppo", {}).get("feedback_distribution", {})
    rnd_dist = exp1.get("random", {}).get("feedback_distribution", {})
    if not ppo_dist:
        return

    labels = ["love", "like", "neutral", "skip", "dislike", "save", "purchase"]
    disp   = ["Love", "Like", "Neutral", "Skip", "Dislike", "Save", "Purchase"]
    ppo_v  = [ppo_dist.get(l, 0) for l in labels]
    rnd_v  = [rnd_dist.get(l, 0) for l in labels]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_title("Feedback Distribution: PPO vs Random", fontsize=13, fontweight="bold")
    w = 0.35
    ax.bar(x - w/2, ppo_v, w, label="PPO", color=COLORS["ppo"], edgecolor="white")
    ax.bar(x + w/2, rnd_v, w, label="Random", color=COLORS["random"], edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(disp)
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()
    save_plot(fig, "05_user_satisfaction")


# ── Plot 6: Exploration vs Exploitation ───────────────────────────────────────
def plot_exploration(exp4):
    if not exp4:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Exploration vs Exploitation (Entropy Coefficient)", fontsize=13, fontweight="bold")

    labels = {"high_exploration": "High Exploration\n(ent=0.05)",
              "balanced":         "Balanced\n(ent=0.01)",
              "exploitation":     "Exploitation\n(ent=0.001)"}

    for key, label in labels.items():
        d = exp4.get(key, {})
        curve = d.get("curve", {})
        ts = curve.get("timesteps", [])
        mr = curve.get("mean_rewards", [])
        if ts and mr:
            axes[0].plot(ts, mr, label=label, color=COLORS.get(key, "#999"), linewidth=2)

    axes[0].set_title("Training Curve by Exploration Level")
    axes[0].set_xlabel("Timesteps")
    axes[0].set_ylabel("Mean Reward")
    axes[0].legend(fontsize=8)

    acc = [exp4.get(k, {}).get("eval", {}).get("acceptance_rate", 0)*100
           for k in labels]
    bars = axes[1].bar(list(labels.values()), acc,
                        color=[COLORS.get(k, "#999") for k in labels],
                        edgecolor="white")
    axes[1].set_title("Acceptance Rate by Exploration Level")
    axes[1].set_ylabel("Acceptance Rate (%)")
    axes[1].set_ylim(0, 100)
    for bar, val in zip(bars, acc):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f"{val:.1f}%", ha="center", fontsize=9)
    plt.tight_layout()
    save_plot(fig, "06_exploration_exploitation")


# ── Plot 7: Recommendation Diversity ─────────────────────────────────────────
def plot_diversity(exp3):
    if not exp3:
        return
    wd = exp3.get("with_diversity", {})
    nd = exp3.get("without_diversity", {})

    labels  = ["With Diversity\nReward", "Without Diversity\nReward"]
    rewards = [wd.get("eval", {}).get("mean_reward", 0),
               nd.get("eval", {}).get("mean_reward", 0)]
    accepts = [wd.get("eval", {}).get("acceptance_rate", 0)*100,
               nd.get("eval", {}).get("acceptance_rate", 0)*100]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("Diversity Component Ablation", fontsize=13, fontweight="bold")

    c = [COLORS["with_diversity"], COLORS["without_diversity"]]
    axes[0].bar(labels, rewards, color=c, edgecolor="white")
    axes[0].set_title("Mean Reward")
    axes[0].set_ylabel("Reward")
    for i, (bar_x, val) in enumerate(zip(axes[0].patches, rewards)):
        axes[0].text(i, val + 0.05, f"{val:.2f}", ha="center", fontsize=10)

    axes[1].bar(labels, accepts, color=c, edgecolor="white")
    axes[1].set_title("Acceptance Rate (%)")
    axes[1].set_ylabel("%")
    axes[1].set_ylim(0, 100)
    plt.tight_layout()
    save_plot(fig, "07_diversity_ablation")


# ── Plot 8: Adaptation After Preference Change ────────────────────────────────
def plot_adaptation(exp2):
    if not exp2:
        return
    static  = exp2.get("static_prefs",   {}).get("curve", {})
    drifted = exp2.get("drifting_prefs", {}).get("curve", {})

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title("Adaptation: Static vs Drifting Preferences", fontsize=13, fontweight="bold")

    if static.get("timesteps") and static.get("mean_rewards"):
        ax.plot(static["timesteps"], static["mean_rewards"],
                label="Static Preferences", color=COLORS["ppo"], linewidth=2)
    if drifted.get("timesteps") and drifted.get("mean_rewards"):
        ax.plot(drifted["timesteps"], drifted["mean_rewards"],
                label="Drifting Preferences", color=COLORS["dqn"],
                linewidth=2, linestyle="--")

    ax.set_xlabel("Training Timesteps")
    ax.set_ylabel("Mean Reward (window=50)")
    ax.legend()
    ax.axvline(x=15000, color="grey", linestyle=":", alpha=0.7, label="Drift events")
    plt.tight_layout()
    save_plot(fig, "08_preference_adaptation")


# ── Plot 9: Budget Violations ────────────────────────────────────────────────
def plot_budget_violations(exp1):
    if not exp1:
        return
    # Estimate budget violation rate from feedback distribution
    # (skip/dislike at high rates often correlate with budget issues)
    agents = ["random", "rule_based", "popularity", "dqn", "ppo"]
    labels = ["Random", "Rule-Based", "Popularity", "DQN", "PPO"]
    # Use dislike+skip as proxy for poor-fit recommendations
    neg_rates = []
    for a in agents:
        dist = exp1.get(a, {}).get("feedback_distribution", {})
        n_total = sum(dist.values()) if dist else 1
        neg = dist.get("dislike", 0) + dist.get("skip", 0)
        neg_rates.append((neg / n_total) * 100 if n_total > 0 else 0)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_title("Negative Response Rate (Dislike+Skip) by Algorithm",
                  fontsize=12, fontweight="bold")
    colors = [COLORS.get(a, "#999") for a in agents]
    bars = ax.bar(labels, neg_rates, color=colors, edgecolor="white")
    ax.set_ylabel("Negative Response Rate (%)")
    ax.set_ylim(0, 100)
    for bar, val in zip(bars, neg_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", fontsize=9)
    plt.tight_layout()
    save_plot(fig, "09_budget_violations")


# ── Plot 10: Reward Component Ablation ───────────────────────────────────────
def plot_ablation(exp3, exp4):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Reward Component Ablation Study", fontsize=13, fontweight="bold")

    # Panel 1: Diversity ablation
    if exp3:
        configs = ["With Diversity", "Without Diversity"]
        rewards_ab = [
            exp3.get("with_diversity",    {}).get("eval", {}).get("mean_reward", 0),
            exp3.get("without_diversity", {}).get("eval", {}).get("mean_reward", 0),
        ]
        colors_ab = [COLORS["with_diversity"], COLORS["without_diversity"]]
        axes[0].bar(configs, rewards_ab, color=colors_ab, edgecolor="white")
        axes[0].set_title("Effect of Diversity Component")
        axes[0].set_ylabel("Mean Reward")
        for i, val in enumerate(rewards_ab):
            axes[0].text(i, val + 0.05, f"{val:.2f}", ha="center", fontsize=10)

    # Panel 2: Exploration ablation
    if exp4:
        configs = ["High Exp.", "Balanced", "Exploit."]
        keys    = ["high_exploration", "balanced", "exploitation"]
        rewards_ex = [exp4.get(k, {}).get("eval", {}).get("mean_reward", 0) for k in keys]
        colors_ex  = [COLORS.get(k, "#999") for k in keys]
        axes[1].bar(configs, rewards_ex, color=colors_ex, edgecolor="white")
        axes[1].set_title("Effect of Entropy Coefficient")
        axes[1].set_ylabel("Mean Reward")
        for i, val in enumerate(rewards_ex):
            axes[1].text(i, val + 0.05, f"{val:.2f}", ha="center", fontsize=10)

    plt.tight_layout()
    save_plot(fig, "10_reward_ablation")


# ── Main ──────────────────────────────────────────────────────────────────────
def generate_all_plots():
    print("\n[Evaluate] Loading experiment results and generating plots...")
    exp1 = load("exp1_baseline_comparison")
    exp2 = load("exp2_preference_drift")
    exp3 = load("exp3_reward_ablation")
    exp4 = load("exp4_exploration")

    plot_reward_curves(exp1)
    plot_baseline_comparison(exp1)
    plot_acceptance_rate(exp1)
    plot_user_satisfaction(exp1)
    plot_exploration(exp4)
    plot_diversity(exp3)
    plot_adaptation(exp2)
    plot_budget_violations(exp1)
    plot_ablation(exp3, exp4)

    print(f"\n[Evaluate] All plots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    generate_all_plots()
