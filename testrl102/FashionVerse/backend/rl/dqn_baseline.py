"""
FashionVerse — DQN Baseline
Used as a secondary RL baseline for comparison with PPO.

Why DQN is used as a BASELINE (not the primary agent):
=======================================================
1. DQN uses experience replay with fixed-size buffer — stale transitions
   cause slower adaptation when user preferences change (Experiment 2).

2. DQN's ε-greedy exploration is less principled than PPO's entropy bonus
   for structured item spaces (fashion compatibility matters, pure random
   exploration wastes episodes on incompatible outfits).

3. DQN's target network update introduces lag in reward estimation that
   accumulates in sequential, correlated outfit-building episodes.

That said, DQN is a legitimate RL algorithm and serves as a useful
comparison point to demonstrate PPO's relative advantage in this domain.
"""

import os
import numpy as np
from typing import Optional, Dict, List

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import torch

from backend.rl.fashion_env import FashionEnv
from backend.rl.reward import RewardConfig


class DQNCallback(BaseCallback):
    def __init__(self, log_freq=100, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.episode_rewards: List[float] = []
        self.episode_feedbacks: List[str] = []
        self.timesteps_at_log: List[int] = []
        self.mean_rewards_log: List[float] = []
        self._ep_reward = 0.0

    def _on_step(self) -> bool:
        self._ep_reward += self.locals.get("rewards", [0])[0]
        dones = self.locals.get("dones", [False])
        infos = self.locals.get("infos", [{}])
        if dones[0]:
            self.episode_rewards.append(self._ep_reward)
            self._ep_reward = 0.0
            self.episode_feedbacks.append(infos[0].get("feedback", "neutral"))

        if self.n_calls % self.log_freq == 0 and self.episode_rewards:
            window = min(50, len(self.episode_rewards))
            mean_r = np.mean(self.episode_rewards[-window:])
            self.timesteps_at_log.append(self.num_timesteps)
            self.mean_rewards_log.append(float(mean_r))
        return True

    def get_training_data(self) -> Dict:
        feedbacks = self.episode_feedbacks
        n = len(feedbacks)
        pos = sum(1 for f in feedbacks if f in ("love", "like", "save", "purchase"))
        return {
            "episode_rewards":   self.episode_rewards,
            "timesteps_at_log":  self.timesteps_at_log,
            "mean_rewards_log":  self.mean_rewards_log,
            "total_episodes":    n,
            "acceptance_rate":   pos / n if n > 0 else 0.0,
        }


class FashionDQNAgent:
    """DQN-based fashion RL agent — secondary baseline."""

    def __init__(
        self,
        reward_config: Optional[RewardConfig] = None,
        learning_rate: float = 1e-4,
        buffer_size: int = 10000,
        learning_starts: int = 500,
        batch_size: int = 64,
        gamma: float = 0.99,
        target_update_interval: int = 500,
        exploration_fraction: float = 0.3,
        exploration_final_eps: float = 0.05,
        seed: int = 42,
        verbose: int = 0,
        personality: Optional[str] = None,
        noise_level: float = 0.15,
    ):
        self.reward_config = reward_config or RewardConfig()
        self.seed = seed

        self._env_fn = lambda: Monitor(FashionEnv(
            reward_config=reward_config,
            seed=seed,
            personality=personality,
            noise_level=noise_level,
        ))
        env = DummyVecEnv([self._env_fn])

        policy_kwargs = dict(
            net_arch=[256, 256],
            activation_fn=torch.nn.ReLU,
        )

        self.model = DQN(
            policy="MlpPolicy",
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            gamma=gamma,
            target_update_interval=target_update_interval,
            exploration_fraction=exploration_fraction,
            exploration_final_eps=exploration_final_eps,
            policy_kwargs=policy_kwargs,
            seed=seed,
            verbose=verbose,
        )
        self.callback = DQNCallback(log_freq=100, verbose=verbose)
        self._is_trained = False

    def train(self, total_timesteps: int = 20000) -> Dict:
        print(f"\n[DQN] Starting training: {total_timesteps} timesteps | Seed={self.seed}")
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=self.callback,
            reset_num_timesteps=True,
            progress_bar=False,
        )
        self._is_trained = True
        data = self.callback.get_training_data()
        print(f"[DQN] Done. Episodes: {data['total_episodes']} | "
              f"Acceptance rate: {data['acceptance_rate']:.3f}")
        return data

    def predict(self, obs: np.ndarray, deterministic: bool = False) -> int:
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return int(action)

    def evaluate(self, n_episodes: int = 100, env=None) -> Dict:
        if env is None:
            env = FashionEnv(reward_config=self.reward_config, seed=self.seed + 1000)
        rewards, feedbacks, lengths = [], [], []
        for _ in range(n_episodes):
            obs, _ = env.reset()
            done, ep_r, ep_f = False, 0, "neutral"
            while not done:
                a = self.predict(obs, deterministic=True)
                obs, r, terminated, truncated, info = env.step(a)
                ep_r += r
                ep_f = info.get("feedback", ep_f)
                done = terminated or truncated
            rewards.append(ep_r)
            feedbacks.append(ep_f)
        n = len(feedbacks)
        pos = sum(1 for f in feedbacks if f in ("love", "like", "save", "purchase"))
        return {
            "n_episodes": n,
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "acceptance_rate": pos / n if n > 0 else 0.0,
            "feedback_distribution": {
                f: feedbacks.count(f) for f in
                ["love", "like", "neutral", "skip", "dislike", "save", "purchase"]
            },
        }

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        self.model.save(path)

    def is_trained(self) -> bool:
        return self._is_trained
