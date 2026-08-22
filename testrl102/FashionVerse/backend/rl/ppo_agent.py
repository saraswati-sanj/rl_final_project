"""
FashionVerse — PPO Agent Wrapper

Why PPO over DQN for this problem?
====================================
1. Action space size: ACTION_DIM=60 discrete actions. DQN works, but PPO's
   stochastic policy naturally provides exploration without epsilon decay tuning.

2. Sequential outfit construction creates correlated episodes — PPO's clipped
   surrogate objective is more stable than DQN's replay buffer for this structure.

3. PPO's entropy bonus explicitly encourages exploration of novel outfit
   combinations, directly mapping to our diversity requirement.

4. PPO handles non-stationary reward landscapes (changing user preferences)
   better due to its on-policy nature — when preferences drift, the policy
   update uses fresh trajectories, not stale replays.

Implementation:
  - Uses Stable-Baselines3 PPO with a custom MultiInputPolicy or MlpPolicy
  - Exposes logging hooks for academic visualization
  - Supports action masking via ActionMasker wrapper
  - Saves/loads model weights for deployment
"""

import os
import numpy as np
from typing import Optional, Dict, List, Callable
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    BaseCallback, EvalCallback, CallbackList
)
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from backend.rl.fashion_env import FashionEnv
from backend.rl.reward import RewardConfig


# ── Logging Callback ──────────────────────────────────────────────────────────

class FashionRLCallback(BaseCallback):
    """
    Custom callback for logging FashionVerse-specific metrics.
    Captures: reward, acceptance rate, feedback distribution, episode length.
    Used to generate real training curves for the RL dashboard.
    """

    def __init__(self, log_freq: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq

        # Accumulated metrics
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.episode_feedbacks: List[str] = []
        self.acceptance_rates: List[float] = []
        self.timesteps_at_log: List[int] = []
        self.mean_rewards_log: List[float] = []

        self._ep_reward = 0.0
        self._ep_len = 0

    def _on_step(self) -> bool:
        self._ep_reward += self.locals.get("rewards", [0])[0]
        self._ep_len += 1

        # SB3 signals episode end via 'dones'
        dones = self.locals.get("dones", [False])
        infos = self.locals.get("infos", [{}])

        if dones[0]:
            self.episode_rewards.append(self._ep_reward)
            self.episode_lengths.append(self._ep_len)
            self._ep_reward = 0.0
            self._ep_len = 0

            feedback = infos[0].get("feedback", "neutral")
            self.episode_feedbacks.append(feedback)

        if self.n_calls % self.log_freq == 0 and self.episode_rewards:
            window = min(50, len(self.episode_rewards))
            mean_r = np.mean(self.episode_rewards[-window:])
            self.timesteps_at_log.append(self.num_timesteps)
            self.mean_rewards_log.append(float(mean_r))

            if self.verbose > 0:
                print(f"  [PPO] Timestep {self.num_timesteps:>7} | "
                      f"Mean reward (last {window}): {mean_r:.3f} | "
                      f"Episodes: {len(self.episode_rewards)}")

        return True

    def get_training_data(self) -> Dict:
        """Returns all logged data for plotting and dashboard."""
        feedbacks = self.episode_feedbacks
        n = len(feedbacks)
        pos = sum(1 for f in feedbacks if f in ("love", "like", "save", "purchase"))
        return {
            "episode_rewards":   self.episode_rewards,
            "episode_lengths":   self.episode_lengths,
            "episode_feedbacks": self.episode_feedbacks,
            "timesteps_at_log":  self.timesteps_at_log,
            "mean_rewards_log":  self.mean_rewards_log,
            "total_episodes":    n,
            "acceptance_rate":   pos / n if n > 0 else 0.0,
            "feedback_distribution": {
                f: feedbacks.count(f) for f in
                ["love", "like", "neutral", "skip", "dislike", "save", "purchase"]
            },
        }


# ── PPO Agent ────────────────────────────────────────────────────────────────

class FashionPPOAgent:
    """
    Wraps Stable-Baselines3 PPO for FashionVerse.

    Hyperparameters:
      learning_rate  : Step size for Adam optimizer (default 3e-4)
      n_steps        : Steps per rollout before update (default 512)
      batch_size     : Mini-batch size for PPO epochs (default 64)
      n_epochs       : PPO epochs per update (default 10)
      gamma          : Discount factor for future rewards (default 0.99)
      gae_lambda     : GAE-lambda for advantage estimation (default 0.95)
      clip_range     : PPO clipping parameter ε (default 0.2)
      ent_coef       : Entropy bonus weight — controls exploration (default 0.01)

    The entropy coefficient (ent_coef) directly maps to our
    exploration requirement: higher = more diverse recommendations.
    """

    def __init__(
        self,
        reward_config: Optional[RewardConfig] = None,
        learning_rate: float = 3e-4,
        n_steps: int = 512,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,   # entropy bonus for exploration
        seed: int = 42,
        device: str = "auto",
        verbose: int = 0,
        personality: Optional[str] = None,
        noise_level: float = 0.15,
    ):
        self.reward_config = reward_config or RewardConfig()
        self.seed = seed
        self.device = device
        self.verbose = verbose

        # Create the training environment
        self._env_fn = lambda: Monitor(FashionEnv(
            reward_config=reward_config,
            seed=seed,
            personality=personality,
            noise_level=noise_level,
        ))

        env = DummyVecEnv([self._env_fn])

        # Network architecture: two hidden layers of 256 neurons
        policy_kwargs = dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
            activation_fn=torch.nn.ReLU,
        )

        self.model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            policy_kwargs=policy_kwargs,
            seed=seed,
            device=device,
            verbose=verbose,
        )

        self.callback = FashionRLCallback(log_freq=100, verbose=verbose)
        self._is_trained = False

    def train(self, total_timesteps: int = 20000) -> Dict:
        """
        Train the PPO agent and return training metrics.
        """
        print(f"\n[PPO] Starting training: {total_timesteps} timesteps")
        print(f"      Seed={self.seed} | Gamma={self.model.gamma} | "
              f"EntCoef={self.model.ent_coef}")

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=self.callback,
            reset_num_timesteps=True,
            progress_bar=False,
        )
        self._is_trained = True
        data = self.callback.get_training_data()
        print(f"[PPO] Training complete. Episodes: {data['total_episodes']} | "
              f"Acceptance rate: {data['acceptance_rate']:.3f}")
        return data

    def predict(self, obs: np.ndarray, deterministic: bool = False) -> int:
        """Predict action for a given observation."""
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return int(action)

    def evaluate(self, n_episodes: int = 100, env: Optional[FashionEnv] = None) -> Dict:
        """
        Evaluate the trained policy on n_episodes without training.
        Returns metrics: mean_reward, acceptance_rate, feedback_distribution.
        """
        if env is None:
            env = FashionEnv(reward_config=self.reward_config, seed=self.seed + 1000)

        rewards = []
        feedbacks = []
        lengths = []

        for ep in range(n_episodes):
            obs, _ = env.reset()
            done = False
            ep_reward = 0
            ep_len = 0
            ep_feedback = "neutral"

            while not done:
                action = self.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward
                ep_len += 1
                if "feedback" in info:
                    ep_feedback = info["feedback"]
                done = terminated or truncated

            rewards.append(ep_reward)
            feedbacks.append(ep_feedback)
            lengths.append(ep_len)

        n = len(feedbacks)
        pos = sum(1 for f in feedbacks if f in ("love", "like", "save", "purchase"))

        return {
            "n_episodes": n,
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "acceptance_rate": pos / n if n > 0 else 0.0,
            "mean_length": float(np.mean(lengths)),
            "feedback_distribution": {
                f: feedbacks.count(f) for f in
                ["love", "like", "neutral", "skip", "dislike", "save", "purchase"]
            },
        }

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        self.model.save(path)
        print(f"[PPO] Model saved to {path}")

    def load(self, path: str):
        self.model = PPO.load(path, env=DummyVecEnv([self._env_fn]))
        self._is_trained = True
        print(f"[PPO] Model loaded from {path}")

    def is_trained(self) -> bool:
        return self._is_trained
