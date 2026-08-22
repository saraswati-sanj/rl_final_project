"""
FashionVerse — Fashion RL Environment (Gymnasium)

MDP Formulation
===============
State  S_t : FashionState encoded as float32 numpy array (STATE_DIM=70)
Action A_t : Integer in [0, ACTION_DIM) = 60 (6 types × 10 candidates)
Reward R_t : Step shaping + terminal user feedback
Next state S_{t+1}: Updated after item added or outfit finished

Episode
-------
An episode represents ONE outfit recommendation session:
  1. Reset: pick a simulated user + fashion request
  2. Loop:
     a. Agent selects action (item to add or finish)
     b. Environment validates the action
     c. If valid item: add to outfit, compute step reward
     d. If invalid: small penalty, no state change
     e. If finish (or max steps): present outfit to user simulator,
        collect terminal reward, update user profile, done=True
  3. The agent's policy learns to maximize cumulative reward across episodes.

Episode termination conditions:
  - Agent selects "finish_outfit"
  - Max steps reached (configurable, default=8)
  - Outfit is already complete (has coverage + shoes)

The key RL loop:
  State → Agent → Action → Env → Reward → Next State → Policy Update (PPO)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
from typing import Optional, Dict, Tuple, List

from backend.rl.state import FashionState, STATE_DIM, REWARD_HISTORY_LEN
from backend.rl.actions import ActionSpace, ACTION_DIM, FINISH_ACTION_ID
from backend.rl.reward import RewardCalculator, RewardConfig
from backend.fashion.catalog import get_catalog
from backend.fashion.constraints import FashionConstraints
from backend.fashion.compatibility import is_complete, score_outfit
from backend.user.user_simulator import UserSimulator, SimulatedUser
from backend.user.user_profile import UserProfile
from backend.user.preference_update import update_preference


# ── Episode config presets (used to generate diverse training scenarios) ───────
EPISODE_PRESETS = [
    {"occasion": "casual",      "season": "summer",  "budget": 2000, "formality": (1, 3)},
    {"occasion": "college",     "season": "summer",  "budget": 2500, "formality": (1, 3)},
    {"occasion": "office",      "season": "all",     "budget": 4000, "formality": (3, 5)},
    {"occasion": "semi_formal", "season": "all",     "budget": 3500, "formality": (3, 5)},
    {"occasion": "formal",      "season": "winter",  "budget": 5000, "formality": (4, 5)},
    {"occasion": "party",       "season": "all",     "budget": 3000, "formality": (2, 4)},
    {"occasion": "festive",     "season": "all",     "budget": 3500, "formality": (2, 5)},
    {"occasion": "date",        "season": "summer",  "budget": 3000, "formality": (2, 4)},
    {"occasion": "travel",      "season": "all",     "budget": 2000, "formality": (1, 3)},
]


class FashionEnv(gym.Env):
    """
    FashionVerse Gymnasium Environment.

    Implements a genuine sequential decision-making MDP for outfit construction.
    Compatible with Stable-Baselines3 PPO.
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        reward_config: Optional[RewardConfig] = None,
        max_steps: int = 8,
        n_candidates: int = 10,
        seed: int = 42,
        render_mode: Optional[str] = None,
        personality: Optional[str] = None,   # None = random each episode
        noise_level: float = 0.15,
        enable_drift: bool = False,           # preference drift experiment
        drift_every_n: int = 100,             # drift every N episodes
    ):
        super().__init__()

        self.render_mode = render_mode
        self.max_steps   = max_steps
        self.seed_val    = seed
        self.personality = personality
        self.noise_level = noise_level
        self.enable_drift = enable_drift
        self.drift_every_n = drift_every_n

        # ── Spaces ────────────────────────────────────────────────────────
        self.observation_space = spaces.Box(
            low=-1.0, high=2.0, shape=(STATE_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(ACTION_DIM)

        # ── Sub-components ────────────────────────────────────────────────
        self.catalog       = get_catalog()
        self.action_space_ = ActionSpace(self.catalog, n_candidates)
        self.reward_calc   = RewardCalculator(reward_config)
        self.simulator     = UserSimulator(seed=seed)

        # ── Episode state ─────────────────────────────────────────────────
        self._state: Optional[FashionState] = None
        self._user: Optional[SimulatedUser] = None
        self._profile: Optional[UserProfile] = None
        self._constraints: Optional[FashionConstraints] = None
        self._episode_count: int = 0
        self._step_infos: List[dict] = []
        self._rng = random.Random(seed)

        # ── Metrics tracking ──────────────────────────────────────────────
        self.episode_rewards: List[float] = []
        self.episode_feedbacks: List[str] = []
        self.episode_lengths: List[int] = []

    # ── Gymnasium interface ───────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = random.Random(seed)

        self._episode_count += 1
        self._step_infos = []

        # ── Create a new simulated user (or reuse if single-user mode) ────
        personality = self.personality
        if personality is None:
            personality = self._rng.choice(self.simulator.list_personalities())

        self._user = self.simulator.create_user(
            user_id=f"ep_{self._episode_count}",
            personality=personality,
            noise_level=self.noise_level,
        )
        self._profile = UserProfile(
            user_id=f"ep_{self._episode_count}",
            budget=self._user.budget,
        )

        # ── Apply preference drift if enabled ─────────────────────────────
        if self.enable_drift and self._episode_count % self.drift_every_n == 0:
            self._user.apply_preference_drift(drift_rate=0.2)

        # ── Sample a random episode preset (scenario) ─────────────────────
        preset = self._rng.choice(EPISODE_PRESETS)
        # Override budget with user's budget
        budget = min(self._user.budget, preset["budget"])

        self._constraints = FashionConstraints(
            budget=budget,
            occasion=preset["occasion"],
            season=preset["season"],
            gender="unisex",
            formality_min=preset["formality"][0],
            formality_max=preset["formality"][1],
        )

        # ── Initialize state ──────────────────────────────────────────────
        self._state = FashionState(
            user_profile_vec=self._profile.to_vector(),
            occasion=preset["occasion"],
            season=preset["season"],
            budget_total=budget,
            budget_remaining=budget,
            formality_target=float(sum(preset["formality"])) / 2,
            selected_items=[],
            step_in_episode=0,
            reward_history=[0.0] * REWARD_HISTORY_LEN,
        )

        # ── Build initial action candidates ───────────────────────────────
        self.action_space_.build_candidates(self._constraints, [])

        obs = self._state.encode()
        info = {
            "episode": self._episode_count,
            "personality": personality,
            "occasion": preset["occasion"],
            "budget": budget,
        }
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one step of the MDP.
        Returns: (observation, reward, terminated, truncated, info)
        """
        assert self._state is not None, "Call reset() first"

        state = self._state
        decoded = self.action_space_.decode(action)
        step_reward = 0.0
        step_info: dict = {"action_type": decoded.action_type, "valid": decoded.is_valid}
        terminated = False
        truncated  = False

        # ── Invalid action ────────────────────────────────────────────────
        if not decoded.is_valid:
            step_reward = self.reward_calc.invalid_action_reward()
            step_info["reason"] = decoded.invalid_reason

        # ── Finish outfit ─────────────────────────────────────────────────
        elif decoded.is_finish or state.step_in_episode >= self.max_steps - 1:
            step_reward, term_info = self._handle_finish()
            step_info.update(term_info)
            terminated = True

        # ── Add item ──────────────────────────────────────────────────────
        else:
            item = decoded.item
            validation = self._constraints.validate_item(item, state.selected_items)

            if not validation["valid"]:
                step_reward = self.reward_calc.invalid_action_reward()
                step_info["reason"] = validation["reason"]
            else:
                # Valid item addition
                state.selected_items.append(item)
                state.budget_remaining -= item.price

                # Step reward (shaping)
                step_reward, sr_info = self.reward_calc.step_reward(
                    new_item=item,
                    current_items=state.selected_items,
                    constraints=self._constraints,
                    recent_item_ids=self._profile.recently_recommended,
                    occasion=state.occasion,
                )
                step_info.update(sr_info)

                # Auto-finish if outfit is complete and budget nearly exhausted
                if (is_complete(state.selected_items) and
                        state.budget_remaining < 200):
                    fin_r, term_info = self._handle_finish()
                    step_reward += fin_r
                    step_info.update(term_info)
                    terminated = True

        # ── Update state ──────────────────────────────────────────────────
        state.step_in_episode += 1
        state.reward_history = state.reward_history[1:] + [step_reward]
        state.user_profile_vec = self._profile.to_vector()

        # Truncate if max steps exceeded
        if state.step_in_episode >= self.max_steps and not terminated:
            fin_r, term_info = self._handle_finish()
            step_reward += fin_r
            step_info.update(term_info)
            truncated = True

        # Rebuild candidates for next step
        if not terminated and not truncated:
            self.action_space_.build_candidates(
                self._constraints, state.selected_items
            )

        self._step_infos.append(step_info)

        obs = state.encode()
        step_info["step"] = state.step_in_episode
        step_info["budget_remaining"] = state.budget_remaining

        return obs, step_reward, terminated, truncated, step_info

    def _handle_finish(self) -> Tuple[float, dict]:
        """
        Present the current outfit to the user simulator.
        Compute terminal reward. Update user profile.
        """
        items = self._state.selected_items
        occasion = self._state.occasion
        outfit_complete = is_complete(items)

        # Get user feedback
        if items:
            feedback, satisfaction = self._user.evaluate_outfit(items, occasion)
        else:
            feedback, satisfaction = "skip", 0.0

        # Terminal reward
        term_reward, term_info = self.reward_calc.terminal_reward(
            feedback=feedback,
            items=items,
            constraints=self._constraints,
            recent_item_ids=self._profile.recently_recommended,
            occasion=occasion,
            is_complete=outfit_complete,
        )

        # Update observable user profile
        if items:
            update_preference(self._profile, items, feedback)

        # Track episode-level metrics
        self.episode_rewards.append(term_reward)
        self.episode_feedbacks.append(feedback)
        self.episode_lengths.append(self._state.step_in_episode)

        term_info.update({
            "feedback": feedback,
            "satisfaction": satisfaction,
            "outfit_complete": outfit_complete,
            "n_items": len(items),
            "total_price": sum(i.price for i in items),
            "terminal_reward": term_reward,
        })
        return term_reward, term_info

    def render(self):
        if self.render_mode == "human" or self.render_mode == "ansi":
            state = self._state
            if state is None:
                print("Environment not initialized. Call reset().")
                return
            print(f"\n=== Episode {self._episode_count} | Step {state.step_in_episode} ===")
            print(f"  Occasion : {state.occasion}")
            print(f"  Budget   : {state.budget_remaining}/{state.budget_total}")
            print(f"  Items    : {len(state.selected_items)}")
            for item in state.selected_items:
                print(f"    [{item.category:10s}] {item.name[:35]} ₹{item.price}")
            print(f"  Complete : {is_complete(state.selected_items)}")

    def get_episode_stats(self) -> dict:
        """Returns aggregate stats over all episodes so far."""
        if not self.episode_rewards:
            return {}
        n = len(self.episode_rewards)
        feedback_counts = {}
        for f in self.episode_feedbacks:
            feedback_counts[f] = feedback_counts.get(f, 0) + 1

        pos_feedbacks = sum(feedback_counts.get(f, 0)
                            for f in ("love", "like", "save", "purchase"))
        return {
            "n_episodes": n,
            "mean_reward": sum(self.episode_rewards) / n,
            "acceptance_rate": pos_feedbacks / n,
            "feedback_distribution": feedback_counts,
            "mean_episode_length": sum(self.episode_lengths) / n,
        }

    def close(self):
        pass
