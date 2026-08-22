"""
FashionVerse — Reward Function

Reward Design Rationale
========================
The reward must represent LONG-TERM user satisfaction, not just immediate clicks.
We use a multi-component weighted reward:

    R = w_u * U          (user satisfaction signal)
      + w_c * C          (outfit compatibility shaping)
      + w_o * O          (occasion match shaping)
      + w_b * B          (budget compliance bonus)
      + w_d * D          (diversity bonus — exploration incentive)
      - w_p * P          (penalties)

Shaping components (C, O, B, D) are applied at each step to guide
the agent toward good outfits without waiting until episode end.

The user satisfaction component (U) is only non-zero after the outfit
is presented to the user (episode end), mapping feedback to a scalar.

All weights are configurable via RewardConfig to support ablation studies.

Episode vs Step rewards:
  - Step reward: shaping only (C, O, B signals)
  - Terminal reward: U + full outfit evaluation
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
import math


# ── Reward Config ─────────────────────────────────────────────────────────────

@dataclass
class RewardConfig:
    """
    Configurable weights for the reward function.
    Adjust for ablation experiments (see training/evaluate.py).
    """
    # User satisfaction weights by feedback type
    feedback_love:     float = 10.0
    feedback_like:     float =  5.0
    feedback_neutral:  float =  0.0
    feedback_skip:     float = -2.0
    feedback_dislike:  float = -8.0
    feedback_save:     float =  7.0
    feedback_purchase: float = 15.0

    # Shaping component weights
    w_compatibility:  float = 0.4   # color + style + formality
    w_occasion:       float = 0.3   # occasion match
    w_budget:         float = 0.2   # budget compliance
    w_diversity:      float = 0.3   # diversity from previous recs

    # Penalties
    budget_violation_penalty: float = -8.0
    poor_compat_penalty:      float = -3.0
    repeated_outfit_penalty:  float = -2.0
    incomplete_outfit_penalty: float = -1.0
    invalid_action_penalty:   float = -0.5

    # Shaping step bonus (given for each valid item addition)
    step_shaping_scale: float = 1.0

    # Whether to use diversity component
    use_diversity: bool = True

    def feedback_to_reward(self, feedback: str) -> float:
        mapping = {
            "love":     self.feedback_love,
            "like":     self.feedback_like,
            "neutral":  self.feedback_neutral,
            "skip":     self.feedback_skip,
            "dislike":  self.feedback_dislike,
            "save":     self.feedback_save,
            "purchase": self.feedback_purchase,
        }
        return mapping.get(feedback, 0.0)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d: dict) -> "RewardConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ── Default config (used throughout unless overridden) ────────────────────────
DEFAULT_CONFIG = RewardConfig()


# ── Reward Calculator ─────────────────────────────────────────────────────────

class RewardCalculator:
    """
    Computes step and terminal rewards for the FashionVerse MDP.

    Step reward (shaping):
      Given when an item is added to the outfit.
      Provides dense feedback to prevent sparse reward problem.

    Terminal reward:
      Given when the outfit is presented to the simulated/real user.
      Based on explicit user feedback + outfit-level evaluation.
    """

    def __init__(self, config: Optional[RewardConfig] = None):
        self.config = config or DEFAULT_CONFIG

    # ── Step Reward ──────────────────────────────────────────────────────

    def step_reward(
        self,
        new_item,                     # FashionItem just added
        current_items: List,          # items including new_item
        constraints,                  # FashionConstraints
        recent_item_ids: List[str],   # previously recommended item IDs
        occasion: str = "casual",
    ) -> Tuple[float, dict]:
        """
        Returns (reward: float, info: dict) for a single step.
        """
        from backend.fashion.compatibility import score_outfit, score_pair
        cfg = self.config

        info = {
            "compat_reward": 0.0,
            "occasion_reward": 0.0,
            "budget_reward": 0.0,
            "diversity_reward": 0.0,
            "penalties": 0.0,
        }
        reward = 0.0

        # 1. Compatibility shaping
        if len(current_items) >= 2:
            compat_scores = score_outfit(current_items, occasion)
            compat_r = cfg.w_compatibility * compat_scores["overall"] * cfg.step_shaping_scale
            reward += compat_r
            info["compat_reward"] = compat_r

            if compat_scores["overall"] < 0.35:
                reward += cfg.poor_compat_penalty * 0.5
                info["penalties"] += cfg.poor_compat_penalty * 0.5

        # 2. Occasion match shaping
        occ_match = float(occasion.lower() in [o.lower() for o in new_item.occasion])
        occ_r = cfg.w_occasion * occ_match * cfg.step_shaping_scale
        reward += occ_r
        info["occasion_reward"] = occ_r

        # 3. Budget compliance bonus
        total_spent = sum(i.price for i in current_items)
        if total_spent <= constraints.budget:
            budget_ratio = total_spent / constraints.budget
            budget_r = cfg.w_budget * (1.0 - budget_ratio * 0.3) * cfg.step_shaping_scale
            reward += budget_r
            info["budget_reward"] = budget_r
        else:
            reward += cfg.budget_violation_penalty
            info["penalties"] += cfg.budget_violation_penalty

        # 4. Diversity bonus (penalize repeated items)
        if cfg.use_diversity and new_item.item_id in recent_item_ids:
            reward += cfg.repeated_outfit_penalty
            info["penalties"] += cfg.repeated_outfit_penalty
        elif cfg.use_diversity:
            div_r = cfg.w_diversity * 0.5 * cfg.step_shaping_scale
            reward += div_r
            info["diversity_reward"] = div_r

        return reward, info

    # ── Terminal Reward ───────────────────────────────────────────────────

    def terminal_reward(
        self,
        feedback: str,
        items: List,
        constraints,
        recent_item_ids: List[str],
        occasion: str = "casual",
        is_complete: bool = True,
    ) -> Tuple[float, dict]:
        """
        Returns (reward: float, info: dict) at episode termination.
        This is the primary user satisfaction signal.
        """
        from backend.fashion.compatibility import score_outfit, is_complete as check_complete
        cfg = self.config
        info = {
            "user_satisfaction_reward": 0.0,
            "compat_bonus": 0.0,
            "occasion_bonus": 0.0,
            "budget_bonus": 0.0,
            "diversity_bonus": 0.0,
            "penalties": 0.0,
        }
        reward = 0.0

        # 1. User feedback reward (primary signal)
        u_reward = cfg.feedback_to_reward(feedback)
        reward += u_reward
        info["user_satisfaction_reward"] = u_reward

        if not items:
            return reward, info

        # 2. Outfit-level compatibility bonus
        compat = score_outfit(items, occasion)
        c_bonus = cfg.w_compatibility * compat["overall"]
        reward += c_bonus
        info["compat_bonus"] = c_bonus

        if compat["overall"] < 0.30:
            reward += cfg.poor_compat_penalty
            info["penalties"] += cfg.poor_compat_penalty

        # 3. Occasion bonus
        occ_bonus = cfg.w_occasion * compat["occasion"]
        reward += occ_bonus
        info["occasion_bonus"] = occ_bonus

        # 4. Budget compliance bonus
        total_spent = sum(i.price for i in items)
        if total_spent <= constraints.budget:
            b_bonus = cfg.w_budget * 2.0
            reward += b_bonus
            info["budget_bonus"] = b_bonus
        else:
            reward += cfg.budget_violation_penalty
            info["penalties"] += cfg.budget_violation_penalty

        # 5. Diversity bonus at terminal
        if cfg.use_diversity:
            outfit_ids = {i.item_id for i in items}
            overlap = len(outfit_ids & set(recent_item_ids))
            div_ratio = 1.0 - overlap / max(len(outfit_ids), 1)
            d_bonus = cfg.w_diversity * div_ratio
            reward += d_bonus
            info["diversity_bonus"] = d_bonus

        # 6. Incomplete outfit penalty
        if not is_complete:
            reward += cfg.incomplete_outfit_penalty
            info["penalties"] += cfg.incomplete_outfit_penalty

        return reward, info

    def invalid_action_reward(self) -> float:
        return self.config.invalid_action_penalty

    def total_info(self, step_infos: List[dict], terminal_info: dict) -> dict:
        """Aggregates step and terminal info for logging."""
        total = {
            "total_step_compat": sum(i.get("compat_reward", 0) for i in step_infos),
            "total_step_occasion": sum(i.get("occasion_reward", 0) for i in step_infos),
            "total_step_budget": sum(i.get("budget_reward", 0) for i in step_infos),
            "total_penalties": (
                sum(i.get("penalties", 0) for i in step_infos) +
                terminal_info.get("penalties", 0)
            ),
            "user_reward": terminal_info.get("user_satisfaction_reward", 0),
            "terminal_compat": terminal_info.get("compat_bonus", 0),
        }
        return total


