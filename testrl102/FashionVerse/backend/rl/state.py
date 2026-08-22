"""
FashionVerse — RL State Representation
Encodes the full MDP state as a flat numpy array consumed by the PPO policy network.

State vector dimensions (documented):
  [0:23]   User profile estimates (styles×9, colors×7, interaction stats×7)
  [23:29]  Fashion request context (occasion×5-hot, season×4-hot, budget_norm, formality target×2)
  [29:51]  Current outfit-so-far (top_vec×11, bottom/dress_vec×11)
  [51:62]  Last selected shoes vector (11)
  [62:65]  Budget remaining (norm), outfit_completeness, n_items_selected (norm)
  [65:70]  Recent reward history (last 5 rewards, normalized)

Total: 70 dimensions (STATE_DIM = 70)
"""

import numpy as np
from typing import List, Optional, Dict
from dataclasses import dataclass, field

# ── Constants ──────────────────────────────────────────────────────────────────
STATE_DIM = 70
MAX_BUDGET = 5000.0
MAX_ITEMS = 4
REWARD_HISTORY_LEN = 5

OCCASION_LIST = ["casual", "college", "office", "semi_formal", "formal",
                 "party", "festive", "date", "travel", "gym"]
SEASON_LIST   = ["summer", "winter", "monsoon", "all"]

ZERO_ITEM_VEC = [0.0] * 11   # placeholder when no item selected


def one_hot(value: str, vocab: List[str]) -> List[float]:
    """Returns a one-hot encoding; all zeros if value not in vocab."""
    return [1.0 if v == value else 0.0 for v in vocab]


@dataclass
class FashionState:
    """
    Full MDP state at decision step t.
    Stores both structured data (for logic) and the numpy encoding (for RL).
    """
    # ── Structured fields (used by env logic) ─────────────────────────────
    user_profile_vec: List[float] = field(default_factory=lambda: [0.5] * 23)
    occasion: str = "casual"
    season: str = "all"
    budget_total: int = 2500
    budget_remaining: int = 2500
    formality_target: float = 2.5     # 1–5

    # Items selected so far in this episode
    selected_items: List = field(default_factory=list)  # List[FashionItem]
    step_in_episode: int = 0
    episode_done: bool = False

    # Recent reward history (for the agent to see its own trajectory)
    reward_history: List[float] = field(default_factory=lambda: [0.0] * REWARD_HISTORY_LEN)

    def encode(self) -> np.ndarray:
        """
        Encode the state to a flat numpy float32 array of shape (STATE_DIM,).
        This is passed directly to the PPO policy network.
        """
        vec = []

        # ── Segment 1: User profile (23) ──────────────────────────────────
        vec.extend(self.user_profile_vec[:23])

        # ── Segment 2: Request context (14) ───────────────────────────────
        occ_hot = one_hot(self.occasion, OCCASION_LIST[:5])         # 5
        sea_hot = one_hot(self.season, SEASON_LIST)                  # 4
        budget_norm = self.budget_total / MAX_BUDGET                 # 1
        formality_norm = self.formality_target / 5.0                 # 1
        formality_low  = max(0.0, (self.formality_target - 1) / 4)  # 1
        formality_high = min(1.0, self.formality_target / 5)        # 1
        context_vec = occ_hot + sea_hot + [budget_norm, formality_norm,
                                            formality_low, formality_high]  # 14
        vec.extend(context_vec)

        # ── Segment 3: Current outfit so far (3 × 11 = 33) ───────────────
        # Slots: top/dress, bottom, shoes
        top_vec = ZERO_ITEM_VEC[:]
        bot_vec = ZERO_ITEM_VEC[:]
        sho_vec = ZERO_ITEM_VEC[:]

        for item in self.selected_items:
            iv = item.to_vector()
            if item.category in ("top", "dress"):
                top_vec = iv
            elif item.category == "bottom":
                bot_vec = iv
            elif item.category == "shoes":
                sho_vec = iv

        vec.extend(top_vec)   # 11
        vec.extend(bot_vec)   # 11
        vec.extend(sho_vec)   # 11

        # ── Segment 4: Progress features (4) ─────────────────────────────
        budget_remaining_norm = max(0.0, self.budget_remaining / MAX_BUDGET)
        n_items_norm = len(self.selected_items) / MAX_ITEMS
        step_norm = self.step_in_episode / 6.0
        has_shoes = float(any(i.category == "shoes" for i in self.selected_items))
        vec.extend([budget_remaining_norm, n_items_norm, step_norm, has_shoes])  # 4

        # ── Segment 5: Reward history (5) ────────────────────────────────
        # Normalize rewards to [-1, 1] approximately
        rh_norm = [max(-1.0, min(1.0, r / 15.0)) for r in self.reward_history]
        vec.extend(rh_norm)  # 5

        # Total: 23 + 14 + 33 + 4 + 5 = 79... trim/pad to STATE_DIM
        vec = vec[:STATE_DIM]
        while len(vec) < STATE_DIM:
            vec.append(0.0)

        return np.array(vec, dtype=np.float32)

    def to_dict(self) -> dict:
        return {
            "occasion": self.occasion,
            "season": self.season,
            "budget_total": self.budget_total,
            "budget_remaining": self.budget_remaining,
            "formality_target": self.formality_target,
            "n_items": len(self.selected_items),
            "selected_item_ids": [i.item_id for i in self.selected_items],
            "step": self.step_in_episode,
        }
