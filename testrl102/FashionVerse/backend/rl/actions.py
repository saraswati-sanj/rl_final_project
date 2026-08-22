"""
FashionVerse — RL Action Space

Design decision: Hierarchical / Staged Discrete Actions
==========================================================
Why NOT a flat action space?
  - Flat space would be: select any of 360 items × 6 slots = 2160+ actions
  - PPO with 2160+ actions requires huge networks and slow convergence
  - Most actions are invalid (e.g., selecting bottom before top)

Chosen design: STAGED DISCRETE ACTIONS
  At each step the agent picks ONE action from a FIXED-SIZE discrete set.
  The action has two parts:
    1. SLOT (what kind of item to add/replace/finish)
    2. CANDIDATE INDEX (which item from a filtered candidate list)

  The environment pre-filters candidates based on current constraints,
  budget, and what has already been selected. The agent's network maps
  the state to an index into this candidate list (max N_CANDIDATES per slot).

  This keeps the action space fixed and small:
    N_ACTION_TYPES   =  6 (select_top, select_bottom, select_dress,
                            select_shoes, select_accessory, finish_outfit)
    N_CANDIDATES     = 10  (top-K items per slot, pre-filtered)
    Total flat dim   = N_ACTION_TYPES × N_CANDIDATES + 1 (finish)
                     = 61

  The agent outputs a single integer in [0, ACTION_DIM-1].
  The environment decodes it back to (action_type, candidate_index).

Academic note:
  This is a form of parameterized/hierarchical action decomposition,
  widely used in large-item recommendation RL (e.g., SlateQ, REINFORCE
  for list recommendation). The key property is that the MDP remains
  a proper Markov process even with this decomposition.
"""

from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import numpy as np

# ── Constants ──────────────────────────────────────────────────────────────────
N_CANDIDATES     = 10      # candidate items per slot (top-K after filtering)
ACTION_TYPES     = [
    "select_top",
    "select_bottom",
    "select_dress",
    "select_shoes",
    "select_accessory",
    "finish_outfit",
]
N_ACTION_TYPES   = len(ACTION_TYPES)       # 6
ACTION_DIM       = N_ACTION_TYPES * N_CANDIDATES  # 60
# finish_outfit is the last action of each type × candidate block
# We encode: action_id = type_idx * N_CANDIDATES + candidate_idx
# For finish_outfit: type_idx=5, candidate_idx=0  → action_id=50
FINISH_ACTION_ID = ACTION_TYPES.index("finish_outfit") * N_CANDIDATES  # 50

SLOT_TO_CATEGORY = {
    "select_top":       "top",
    "select_bottom":    "bottom",
    "select_dress":     "dress",
    "select_shoes":     "shoes",
    "select_accessory": "accessory",
}


@dataclass
class ActionCandidate:
    """One pre-filtered candidate item for a given slot."""
    item_id: str
    item: object        # FashionItem
    slot: str
    score: float = 0.0  # pre-computed relevance score


@dataclass
class DecodedAction:
    """Result of decoding an integer action ID."""
    action_id: int
    action_type: str       # one of ACTION_TYPES
    candidate_idx: int
    item: Optional[object] = None   # FashionItem or None for finish
    is_finish: bool = False
    is_valid: bool = True
    invalid_reason: str = ""


class ActionSpace:
    """
    Manages the construction, encoding, and decoding of actions
    for the FashionVerse Gymnasium environment.

    At each step, the environment calls `build_candidates()` to create
    a fresh candidate list from the current state. The agent's integer
    output is decoded via `decode()`.
    """

    def __init__(self, catalog, n_candidates: int = N_CANDIDATES):
        self.catalog = catalog
        self.n_candidates = n_candidates
        self._candidates: Dict[str, List[ActionCandidate]] = {}

    def build_candidates(
        self,
        constraints,
        current_items: List,
        scorer=None,
    ) -> Dict[str, List[ActionCandidate]]:
        """
        Pre-filter and rank candidate items for each slot given the
        current state. Called by the environment at each step.

        Ranking: compatibility with current outfit × versatility.
        """
        from backend.fashion.compatibility import score_pair
        exclude_ids = {i.item_id for i in current_items}
        existing_cats = {i.category for i in current_items}
        has_dress = "dress" in existing_cats
        has_top   = "top"   in existing_cats

        candidates: Dict[str, List[ActionCandidate]] = {}

        for slot, category in SLOT_TO_CATEGORY.items():
            # Skip logically impossible slots
            if slot == "select_top"    and (has_dress or "top" in existing_cats):
                candidates[slot] = []
                continue
            if slot == "select_bottom" and ("bottom" in existing_cats or has_dress):
                candidates[slot] = []
                continue
            if slot == "select_dress"  and (has_top or has_dress):
                candidates[slot] = []
                continue
            if slot == "select_shoes"  and "shoes" in existing_cats:
                candidates[slot] = []
                continue
            if slot == "select_accessory" and "accessory" in existing_cats:
                candidates[slot] = []
                continue

            items = self.catalog.filter(
                category=category,
                occasion=constraints.occasion,
                season=constraints.season,
                max_budget=constraints.budget_remaining(current_items),
                gender=constraints.gender,
                formality_min=constraints.formality_min,
                formality_max=constraints.formality_max,
                exclude_ids=exclude_ids,
            )

            # Score and rank candidates
            anchor = current_items[-1] if current_items else None
            scored = []
            for item in items:
                compat = score_pair(anchor, item) if anchor else item.versatility_score
                score = 0.6 * compat + 0.4 * item.versatility_score
                scored.append(ActionCandidate(
                    item_id=item.item_id,
                    item=item,
                    slot=slot,
                    score=score,
                ))
            # Top-K by score
            scored.sort(key=lambda x: x.score, reverse=True)
            candidates[slot] = scored[:self.n_candidates]

        self._candidates = candidates
        return candidates

    def decode(self, action_id: int) -> DecodedAction:
        """
        Decode an integer action_id from [0, ACTION_DIM) into
        an (action_type, candidate) pair.
        """
        action_id = int(action_id)
        if action_id < 0 or action_id >= ACTION_DIM:
            return DecodedAction(
                action_id=action_id,
                action_type="invalid",
                candidate_idx=0,
                is_valid=False,
                invalid_reason=f"action_id {action_id} out of range [0, {ACTION_DIM})",
            )

        type_idx = action_id // self.n_candidates
        cand_idx = action_id % self.n_candidates
        action_type = ACTION_TYPES[type_idx]

        if action_type == "finish_outfit":
            return DecodedAction(
                action_id=action_id,
                action_type="finish_outfit",
                candidate_idx=cand_idx,
                item=None,
                is_finish=True,
                is_valid=True,
            )

        slot_candidates = self._candidates.get(action_type, [])
        if cand_idx >= len(slot_candidates):
            return DecodedAction(
                action_id=action_id,
                action_type=action_type,
                candidate_idx=cand_idx,
                is_valid=False,
                invalid_reason=f"No candidate at index {cand_idx} for slot {action_type}",
            )

        candidate = slot_candidates[cand_idx]
        return DecodedAction(
            action_id=action_id,
            action_type=action_type,
            candidate_idx=cand_idx,
            item=candidate.item,
            is_finish=False,
            is_valid=True,
        )

    def get_action_mask(self) -> np.ndarray:
        """
        Returns a boolean mask of shape (ACTION_DIM,) where True means
        the action is currently valid. Used for masked PPO.
        """
        mask = np.zeros(ACTION_DIM, dtype=bool)

        # finish_outfit is always valid
        mask[FINISH_ACTION_ID] = True

        for slot, candidates in self._candidates.items():
            if slot not in ACTION_TYPES:
                continue
            type_idx = ACTION_TYPES.index(slot)
            for cand_idx in range(len(candidates)):
                a_id = type_idx * self.n_candidates + cand_idx
                mask[a_id] = True

        return mask

    def n_valid_actions(self) -> int:
        return int(self.get_action_mask().sum())

    def sample_valid_action(self, rng=None) -> int:
        """Sample a uniformly random valid action. Used in random baseline."""
        import random
        mask = self.get_action_mask()
        valid_ids = [i for i, m in enumerate(mask) if m]
        if not valid_ids:
            return FINISH_ACTION_ID
        if rng:
            return rng.choice(valid_ids)
        return random.choice(valid_ids)
