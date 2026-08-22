"""
FashionVerse — User Simulator
Creates simulated users with hidden preferences for RL training.

Design principle:
  The RL agent NEVER sees the hidden preference vector directly.
  It must infer preferences from the feedback signals (reward) it receives.

Each simulator:
  - Has a personality type with hidden style, color, and budget preferences.
  - Evaluates an outfit and returns a stochastic feedback signal.
  - Supports preference drift (changing tastes over time).
"""

import random
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from backend.fashion.catalog import FashionItem
from backend.fashion.compatibility import score_outfit


# ── Feedback signal constants ─────────────────────────────────────────────────

FEEDBACK_LOVE     = "love"
FEEDBACK_LIKE     = "like"
FEEDBACK_NEUTRAL  = "neutral"
FEEDBACK_SKIP     = "skip"
FEEDBACK_DISLIKE  = "dislike"
FEEDBACK_SAVE     = "save"
FEEDBACK_PURCHASE = "purchase"

FEEDBACK_VALUES = {
    FEEDBACK_LOVE:     1.0,
    FEEDBACK_LIKE:     0.7,
    FEEDBACK_NEUTRAL:  0.4,
    FEEDBACK_SKIP:     0.2,
    FEEDBACK_DISLIKE: -0.3,
    FEEDBACK_SAVE:     0.85,
    FEEDBACK_PURCHASE: 1.0,
}


# ── User Personality Profiles ─────────────────────────────────────────────────

USER_PERSONALITIES = {
    "casual_student": {
        "description": "College student who prefers casual, comfortable, affordable looks",
        "style_prefs": {"casual": 0.85, "streetwear": 0.70, "formal": 0.10,
                         "semi_formal": 0.20, "minimalist": 0.55, "bohemian": 0.30},
        "color_prefs": {"black": 0.90, "white": 0.80, "navy": 0.75, "grey": 0.70,
                         "bright_colors": 0.20, "pastels": 0.35},
        "formality_ideal": 1.5,
        "budget": 2000,
        "comfort_weight": 0.80,
        "trend_weight": 0.50,
    },
    "formal_professional": {
        "description": "Working professional who prefers clean, formal looks",
        "style_prefs": {"formal": 0.90, "semi_formal": 0.80, "minimalist": 0.70,
                         "casual": 0.15, "streetwear": 0.05},
        "color_prefs": {"black": 0.85, "navy": 0.80, "grey": 0.75, "white": 0.90,
                         "beige": 0.65, "bright_colors": 0.10},
        "formality_ideal": 4.5,
        "budget": 4000,
        "comfort_weight": 0.50,
        "trend_weight": 0.30,
    },
    "minimalist": {
        "description": "Prefers clean lines, neutral colors, capsule wardrobe style",
        "style_prefs": {"minimalist": 0.95, "formal": 0.50, "casual": 0.60,
                         "semi_formal": 0.55, "streetwear": 0.20},
        "color_prefs": {"black": 0.90, "white": 0.90, "grey": 0.85, "beige": 0.80,
                         "navy": 0.70, "cream": 0.75, "bright_colors": 0.05},
        "formality_ideal": 2.5,
        "budget": 3000,
        "comfort_weight": 0.65,
        "trend_weight": 0.20,
    },
    "streetwear_enthusiast": {
        "description": "Loves bold streetwear, graphic tees, sneakers, and hype brands",
        "style_prefs": {"streetwear": 0.95, "casual": 0.75, "athleisure": 0.70,
                         "formal": 0.05, "bohemian": 0.10},
        "color_prefs": {"black": 0.80, "white": 0.75, "bright_colors": 0.70,
                         "grey": 0.60, "navy": 0.55},
        "formality_ideal": 1.0,
        "budget": 3500,
        "comfort_weight": 0.70,
        "trend_weight": 0.90,
    },
    "colorful_fashion_lover": {
        "description": "Loves expressive, colorful, statement outfits",
        "style_prefs": {"bohemian": 0.80, "casual": 0.65, "festive": 0.75,
                         "indo_western": 0.70, "streetwear": 0.50},
        "color_prefs": {"bright_colors": 0.95, "pastels": 0.80, "navy": 0.50,
                         "black": 0.40, "white": 0.60},
        "formality_ideal": 2.0,
        "budget": 2500,
        "comfort_weight": 0.60,
        "trend_weight": 0.75,
    },
    "budget_shopper": {
        "description": "Extremely price-sensitive, prioritizes value over style",
        "style_prefs": {"casual": 0.80, "minimalist": 0.60, "streetwear": 0.40},
        "color_prefs": {"black": 0.80, "grey": 0.70, "navy": 0.65},
        "formality_ideal": 2.0,
        "budget": 1500,
        "comfort_weight": 0.75,
        "trend_weight": 0.20,
    },
    "trend_focused": {
        "description": "Always follows current fashion trends, high engagement",
        "style_prefs": {"streetwear": 0.75, "casual": 0.70, "semi_formal": 0.65,
                         "bohemian": 0.60, "preppy": 0.55},
        "color_prefs": {"bright_colors": 0.80, "pastels": 0.75, "navy": 0.60,
                         "black": 0.50},
        "formality_ideal": 2.5,
        "budget": 3000,
        "comfort_weight": 0.50,
        "trend_weight": 0.95,
    },
}

COLOR_WARMTH = {
    "black": "neutral", "white": "neutral", "grey": "neutral",
    "beige": "neutral", "cream": "neutral", "charcoal": "neutral", "tan": "neutral",
    "navy": "cool", "royal_blue": "cool", "teal": "cool", "sky_blue": "cool",
    "slate": "cool", "indigo": "cool", "lavender": "cool",
    "forest_green": "cool", "sage_green": "cool",
    "maroon": "warm", "wine": "warm", "rust": "warm",
    "burnt_orange": "warm", "coral": "warm", "dusty_rose": "warm", "blush": "warm",
    "olive": "earth", "mustard": "earth",
    "mustard_yellow": "bright", "hot_pink": "bright", "lime_green": "bright",
}


@dataclass
class SimulatedUser:
    """
    A simulated user with hidden preferences for training the RL agent.
    The agent cannot access hidden_prefs directly — it must learn from feedback.
    """
    user_id: str
    personality: str

    # Hidden preferences (NOT visible to RL agent)
    hidden_prefs: Dict = field(default_factory=dict)
    budget: int = 2500
    interaction_count: int = 0
    preference_version: int = 0   # increments when prefs drift

    # Noise level (0 = deterministic, 1 = very noisy)
    noise_level: float = 0.15

    # History of feedback (item_id → feedback)
    feedback_history: Dict = field(default_factory=dict)
    recently_seen_outfits: List = field(default_factory=list)

    def evaluate_outfit(
        self,
        items: List[FashionItem],
        occasion: str = "casual",
    ) -> Tuple[str, float]:
        """
        Evaluates a presented outfit and returns (feedback_label, satisfaction_score).
        Satisfaction is in [0, 1]. Feedback is a discrete label.
        The RL agent should NOT call this directly — only the environment can.
        """
        if not items:
            return FEEDBACK_SKIP, 0.0

        score = self._compute_satisfaction(items, occasion)

        # Add Gaussian noise to make env stochastic
        noise = random.gauss(0, self.noise_level)
        score_noisy = max(0.0, min(1.0, score + noise))

        label = self._score_to_feedback(score_noisy, items)

        # Track history
        outfit_key = "|".join(sorted(i.item_id for i in items))
        self.feedback_history[outfit_key] = label
        self.recently_seen_outfits.append(outfit_key)
        if len(self.recently_seen_outfits) > 20:
            self.recently_seen_outfits.pop(0)
        self.interaction_count += 1

        return label, score_noisy

    def _compute_satisfaction(self, items: List[FashionItem], occasion: str) -> float:
        """Hidden scoring based on user's true preferences."""
        prefs = self.hidden_prefs
        n = len(items)

        # Style match score
        style_score = 0.0
        for item in items:
            style_score += prefs.get("style_prefs", {}).get(item.style, 0.3)
        style_score /= n

        # Color match score
        color_score = 0.0
        for item in items:
            c = item.color
            warmth = COLOR_WARMTH.get(c, "bright")
            likes_bright = prefs.get("color_prefs", {}).get("bright_colors", 0.5)
            likes_pastels = prefs.get("color_prefs", {}).get("pastels", 0.5)
            base = prefs.get("color_prefs", {}).get(c, None)
            if base is None:
                if warmth == "bright":
                    base = likes_bright
                elif warmth in ("cool", "warm", "earth"):
                    base = 0.6
                else:
                    base = 0.5
            color_score += base
        color_score /= n

        # Formality preference
        ideal_form = prefs.get("formality_ideal", 2.5)
        form_scores = []
        for item in items:
            diff = abs(item.formality - ideal_form)
            form_scores.append(max(0, 1.0 - diff / 4.0))
        formality_score = sum(form_scores) / len(form_scores)

        # Budget satisfaction
        total_price = sum(i.price for i in items)
        budget = prefs.get("budget", 2500)
        if total_price <= budget:
            budget_score = 1.0 - 0.3 * (total_price / budget - 0.5) ** 2
        else:
            overshoot = (total_price - budget) / budget
            budget_score = max(0.0, 1.0 - overshoot * 2.0)

        # Comfort and trend
        avg_comfort = sum(i.comfort_score for i in items) / n
        avg_trend = sum(i.trend_score for i in items) / n
        comfort_weight = prefs.get("comfort_weight", 0.6)
        trend_weight = prefs.get("trend_weight", 0.5)
        quality_score = comfort_weight * avg_comfort + (1 - comfort_weight) * avg_trend

        # Compatibility bonus
        compat = score_outfit(items, occasion)["overall"]

        # Repetition penalty
        outfit_key = "|".join(sorted(i.item_id for i in items))
        repetition_penalty = 0.3 if outfit_key in self.recently_seen_outfits[:-1] else 0.0

        # Weighted aggregate
        total = (
            0.30 * style_score +
            0.20 * color_score +
            0.15 * formality_score +
            0.20 * budget_score +
            0.10 * quality_score +
            0.05 * compat -
            repetition_penalty
        )
        return max(0.0, min(1.0, total))

    def _score_to_feedback(self, score: float, items: List[FashionItem]) -> str:
        """Maps satisfaction score to discrete feedback label stochastically."""
        if score >= 0.88:
            return random.choices(
                [FEEDBACK_LOVE, FEEDBACK_PURCHASE, FEEDBACK_SAVE],
                weights=[0.50, 0.30, 0.20]
            )[0]
        elif score >= 0.72:
            return random.choices(
                [FEEDBACK_SAVE, FEEDBACK_LIKE, FEEDBACK_LOVE],
                weights=[0.40, 0.45, 0.15]
            )[0]
        elif score >= 0.55:
            return random.choices(
                [FEEDBACK_LIKE, FEEDBACK_NEUTRAL],
                weights=[0.60, 0.40]
            )[0]
        elif score >= 0.40:
            return random.choices(
                [FEEDBACK_NEUTRAL, FEEDBACK_SKIP],
                weights=[0.50, 0.50]
            )[0]
        elif score >= 0.25:
            return random.choices(
                [FEEDBACK_SKIP, FEEDBACK_DISLIKE],
                weights=[0.60, 0.40]
            )[0]
        else:
            return FEEDBACK_DISLIKE

    def apply_preference_drift(self, drift_rate: float = 0.1):
        """
        Simulates preference change over time. Used in Experiment 2.
        Gradually shifts style and color preferences.
        """
        prefs = self.hidden_prefs
        for style in prefs.get("style_prefs", {}):
            delta = random.uniform(-drift_rate, drift_rate)
            prefs["style_prefs"][style] = max(0.0, min(1.0, prefs["style_prefs"][style] + delta))
        self.preference_version += 1

    def get_observable_state_hint(self) -> Dict:
        """
        Returns ONLY the observable signals the RL agent CAN use:
          - interaction count
          - recent feedback
          - inferred budget estimate
        The agent does NOT see style_prefs, color_prefs, etc.
        """
        return {
            "interaction_count": self.interaction_count,
            "recent_feedback_count": len(self.feedback_history),
            "preference_version": self.preference_version,
        }


class UserSimulator:
    """Factory that creates and manages simulated users for RL training."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def create_user(
        self,
        user_id: str,
        personality: Optional[str] = None,
        noise_level: float = 0.15,
    ) -> SimulatedUser:
        if personality is None:
            personality = self.rng.choice(list(USER_PERSONALITIES.keys()))

        profile = USER_PERSONALITIES[personality]

        user = SimulatedUser(
            user_id=user_id,
            personality=personality,
            hidden_prefs={
                "style_prefs": dict(profile["style_prefs"]),
                "color_prefs": dict(profile["color_prefs"]),
                "formality_ideal": profile["formality_ideal"],
                "budget": profile["budget"],
                "comfort_weight": profile["comfort_weight"],
                "trend_weight": profile["trend_weight"],
            },
            budget=profile["budget"],
            noise_level=noise_level,
        )
        return user

    def create_batch(self, n: int, noise_level: float = 0.15) -> List[SimulatedUser]:
        """Create n simulated users covering all personality types."""
        users = []
        personalities = list(USER_PERSONALITIES.keys())
        for i in range(n):
            personality = personalities[i % len(personalities)]
            user = self.create_user(
                user_id=f"sim_user_{i:04d}",
                personality=personality,
                noise_level=noise_level,
            )
            users.append(user)
        return users

    def list_personalities(self) -> List[str]:
        return list(USER_PERSONALITIES.keys())
