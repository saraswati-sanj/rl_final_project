"""
FashionVerse — User Profile
Tracks observable (RL-visible) user preference state that is updated
over interactions. This is NOT the hidden simulator vector.
It represents the RL agent's BELIEF about the user.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import json


@dataclass
class UserProfile:
    """
    Observable user preference estimate maintained by the system.
    Updated after each interaction using feedback signals.
    The RL state vector is partly derived from this profile.
    """
    user_id: str
    budget: int = 2500
    gender: str = "unisex"

    # Estimated style preference weights (sum to 1 ideally, but not enforced)
    style_estimates: Dict[str, float] = field(default_factory=lambda: {
        "casual": 0.5, "formal": 0.5, "semi_formal": 0.5,
        "streetwear": 0.5, "minimalist": 0.5, "bohemian": 0.5,
        "athleisure": 0.5, "festive": 0.5, "indo_western": 0.5,
    })

    # Estimated color preference weights
    color_estimates: Dict[str, float] = field(default_factory=lambda: {
        "black": 0.5, "white": 0.5, "navy": 0.5, "grey": 0.5,
        "beige": 0.5, "bright_colors": 0.5, "pastels": 0.5,
    })

    # Estimated formality preference
    formality_estimate: float = 2.5   # 1–5 scale

    # Interaction stats
    total_interactions: int = 0
    likes: int = 0
    dislikes: int = 0
    saves: int = 0
    purchases: int = 0
    skips: int = 0

    # Recent history for diversity tracking
    recently_recommended: List[str] = field(default_factory=list)  # item_ids

    def update_from_feedback(
        self,
        items: List,
        feedback: str,
        learning_rate: float = 0.05,
    ):
        """
        Updates preference estimates using exponential moving average.
        Called by the environment after each step.
        """
        from backend.user.user_simulator import FEEDBACK_VALUES, COLOR_WARMTH

        val = FEEDBACK_VALUES.get(feedback, 0.0)
        is_positive = val >= 0.5
        self.total_interactions += 1

        # Update feedback counters
        if feedback in ("like", "love", "save"):
            self.likes += 1
        elif feedback == "dislike":
            self.dislikes += 1
        elif feedback == "save":
            self.saves += 1
        elif feedback == "purchase":
            self.purchases += 1
        elif feedback == "skip":
            self.skips += 1

        # Update style estimates
        for item in items:
            style = item.style
            if style in self.style_estimates:
                current = self.style_estimates[style]
                target = 0.8 if is_positive else 0.2
                self.style_estimates[style] = current + learning_rate * (target - current)

            # Update color estimates
            c = item.color
            warmth = COLOR_WARMTH.get(c, "bright")
            if c in self.color_estimates:
                current = self.color_estimates[c]
                target = 0.8 if is_positive else 0.2
                self.color_estimates[c] = current + learning_rate * (target - current)
            elif warmth == "bright" and "bright_colors" in self.color_estimates:
                current = self.color_estimates["bright_colors"]
                target = 0.8 if is_positive else 0.2
                self.color_estimates["bright_colors"] = current + learning_rate * (target - current)

            # Update formality estimate
            if is_positive:
                self.formality_estimate = (
                    self.formality_estimate + learning_rate * (item.formality - self.formality_estimate)
                )

        # Track recently recommended items
        for item in items:
            self.recently_recommended.append(item.item_id)
        if len(self.recently_recommended) > 50:
            self.recently_recommended = self.recently_recommended[-50:]

    def to_vector(self) -> List[float]:
        """
        Encodes the user profile as a fixed-size normalized feature vector.
        Used as part of the RL state.
        """
        styles = ["casual", "formal", "semi_formal", "streetwear",
                  "minimalist", "bohemian", "athleisure", "festive", "indo_western"]
        style_vec = [self.style_estimates.get(s, 0.5) for s in styles]

        colors = ["black", "white", "navy", "grey", "beige", "bright_colors", "pastels"]
        color_vec = [self.color_estimates.get(c, 0.5) for c in colors]

        interaction_features = [
            self.budget / 5000.0,
            self.formality_estimate / 5.0,
            min(self.total_interactions / 100.0, 1.0),
            self.likes / max(self.total_interactions, 1),
            self.dislikes / max(self.total_interactions, 1),
            self.saves / max(self.total_interactions, 1),
            self.skips / max(self.total_interactions, 1),
        ]

        return style_vec + color_vec + interaction_features
        # Total dim: 9 + 7 + 7 = 23

    def acceptance_rate(self) -> float:
        if self.total_interactions == 0:
            return 0.0
        return (self.likes + self.saves + self.purchases) / self.total_interactions

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "budget": self.budget,
            "gender": self.gender,
            "style_estimates": self.style_estimates,
            "color_estimates": self.color_estimates,
            "formality_estimate": self.formality_estimate,
            "total_interactions": self.total_interactions,
            "likes": self.likes,
            "dislikes": self.dislikes,
            "saves": self.saves,
            "purchases": self.purchases,
            "skips": self.skips,
            "acceptance_rate": self.acceptance_rate(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "UserProfile":
        obj = cls(user_id=d["user_id"], budget=d.get("budget", 2500),
                  gender=d.get("gender", "unisex"))
        obj.style_estimates = d.get("style_estimates", obj.style_estimates)
        obj.color_estimates = d.get("color_estimates", obj.color_estimates)
        obj.formality_estimate = d.get("formality_estimate", 2.5)
        obj.total_interactions = d.get("total_interactions", 0)
        obj.likes = d.get("likes", 0)
        obj.dislikes = d.get("dislikes", 0)
        obj.saves = d.get("saves", 0)
        obj.purchases = d.get("purchases", 0)
        obj.skips = d.get("skips", 0)
        return obj
