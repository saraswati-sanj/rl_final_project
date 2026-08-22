"""
FashionVerse — Constraint Manager
Validates outfit selections against user-specified hard constraints:
  budget, occasion, season, gender.
Used by RL environment and outfit generator.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from backend.fashion.catalog import FashionItem


@dataclass
class FashionConstraints:
    """
    Hard constraints derived from a user's fashion request.
    The RL agent's action must satisfy these before reward is computed.
    """
    budget: int = 3000          # total INR budget for outfit
    occasion: str = "casual"
    season: str = "all"
    gender: str = "unisex"
    formality_min: int = 1
    formality_max: int = 5
    style_preference: Optional[str] = None
    color_preference: Optional[str] = None
    max_items: int = 4           # top + bottom/dress + shoes + optional acc.

    def budget_remaining(self, items: List[FashionItem]) -> int:
        spent = sum(i.price for i in items)
        return self.budget - spent

    def is_over_budget(self, items: List[FashionItem]) -> bool:
        return sum(i.price for i in items) > self.budget

    def validate_item(self, item: FashionItem, current_items: List[FashionItem]) -> dict:
        """
        Check whether adding `item` to current_items violates constraints.
        Returns dict with 'valid' bool and 'reason' string.
        """
        if item.price > self.budget_remaining(current_items):
            return {"valid": False, "reason": "over_budget"}

        if self.gender != "unisex" and item.gender not in (self.gender, "unisex"):
            return {"valid": False, "reason": "gender_mismatch"}

        if self.formality_min <= item.formality <= self.formality_max:
            pass  # ok
        else:
            return {"valid": False, "reason": "formality_mismatch"}

        # Duplicate category guard (except accessory – can have one)
        cat = item.category
        existing_cats = [i.category for i in current_items]
        if cat in ("top", "bottom", "shoes") and existing_cats.count(cat) >= 1:
            return {"valid": False, "reason": f"duplicate_{cat}"}
        if cat == "accessory" and existing_cats.count(cat) >= 1:
            return {"valid": False, "reason": "duplicate_accessory"}
        if cat == "dress" and ("top" in existing_cats or "dress" in existing_cats):
            return {"valid": False, "reason": "dress_conflict"}
        if cat == "top" and "dress" in existing_cats:
            return {"valid": False, "reason": "dress_conflict"}

        return {"valid": True, "reason": "ok"}

    def to_dict(self) -> dict:
        return {
            "budget": self.budget,
            "occasion": self.occasion,
            "season": self.season,
            "gender": self.gender,
            "formality_min": self.formality_min,
            "formality_max": self.formality_max,
            "style_preference": self.style_preference,
            "color_preference": self.color_preference,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FashionConstraints":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
