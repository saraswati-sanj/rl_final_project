"""
FashionVerse — Outfit Generator
Generates candidate outfits using rule-based logic and compatibility scoring.
Used as:
  (a) a baseline recommender
  (b) a candidate pool for the RL agent's action space
"""

import random
from typing import List, Optional, Tuple, Dict
from backend.fashion.catalog import FashionItem, get_catalog
from backend.fashion.compatibility import OutfitCompatibilityScorer, is_complete
from backend.fashion.constraints import FashionConstraints


class OutfitGenerator:
    """
    Generates outfits by:
      1. Filtering catalog by constraints
      2. Picking best-compatible items greedily or randomly
    Provides both a rule-based (greedy) and a random approach for baselines.
    """

    def __init__(self):
        self.catalog = get_catalog()
        self.scorer = OutfitCompatibilityScorer()

    # ── Public API ────────────────────────────────────────────────────────

    def generate_random(
        self,
        constraints: FashionConstraints,
        seed: Optional[int] = None,
    ) -> List[FashionItem]:
        """
        Randomly selects a valid outfit within constraints.
        Used as the Random baseline in evaluations.
        """
        rng = random.Random(seed)
        return self._build_outfit(constraints, strategy="random", rng=rng)

    def generate_rule_based(
        self,
        constraints: FashionConstraints,
        seed: Optional[int] = None,
    ) -> List[FashionItem]:
        """
        Greedily selects the highest-compatibility outfit within constraints.
        Used as the Rule-Based baseline.
        """
        rng = random.Random(seed)
        return self._build_outfit(constraints, strategy="greedy", rng=rng)

    def generate_popular(
        self,
        constraints: FashionConstraints,
    ) -> List[FashionItem]:
        """
        Selects the most popular items within constraints.
        Used as the Popularity-Based baseline.
        """
        return self._build_outfit(constraints, strategy="popular", rng=random.Random(0))

    # ── Internal builder ─────────────────────────────────────────────────

    def _build_outfit(
        self,
        c: FashionConstraints,
        strategy: str,
        rng: random.Random,
    ) -> List[FashionItem]:
        """
        Staged outfit construction:
          Stage 1: Top OR Dress
          Stage 2: Bottom (if top was chosen)
          Stage 3: Shoes
          Stage 4: Optional accessory
        """
        outfit: List[FashionItem] = []
        used_ids = set()

        def pool(category, extra_filter=None):
            items = self.catalog.filter(
                category=category,
                occasion=c.occasion,
                season=c.season,
                max_budget=c.budget_remaining(outfit),
                gender=c.gender,
                formality_min=c.formality_min,
                formality_max=c.formality_max,
                exclude_ids=used_ids,
            )
            if extra_filter:
                items = [i for i in items if extra_filter(i)]
            return items

        def pick(items: List[FashionItem], anchor: Optional[FashionItem] = None) -> Optional[FashionItem]:
            if not items:
                return None
            if strategy == "random":
                return rng.choice(items)
            elif strategy == "popular":
                return max(items, key=lambda x: x.popularity_score)
            else:  # greedy / compatibility-based
                if anchor is None:
                    return max(items, key=lambda x: x.versatility_score * 0.5 + x.comfort_score * 0.5)
                return max(items, key=lambda x: self.scorer.score_pair(anchor, x))

        # Stage 1: Dress or Top
        use_dress = (c.occasion in ("formal", "party", "festive", "semi_formal", "date")
                     and rng.random() < 0.35)
        if use_dress:
            dresses = pool("dress")
            chosen = pick(dresses)
            if chosen:
                outfit.append(chosen)
                used_ids.add(chosen.item_id)
        if not outfit:  # fallback to top
            tops = pool("top")
            chosen = pick(tops)
            if chosen:
                outfit.append(chosen)
                used_ids.add(chosen.item_id)

        # Stage 2: Bottom (if no dress)
        if outfit and outfit[0].category == "top":
            anchor = outfit[0]
            bottoms = pool("bottom")
            chosen = pick(bottoms, anchor)
            if chosen:
                outfit.append(chosen)
                used_ids.add(chosen.item_id)

        # Stage 3: Shoes
        anchor = outfit[-1] if outfit else None
        shoes = pool("shoes")
        chosen = pick(shoes, anchor)
        if chosen:
            outfit.append(chosen)
            used_ids.add(chosen.item_id)

        # Stage 4: Accessory (optional, ~60% chance)
        if rng.random() < 0.60 and c.budget_remaining(outfit) > 100:
            anchor = outfit[0] if outfit else None
            accs = pool("accessory")
            chosen = pick(accs, anchor)
            if chosen:
                outfit.append(chosen)
                used_ids.add(chosen.item_id)

        return outfit

    # ── Utility ──────────────────────────────────────────────────────────

    def score_outfit(self, items: List[FashionItem], occasion: str) -> Dict:
        return self.scorer.score_outfit(items, occasion)

    def outfit_summary(self, items: List[FashionItem]) -> Dict:
        total_price = sum(i.price for i in items)
        score = self.scorer.score_outfit(items)
        return {
            "items": [{"id": i.item_id, "name": i.name, "category": i.category,
                        "price": i.price, "color": i.color} for i in items],
            "total_price": total_price,
            "compatibility_score": score["overall"],
            "is_complete": is_complete(items),
        }
