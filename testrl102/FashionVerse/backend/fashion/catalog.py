"""
FashionVerse — Fashion Catalog Module
Loads, filters, and indexes the fashion catalog for use throughout the system.
"""

import json
import csv
import os
import random
from typing import List, Dict, Optional, Set
from dataclasses import dataclass, field


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
CATALOG_JSON = os.path.join(DATA_DIR, "fashion_items.json")
CATALOG_CSV = os.path.join(DATA_DIR, "fashion_items.csv")


@dataclass
class FashionItem:
    item_id: str
    name: str
    category: str           # top | bottom | dress | shoes | accessory
    subcategory: str
    color: str
    secondary_color: str
    style: str
    occasion: List[str]
    season: List[str]
    formality: int          # 1 (very casual) → 5 (very formal)
    pattern: str
    material: str
    price: int              # INR
    gender: str
    comfort_score: float    # 0–1
    versatility_score: float  # 0–1
    popularity_score: float   # 0–1
    trend_score: float        # 0–1
    image_placeholder: str

    def matches_occasion(self, occasion: str) -> bool:
        return occasion.lower() in [o.lower() for o in self.occasion]

    def matches_season(self, season: str) -> bool:
        return "all" in self.season or season.lower() in [s.lower() for s in self.season]

    def within_budget(self, budget: int) -> bool:
        return self.price <= budget

    def to_vector(self) -> List[float]:
        """
        Returns a normalized feature vector for RL state encoding.
        Dimensions: [price_norm, formality_norm, comfort, versatility,
                     popularity, trend, is_casual, is_formal, is_streetwear]
        """
        max_price = 5000.0
        style_flags = {
            "casual":      [1.0, 0.0, 0.0, 0.0, 0.0],
            "formal":      [0.0, 1.0, 0.0, 0.0, 0.0],
            "streetwear":  [0.0, 0.0, 1.0, 0.0, 0.0],
            "minimalist":  [0.0, 0.0, 0.0, 1.0, 0.0],
            "semi_formal": [0.0, 0.0, 0.0, 0.0, 1.0],
        }
        style_vec = style_flags.get(self.style, [0.0, 0.0, 0.0, 0.0, 0.0])
        return [
            self.price / max_price,
            self.formality / 5.0,
            self.comfort_score,
            self.versatility_score,
            self.popularity_score,
            self.trend_score,
        ] + style_vec

    def __repr__(self):
        return f"FashionItem({self.item_id}, {self.name[:30]}, ₹{self.price})"


class FashionCatalog:
    """
    Central fashion catalog. Supports filtering by category, occasion,
    season, budget, gender, style, and formality range.
    """

    def __init__(self):
        self._items: Dict[str, FashionItem] = {}
        self._by_category: Dict[str, List[FashionItem]] = {}
        self._loaded = False

    def load(self, path: str = CATALOG_JSON):
        """Load from JSON. Falls back to CSV."""
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        elif os.path.exists(CATALOG_CSV):
            raw = self._load_from_csv(CATALOG_CSV)
        else:
            raise FileNotFoundError(
                f"Catalog not found. Run `python data/generate_catalog.py` first."
            )

        for d in raw:
            item = self._parse(d)
            self._items[item.item_id] = item
            self._by_category.setdefault(item.category, []).append(item)

        self._loaded = True
        print(f"[OK] FashionCatalog loaded: {len(self._items)} items across "
              f"{len(self._by_category)} categories.")

    def _parse(self, d: dict) -> FashionItem:
        occ = d["occasion"]
        sea = d["season"]
        if isinstance(occ, str):
            occ = occ.split("|")
        if isinstance(sea, str):
            sea = sea.split("|")
        return FashionItem(
            item_id=d["item_id"],
            name=d["name"],
            category=d["category"],
            subcategory=d["subcategory"],
            color=d["color"],
            secondary_color=d.get("secondary_color", ""),
            style=d["style"],
            occasion=occ,
            season=sea,
            formality=int(d["formality"]),
            pattern=d["pattern"],
            material=d["material"],
            price=int(d["price"]),
            gender=d["gender"],
            comfort_score=float(d["comfort_score"]),
            versatility_score=float(d["versatility_score"]),
            popularity_score=float(d["popularity_score"]),
            trend_score=float(d["trend_score"]),
            image_placeholder=d.get("image_placeholder", ""),
        )

    def _load_from_csv(self, path: str) -> List[dict]:
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows.append(row)
        return rows

    # ── Query Methods ─────────────────────────────────────────────────────

    def get_by_id(self, item_id: str) -> Optional[FashionItem]:
        return self._items.get(item_id)

    def get_by_category(self, category: str) -> List[FashionItem]:
        return self._by_category.get(category, [])

    def filter(
        self,
        category: Optional[str] = None,
        occasion: Optional[str] = None,
        season: Optional[str] = None,
        max_budget: Optional[int] = None,
        gender: Optional[str] = None,
        style: Optional[str] = None,
        formality_min: int = 1,
        formality_max: int = 5,
        exclude_ids: Optional[Set[str]] = None,
    ) -> List[FashionItem]:
        items = list(self._items.values())

        if category:
            items = [i for i in items if i.category == category]
        if occasion:
            items = [i for i in items if i.matches_occasion(occasion)]
        if season:
            items = [i for i in items if i.matches_season(season)]
        if max_budget is not None:
            items = [i for i in items if i.price <= max_budget]
        if gender and gender != "unisex":
            items = [i for i in items if i.gender in (gender, "unisex")]
        if style:
            items = [i for i in items if i.style == style]
        items = [i for i in items if formality_min <= i.formality <= formality_max]
        if exclude_ids:
            items = [i for i in items if i.item_id not in exclude_ids]

        return items

    def all_items(self) -> List[FashionItem]:
        return list(self._items.values())

    def categories(self) -> List[str]:
        return list(self._by_category.keys())

    def size(self) -> int:
        return len(self._items)

    def summary(self) -> Dict[str, int]:
        return {cat: len(items) for cat, items in self._by_category.items()}


# Singleton instance
_catalog_instance: Optional[FashionCatalog] = None


def get_catalog() -> FashionCatalog:
    global _catalog_instance
    if _catalog_instance is None:
        _catalog_instance = FashionCatalog()
        _catalog_instance.load()
    return _catalog_instance
