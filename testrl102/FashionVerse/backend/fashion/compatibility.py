"""
FashionVerse — Compatibility Engine
Scores outfit compatibility across color, style, occasion, and formality axes.
Used by the RL reward function and the outfit generator.
"""

from typing import List, Optional, Dict, Tuple
import math


# ── Color compatibility ───────────────────────────────────────────────────────
# Groups: neutral, cool, warm, earth, bright
# Within-group and neutral combinations score higher.

COLOR_GROUPS: Dict[str, str] = {
    "black":       "neutral",
    "white":       "neutral",
    "grey":        "neutral",
    "beige":       "neutral",
    "cream":       "neutral",
    "charcoal":    "neutral",
    "tan":         "neutral",
    "navy":        "cool",
    "royal_blue":  "cool",
    "teal":        "cool",
    "sky_blue":    "cool",
    "slate":       "cool",
    "indigo":      "cool",
    "lavender":    "cool",
    "forest_green":"cool",
    "sage_green":  "cool",
    "maroon":      "warm",
    "wine":        "warm",
    "rust":        "warm",
    "burnt_orange":"warm",
    "coral":       "warm",
    "dusty_rose":  "warm",
    "blush":       "warm",
    "olive":       "earth",
    "mustard":     "earth",
    "mustard":     "earth",
}

COMPLEMENTARY: Dict[str, List[str]] = {
    "neutral": ["neutral", "cool", "warm", "earth", "bright"],
    "cool":    ["neutral", "cool", "earth"],
    "warm":    ["neutral", "warm", "earth"],
    "earth":   ["neutral", "cool", "warm", "earth"],
    "bright":  ["neutral"],
}


def color_compatibility(c1: str, c2: str) -> float:
    """Returns a compatibility score in [0, 1]."""
    if c1 == c2:
        return 1.0 if c1 in ("black", "white", "grey", "navy") else 0.7
    g1 = COLOR_GROUPS.get(c1, "bright")
    g2 = COLOR_GROUPS.get(c2, "bright")
    if g2 in COMPLEMENTARY.get(g1, []):
        return 0.85
    return 0.40


# ── Style compatibility ───────────────────────────────────────────────────────

STYLE_COMPAT: Dict[str, List[str]] = {
    "casual":      ["casual", "streetwear", "minimalist", "athleisure", "bohemian"],
    "streetwear":  ["casual", "streetwear", "athleisure"],
    "formal":      ["formal", "semi_formal", "minimalist"],
    "semi_formal": ["formal", "semi_formal", "minimalist", "preppy", "indo_western"],
    "minimalist":  ["minimalist", "casual", "formal", "semi_formal"],
    "bohemian":    ["bohemian", "casual", "indo_western", "festive"],
    "athleisure":  ["athleisure", "casual", "streetwear"],
    "indo_western":["indo_western", "festive", "semi_formal", "bohemian"],
    "festive":     ["festive", "indo_western", "bohemian"],
    "preppy":      ["preppy", "formal", "semi_formal", "minimalist"],
}


def style_compatibility(s1: str, s2: str) -> float:
    if s1 == s2:
        return 1.0
    if s2 in STYLE_COMPAT.get(s1, []):
        return 0.80
    return 0.30


# ── Formality compatibility ───────────────────────────────────────────────────

def formality_compatibility(f1: int, f2: int) -> float:
    """Scores how well two formality levels (1–5) match."""
    diff = abs(f1 - f2)
    return max(0.0, 1.0 - diff * 0.25)


# ── Occasion match ────────────────────────────────────────────────────────────

def occasion_match(item_occasions: List[str], target_occasion: str) -> float:
    """Returns 1.0 if target occasion is in item's occasion list, else 0."""
    return 1.0 if target_occasion.lower() in [o.lower() for o in item_occasions] else 0.0


# ── Outfit scorer ─────────────────────────────────────────────────────────────

class OutfitCompatibilityScorer:
    """
    Scores a partial or complete outfit.
    Used by:
      - RL reward function (shaping signal)
      - Rule-based recommender
      - Frontend display
    """

    def score_pair(self, item1, item2) -> float:
        """Score compatibility between two FashionItems."""
        c = color_compatibility(item1.color, item2.color)
        s = style_compatibility(item1.style, item2.style)
        f = formality_compatibility(item1.formality, item2.formality)
        return round((0.35 * c + 0.40 * s + 0.25 * f), 3)

    def score_outfit(self, items: List, occasion: Optional[str] = None) -> Dict:
        """
        Score a list of items as a complete outfit.
        Returns per-axis scores and overall score.
        """
        if not items:
            return {"overall": 0.0, "color": 0.0, "style": 0.0,
                    "formality": 0.0, "occasion": 0.0}

        n = len(items)
        if n == 1:
            occ_score = occasion_match(items[0].occasion, occasion) if occasion else 0.5
            return {"overall": 0.5 + 0.3 * occ_score,
                    "color": 1.0, "style": 1.0, "formality": 1.0,
                    "occasion": occ_score}

        # Pairwise averages
        pair_scores = {"color": [], "style": [], "formality": []}
        for i in range(n):
            for j in range(i + 1, n):
                a, b = items[i], items[j]
                pair_scores["color"].append(color_compatibility(a.color, b.color))
                pair_scores["style"].append(style_compatibility(a.style, b.style))
                pair_scores["formality"].append(
                    formality_compatibility(a.formality, b.formality)
                )

        avg_c = sum(pair_scores["color"]) / len(pair_scores["color"])
        avg_s = sum(pair_scores["style"]) / len(pair_scores["style"])
        avg_f = sum(pair_scores["formality"]) / len(pair_scores["formality"])

        # Occasion match: average over all items
        if occasion:
            occ_scores = [occasion_match(it.occasion, occasion) for it in items]
            avg_occ = sum(occ_scores) / len(occ_scores)
        else:
            avg_occ = 0.5

        overall = 0.30 * avg_c + 0.35 * avg_s + 0.20 * avg_f + 0.15 * avg_occ

        return {
            "overall": round(overall, 3),
            "color": round(avg_c, 3),
            "style": round(avg_s, 3),
            "formality": round(avg_f, 3),
            "occasion": round(avg_occ, 3),
        }

    def is_complete_outfit(self, items: List, allow_dress: bool = True) -> bool:
        """
        A complete outfit requires:
          - (top + bottom) OR dress
          - shoes
        """
        cats = {i.category for i in items}
        has_shoes = "shoes" in cats
        has_coverage = ("top" in cats and "bottom" in cats) or (
            allow_dress and "dress" in cats
        )
        return has_shoes and has_coverage


# ── Module-level convenience ──────────────────────────────────────────────────
_scorer = OutfitCompatibilityScorer()


def score_outfit(items: List, occasion: Optional[str] = None) -> Dict:
    return _scorer.score_outfit(items, occasion)


def score_pair(item1, item2) -> float:
    return _scorer.score_pair(item1, item2)


def is_complete(items: List) -> bool:
    return _scorer.is_complete_outfit(items)
