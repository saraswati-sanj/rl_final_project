"""
FashionVerse — GenAI Intent Parser
Extracts structured fashion constraints from natural language requests.
Operates in 2 modes:
  1. LLM Mode (OpenAI / local LLM if configured via environment)
  2. Rule-Based Fallback Mode (heuristic NLP regex & keyword extraction)

CRITICAL PRINCIPLE:
  GenAI ONLY extracts constraints and context.
  The RL agent chooses the outfit items.
"""

import os, re, json
from typing import Optional, Dict
from backend.fashion.constraints import FashionConstraints


class IntentParser:

    def __init__(self):
        self.api_key = os.getenv("LLM_API_KEY", "").strip()
        self.model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.base_url = os.getenv("LLM_BASE_URL", "").strip() or None

    def parse(self, text: str, user_budget: Optional[int] = None) -> FashionConstraints:
        """
        Parses natural language prompt into a structured FashionConstraints object.
        """
        if not text or not text.strip():
            return FashionConstraints(budget=user_budget or 2500)

        if self.api_key:
            try:
                return self._parse_llm(text, user_budget)
            except Exception as e:
                print(f"[IntentParser] LLM parsing failed: {e}. Using rule-based fallback.")

        return self._parse_rules(text, user_budget)

    def _parse_llm(self, text: str, user_budget: Optional[int]) -> FashionConstraints:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        system_prompt = """You are a fashion request parser. Extract structured constraints from user fashion requests into JSON with exact keys:
- occasion: one of ["casual", "college", "office", "semi_formal", "formal", "party", "festive", "date", "travel", "gym", "beach"]
- season: one of ["summer", "winter", "monsoon", "all"]
- budget: integer in INR (e.g. 2500)
- formality_min: integer 1-5
- formality_max: integer 1-5
- style_preference: one of ["casual", "streetwear", "formal", "semi_formal", "minimalist", "bohemian", "athleisure", "indo_western", "festive", "preppy"] or null
- color_preference: color string or null
- gender: one of ["male", "female", "unisex"]

Return ONLY raw JSON."""

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Request: {text}\nDefault Budget: {user_budget or 2500}"},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )

        data = json.loads(response.choices[0].message.content)
        return FashionConstraints(
            budget=int(data.get("budget") or user_budget or 2500),
            occasion=data.get("occasion") or "casual",
            season=data.get("season") or "all",
            gender=data.get("gender") or "unisex",
            formality_min=int(data.get("formality_min") or 1),
            formality_max=int(data.get("formality_max") or 5),
            style_preference=data.get("style_preference"),
            color_preference=data.get("color_preference"),
        )

    def _parse_rules(self, text: str, user_budget: Optional[int]) -> FashionConstraints:
        lower = text.lower()

        # 1. Budget extraction
        budget = user_budget or 2500
        budget_match = re.search(r'(?:under|below|less than|within|budget|rs\.?|inr|₹)\s*[:=]?\s*(\d{3,5})', lower)
        if not budget_match:
            budget_match = re.search(r'(\d{3,5})\s*(?:rs|rupees|inr|bucks)?', lower)
        if budget_match:
            try:
                budget = int(budget_match.group(1))
            except ValueError:
                pass

        # 2. Occasion extraction
        occasion = "casual"
        if any(w in lower for w in ["college", "presentation", "campus", "class", "university"]):
            occasion = "college"
        elif any(w in lower for w in ["office", "work", "meeting", "interview", "corporate"]):
            occasion = "office"
        elif any(w in lower for w in ["formal", "business formal", "black tie"]):
            occasion = "formal"
        elif any(w in lower for w in ["semi formal", "semi-formal", "smart casual", "presentation"]):
            occasion = "semi_formal"
        elif any(w in lower for w in ["party", "club", "night out", "celebration"]):
            occasion = "party"
        elif any(w in lower for w in ["festive", "diwali", "wedding", "pooja", "traditional"]):
            occasion = "festive"
        elif any(w in lower for w in ["date", "dinner", "romantic"]):
            occasion = "date"
        elif any(w in lower for w in ["travel", "vacation", "trip", "flight", "airport"]):
            occasion = "travel"
        elif any(w in lower for w in ["gym", "workout", "running", "sports"]):
            occasion = "gym"
        elif any(w in lower for w in ["beach", "pool", "resort"]):
            occasion = "beach"

        # 3. Season extraction
        season = "all"
        if any(w in lower for w in ["summer", "hot", "sunny", "humid"]):
            season = "summer"
        elif any(w in lower for w in ["winter", "cold", "jacket", "sweater", "warm"]):
            season = "winter"
        elif any(w in lower for w in ["monsoon", "rainy", "rain"]):
            season = "monsoon"

        # 4. Formality range
        if occasion in ("formal", "office"):
            formality_min, formality_max = 3, 5
        elif occasion in ("semi_formal", "college"):
            formality_min, formality_max = 2, 4
        elif occasion in ("party", "festive"):
            formality_min, formality_max = 2, 5
        else:
            formality_min, formality_max = 1, 3

        if "strictly formal" in lower or "very formal" in lower:
            formality_min, formality_max = 4, 5
        elif "very casual" in lower or "super chill" in lower or "loungewear" in lower:
            formality_min, formality_max = 1, 2

        # 5. Style preference
        style_preference = None
        for s in ["streetwear", "minimalist", "bohemian", "athleisure", "indo_western", "preppy", "casual", "formal", "semi_formal"]:
            if s in lower or s.replace("_", " ") in lower or s.replace("_", "-") in lower:
                style_preference = s
                break

        # 6. Color preference
        colors = ["black", "white", "navy", "grey", "beige", "olive", "maroon", "royal_blue", "teal", "mustard", "charcoal", "rust", "wine", "coral"]
        color_preference = None
        for c in colors:
            if c in lower or c.replace("_", " ") in lower:
                color_preference = c
                break

        # 7. Gender hint
        gender = "unisex"
        if any(w in lower for w in ["women", "female", "girl", "ladies", "she", "her", "dress", "kurti", "skirt"]):
            gender = "female"
        elif any(w in lower for w in ["men", "male", "guy", "gents", "he", "his", "kurta"]):
            gender = "male"

        return FashionConstraints(
            budget=budget,
            occasion=occasion,
            season=season,
            gender=gender,
            formality_min=formality_min,
            formality_max=formality_max,
            style_preference=style_preference,
            color_preference=color_preference,
        )


_parser_instance: Optional[IntentParser] = None

def get_intent_parser() -> IntentParser:
    global _parser_instance
    if _parser_instance is None:
        _parser_instance = IntentParser()
    return _parser_instance
