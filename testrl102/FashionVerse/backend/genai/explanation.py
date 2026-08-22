"""
FashionVerse — GenAI Explanation Engine
Generates transparent, grounded explanations for why the RL agent selected this specific outfit.
Passes exact decision features (compatibility score, color harmony, occasion suitability, budget utilization)
into the explanation template or LLM prompt.

CRITICAL PRINCIPLE:
  Explanations must be derived directly from actual RL decision data.
  The LLM is NEVER allowed to hallucinate reasons unrelated to the RL features.
"""

import os
from typing import List, Dict, Optional


class ExplanationEngine:

    def __init__(self):
        self.api_key = os.getenv("LLM_API_KEY", "").strip()
        self.model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.base_url = os.getenv("LLM_BASE_URL", "").strip() or None

    def explain(
        self,
        outfit_items: List[Dict],
        constraints: Dict,
        compatibility_score: float,
        user_profile_summary: Dict,
        rl_agent_name: str = "PPO",
    ) -> str:
        """
        Generates an explanation of the RL decision.
        """
        if not outfit_items:
            return "No complete outfit could be constructed within the given constraints."

        total_price = sum(i.get("price", 0) for i in outfit_items)
        occasion = constraints.get("occasion", "casual")
        budget = constraints.get("budget", 2500)
        colors = list({i.get("color", "") for i in outfit_items if i.get("color")})
        styles = list({i.get("style", "") for i in outfit_items if i.get("style")})
        top_style = user_profile_summary.get("top_preferred_style", "casual")

        features = {
            "total_price": total_price,
            "budget": budget,
            "budget_saved": budget - total_price,
            "occasion": occasion,
            "compatibility_pct": int(compatibility_score * 100),
            "colors": ", ".join(colors),
            "styles": ", ".join(styles),
            "user_top_style": top_style,
            "agent": rl_agent_name,
        }

        if self.api_key:
            try:
                return self._explain_llm(outfit_items, features)
            except Exception as e:
                print(f"[ExplanationEngine] LLM explanation failed: {e}. Falling back to template.")

        return self._explain_template(outfit_items, features)

    def _explain_llm(self, outfit_items: List[Dict], features: Dict) -> str:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        item_descriptions = [f"- {i.get('category', '').title()}: {i.get('name')} (₹{i.get('price')}, {i.get('color')}, {i.get('style')})" for i in outfit_items]
        items_str = "\n".join(item_descriptions)

        prompt = f"""You are the FashionVerse AI Stylist. Explain in 2-3 concise, friendly sentences why the {features['agent']} RL agent selected this specific outfit for the user.

FACTUAL DATA:
{items_str}
- Occasion Target: {features['occasion']}
- Overall Compatibility Score: {features['compatibility_pct']}%
- Budget: ₹{features['budget']} (Total Outfit Price: ₹{features['total_price']}, Saving: ₹{features['budget_saved']})
- Color Palette: {features['colors']}
- User Estimated Style Preference: {features['user_top_style']}

RULES:
- Explain based strictly on the factual data above (color harmony, occasion appropriateness, budget fit, and user feedback history).
- Do not invent fictional brands or details."""

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You provide clear, accurate explanations for AI fashion decisions."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
            max_tokens=150,
        )

        return response.choices[0].message.content.strip()

    def _explain_template(self, outfit_items: List[Dict], features: Dict) -> str:
        savings = features["budget_saved"]
        savings_text = f"saving ₹{savings}" if savings > 0 else "matching your exact budget"
        return (
            f"The {features['agent']} stylist selected this combination with a {features['compatibility_pct']}% compatibility score. "
            f"The harmonious {features['colors']} palette is tailored for a {features['occasion'].replace('_', ' ')} setting, "
            f"staying well within budget at ₹{features['total_price']} ({savings_text}) while adapting to your preference for {features['user_top_style']} aesthetics."
        )


_explanation_instance: Optional[ExplanationEngine] = None

def get_explanation_engine() -> ExplanationEngine:
    global _explanation_instance
    if _explanation_instance is None:
        _explanation_instance = ExplanationEngine()
    return _explanation_instance
