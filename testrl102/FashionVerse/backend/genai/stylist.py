"""
FashionVerse — AI Stylist Service
Coordinates GenAI Intent Parser -> RL Recommendation -> GenAI Explanation.
"""

from typing import Dict, Optional, List
from backend.genai.intent_parser import get_intent_parser
from backend.genai.explanation import get_explanation_engine
from backend.rl.ppo_agent_inference import get_agent
from backend.user.user_profile import UserProfile
from backend.fashion.catalog import get_catalog


class StylistService:

    def __init__(self):
        self.parser = get_intent_parser()
        self.explainer = get_explanation_engine()
        self.rl_agent = get_agent()
        self.catalog = get_catalog()

    def process_request(
        self,
        user_profile: UserProfile,
        request_text: str,
    ) -> Dict:
        # Step 1: GenAI extracts structured constraints
        constraints = self.parser.parse(request_text, user_budget=user_profile.budget)

        # Step 2: RL agent constructs the outfit
        rec = self.rl_agent.recommend(user_profile, constraints)

        # Step 3: Extract top preferred style from observable profile
        top_style = "casual"
        if user_profile.style_estimates:
            top_style = max(user_profile.style_estimates.items(), key=lambda x: x[1])[0]

        profile_summary = {
            "top_preferred_style": top_style,
            "total_interactions": user_profile.total_interactions,
            "acceptance_rate": user_profile.acceptance_rate(),
        }

        # Step 4: GenAI explains why the RL agent chose this outfit
        explanation = self.explainer.explain(
            outfit_items=rec["outfit"],
            constraints=constraints.to_dict(),
            compatibility_score=rec["compatibility_score"],
            user_profile_summary=profile_summary,
            rl_agent_name=rec.get("agent", "PPO"),
        )

        return {
            "request_text": request_text,
            "constraints": constraints.to_dict(),
            "recommendation": rec,
            "explanation": explanation,
            "user_profile": user_profile.to_dict(),
        }


_stylist_instance: Optional[StylistService] = None

def get_stylist_service() -> StylistService:
    global _stylist_instance
    if _stylist_instance is None:
        _stylist_instance = StylistService()
    return _stylist_instance
