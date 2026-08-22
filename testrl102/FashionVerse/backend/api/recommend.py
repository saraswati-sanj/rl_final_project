"""
FashionVerse — Recommend API Endpoint
Direct endpoint to get outfit recommendations for given structured or semi-structured constraints.
"""

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from sqlalchemy.orm import Session

from backend.database.database import get_db
from backend.database.repository import FashionRepository
from backend.fashion.constraints import FashionConstraints
from backend.rl.ppo_agent_inference import get_agent
from backend.genai.explanation import get_explanation_engine

router = APIRouter(prefix="/recommend", tags=["Recommend"])


class RecommendRequest(BaseModel):
    user_id: str = "user_default"
    occasion: Optional[str] = "casual"
    season: Optional[str] = "all"
    budget: Optional[int] = 2500
    gender: Optional[str] = "unisex"
    style: Optional[str] = None
    formality_min: Optional[int] = 1
    formality_max: Optional[int] = 5


class RecommendResponse(BaseModel):
    user_id: str
    outfit: List[Dict[str, Any]]
    compatibility_score: float
    total_price: int
    is_complete: bool
    agent: str
    explanation: str
    constraints: Dict[str, Any]


@router.post("", response_model=RecommendResponse)
def handle_recommend(payload: RecommendRequest, db: Session = Depends(get_db)):
    profile = FashionRepository.get_or_create_user(
        db, user_id=payload.user_id, budget=payload.budget or 2500, gender=payload.gender or "unisex"
    )

    constraints = FashionConstraints(
        budget=payload.budget or 2500,
        occasion=payload.occasion or "casual",
        season=payload.season or "all",
        gender=payload.gender or "unisex",
        style_preference=payload.style,
        formality_min=payload.formality_min or 1,
        formality_max=payload.formality_max or 5,
    )

    agent = get_agent()
    rec = agent.recommend(profile, constraints)

    top_style = "casual"
    if profile.style_estimates:
        top_style = max(profile.style_estimates.items(), key=lambda x: x[1])[0]

    explainer = get_explanation_engine()
    explanation = explainer.explain(
        outfit_items=rec["outfit"],
        constraints=constraints.to_dict(),
        compatibility_score=rec["compatibility_score"],
        user_profile_summary={"top_preferred_style": top_style},
        rl_agent_name=rec.get("agent", "PPO"),
    )

    FashionRepository.save_recommendation(
        db=db,
        user_id=payload.user_id,
        request_text=f"{payload.occasion} outfit under ₹{payload.budget}",
        occasion=payload.occasion,
        budget=payload.budget,
        outfit=rec["outfit"],
        compatibility_score=rec["compatibility_score"],
        rl_agent=rec.get("agent", "PPO"),
        explanation=explanation,
    )

    return RecommendResponse(
        user_id=payload.user_id,
        outfit=rec["outfit"],
        compatibility_score=rec["compatibility_score"],
        total_price=rec["total_price"],
        is_complete=rec["is_complete"],
        agent=rec.get("agent", "PPO"),
        explanation=explanation,
        constraints=constraints.to_dict(),
    )
