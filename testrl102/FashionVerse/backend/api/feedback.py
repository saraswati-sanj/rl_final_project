"""
FashionVerse — Feedback API Endpoint
Receives explicit and implicit user feedback, computes RL reward, updates user profile state,
and records interaction in the database.
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session

from backend.database.database import get_db
from backend.database.repository import FashionRepository
from backend.user.preference_update import update_preference
from backend.rl.reward import RewardCalculator
from backend.rl.ppo_agent_inference import get_agent
from backend.fashion.catalog import get_catalog

router = APIRouter(prefix="/feedback", tags=["Feedback"])


class FeedbackRequest(BaseModel):
    user_id: str = "user_default"
    outfit_id: str = Field(default="outfit_current")
    feedback: str = Field(..., example="love")  # love, like, neutral, dislike, skip, save, purchase
    action_type: Optional[str] = "explicit_feedback"  # explicit_feedback, try_on, swap_item, remove_item
    item_ids: Optional[List[str]] = Field(default_factory=list)
    view_duration_seconds: Optional[float] = 0.0
    occasion: Optional[str] = "casual"


class FeedbackResponse(BaseModel):
    status: str
    user_id: str
    feedback: str
    computed_reward: float
    updated_profile: Dict[str, Any]
    acceptance_rate: float


@router.post("", response_model=FeedbackResponse)
def handle_feedback(payload: FeedbackRequest, db: Session = Depends(get_db)):
    profile = FashionRepository.get_or_create_user(db, user_id=payload.user_id)
    catalog = get_catalog()

    # Resolve items from item_ids
    items = []
    for i_id in payload.item_ids:
        item_obj = catalog.get_by_id(i_id)
        if item_obj:
            items.append(item_obj)

    # Calculate RL reward
    reward_calc = RewardCalculator()
    # If no items provided, compute baseline feedback reward
    if items:
        from backend.fashion.constraints import FashionConstraints
        constraints = FashionConstraints(budget=profile.budget, occasion=payload.occasion or "casual")
        reward, _ = reward_calc.terminal_reward(
            feedback=payload.feedback,
            items=items,
            constraints=constraints,
            recent_item_ids=profile.recently_recommended,
            occasion=payload.occasion or "casual",
        )
        # Update user profile via EMA
        update_preference(profile, items, payload.feedback)
    else:
        reward = reward_calc.config.feedback_to_reward(payload.feedback)

    # Adjust reward slightly for implicit signals if applicable
    if payload.action_type == "remove_item":
        reward -= 2.0
    elif payload.action_type == "try_on":
        reward += 1.0

    # Save to database
    FashionRepository.record_interaction(
        db=db,
        user_id=payload.user_id,
        outfit_id=payload.outfit_id,
        action=payload.action_type or "feedback",
        feedback=payload.feedback,
        reward=reward,
        items=[{"item_id": i.item_id, "name": i.name, "category": i.category} for i in items],
    )

    FashionRepository.save_user_profile(db, profile)

    # Update online agent tracking
    agent = get_agent()
    agent.record_feedback(payload.feedback, reward)

    return FeedbackResponse(
        status="success",
        user_id=payload.user_id,
        feedback=payload.feedback,
        computed_reward=round(reward, 3),
        updated_profile=profile.to_dict(),
        acceptance_rate=round(profile.acceptance_rate(), 3),
    )
