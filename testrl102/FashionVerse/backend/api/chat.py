"""
FashionVerse — Chat API Endpoint
Handles natural-language conversations, parses intent, invokes RL recommender, and returns structured outfit + explanation.
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from sqlalchemy.orm import Session

from backend.database.database import get_db
from backend.database.repository import FashionRepository
from backend.genai.stylist import get_stylist_service

router = APIRouter(prefix="/chat", tags=["Chat"])


class ChatRequest(BaseModel):
    user_id: str = Field(default="user_default")
    message: str = Field(..., example="I need a semi-formal outfit for a college presentation under ₹2500.")
    budget: Optional[int] = None
    gender: Optional[str] = "unisex"


class ChatResponse(BaseModel):
    user_id: str
    reply: str
    constraints: Dict[str, Any]
    outfit: Dict[str, Any]
    explanation: str
    user_profile: Dict[str, Any]


@router.post("", response_model=ChatResponse)
def handle_chat(payload: ChatRequest, db: Session = Depends(get_db)):
    # 1. Fetch or create user profile
    profile = FashionRepository.get_or_create_user(
        db, user_id=payload.user_id, budget=payload.budget or 2500, gender=payload.gender or "unisex"
    )

    # 2. Process request through Stylist pipeline
    stylist = get_stylist_service()
    result = stylist.process_request(profile, payload.message)

    # 3. Save recommendation to database
    FashionRepository.save_recommendation(
        db=db,
        user_id=payload.user_id,
        request_text=payload.message,
        occasion=result["constraints"].get("occasion"),
        budget=result["constraints"].get("budget"),
        outfit=result["recommendation"].get("outfit", []),
        compatibility_score=result["recommendation"].get("compatibility_score", 0.0),
        rl_agent=result["recommendation"].get("agent", "PPO"),
        explanation=result["explanation"],
    )

    # 4. Save updated profile state
    FashionRepository.save_user_profile(db, profile)

    reply_msg = f"Here is a personalized look for your {result['constraints'].get('occasion', 'event').replace('_', ' ')}: {result['explanation']}"

    return ChatResponse(
        user_id=payload.user_id,
        reply=reply_msg,
        constraints=result["constraints"],
        outfit=result["recommendation"],
        explanation=result["explanation"],
        user_profile=profile.to_dict(),
    )
