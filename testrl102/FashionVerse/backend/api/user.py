"""
FashionVerse — User API Endpoint
Retrieves and updates observable user preference state.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Dict, Any

from backend.database.database import get_db
from backend.database.repository import FashionRepository

router = APIRouter(prefix="/user", tags=["User"])


@router.get("/{user_id}")
def get_user_profile(user_id: str, db: Session = Depends(get_db)):
    profile = FashionRepository.get_or_create_user(db, user_id=user_id)
    return {
        "user_id": profile.user_id,
        "budget": profile.budget,
        "gender": profile.gender,
        "style_estimates": profile.style_estimates,
        "color_estimates": profile.color_estimates,
        "formality_estimate": profile.formality_estimate,
        "total_interactions": profile.total_interactions,
        "likes": profile.likes,
        "dislikes": profile.dislikes,
        "saves": profile.saves,
        "purchases": profile.purchases,
        "skips": profile.skips,
        "acceptance_rate": profile.acceptance_rate(),
    }
