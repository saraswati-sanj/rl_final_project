"""
FashionVerse — Avatar & 3D Try-On API
Provides avatar configurations, body parameters, and 3D clothing mappings for Three.js/WebXR.
"""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional, Dict, Any, List

router = APIRouter(prefix="/avatar", tags=["Avatar"])

# Standard color mappings for 3D procedural clothing materials in Three.js
COLOR_HEX_MAP = {
    "black": "#1A1A1A",
    "white": "#F8F9FA",
    "navy": "#1A2E40",
    "grey": "#6C757D",
    "beige": "#D2B48C",
    "olive": "#556B2F",
    "maroon": "#800000",
    "royal_blue": "#206095",
    "forest_green": "#228B22",
    "burnt_orange": "#CC5500",
    "dusty_rose": "#DCAE96",
    "lavender": "#B57EDC",
    "teal": "#008080",
    "mustard": "#E1AD01",
    "cream": "#FFFDD0",
    "charcoal": "#36454F",
    "rust": "#B7410E",
    "sage_green": "#9DC183",
    "wine": "#722F37",
    "sky_blue": "#87CEEB",
    "coral": "#FF7F50",
    "tan": "#D2B48C",
    "blush": "#DE5D83",
    "slate": "#708090",
    "indigo": "#4B0082",
}


class AvatarConfig(BaseModel):
    user_id: str = "user_default"
    gender: str = "female"
    skin_tone: str = "#E0AC69"
    height: float = 1.70
    hair_style: str = "short"
    hair_color: str = "#2C221E"


@router.get("/config", response_model=AvatarConfig)
def get_avatar_config(user_id: str = "user_default"):
    return AvatarConfig(user_id=user_id)


@router.get("/materials")
def get_avatar_materials():
    return {
        "color_map": COLOR_HEX_MAP,
        "skin_tones": ["#8D5524", "#C68642", "#E0AC69", "#F1C27D", "#FFDBAC"],
        "clothing_slots": ["top", "bottom", "dress", "shoes", "accessory"],
    }
