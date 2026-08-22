"""
FashionVerse — Main FastAPI Application
Adaptive AI Fashion Stylist Backend using Reinforcement Learning, GenAI, and 3D/VR Try-On.
"""

import os
from dotenv import load_dotenv
load_dotenv()  # Load .env file at startup

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.database.database import init_db
from backend.fashion.catalog import get_catalog
from backend.rl.ppo_agent_inference import get_agent
from backend.api.chat import router as chat_router
from backend.api.recommend import router as recommend_router
from backend.api.feedback import router as feedback_router
from backend.api.avatar import router as avatar_router
from backend.api.analytics import router as analytics_router
from backend.api.user import router as user_router

app = FastAPI(
    title="FashionVerse API",
    description="Adaptive AI Fashion Stylist using Reinforcement Learning (PPO) with GenAI and 3D/VR Try-On",
    version="1.0.0",
)

# CORS middleware for local frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static plots directory
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "experiments", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)
app.mount("/plots", StaticFiles(directory=PLOTS_DIR), name="plots")

# Register routers
app.include_router(chat_router)
app.include_router(recommend_router)
app.include_router(feedback_router)
app.include_router(avatar_router)
app.include_router(analytics_router)
app.include_router(user_router)


@app.on_event("startup")
def startup_event():
    print("\n[FashionVerse] Starting application...")
    init_db()
    catalog = get_catalog()
    print(f"[FashionVerse] Loaded catalog: {catalog.size()} items.")
    agent = get_agent()
    print(f"[FashionVerse] RL Agent initialized. Model loaded: {agent.get_status()['model_loaded']}")


@app.get("/")
def root():
    return {
        "name": "FashionVerse API",
        "description": "Adaptive AI Fashion Stylist using Reinforcement Learning, GenAI and 3D/VR Try-On",
        "status": "online",
        "version": "1.0.0",
        "docs_url": "/docs",
    }


@app.get("/rl-status")
def rl_status():
    agent = get_agent()
    return agent.get_status()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
