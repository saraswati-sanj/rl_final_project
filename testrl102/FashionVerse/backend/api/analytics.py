"""
FashionVerse — Analytics & RL Dashboard API
Serves real-time RL metrics, experiment comparisons, training curves, and preference adaptation data.
"""

import os, json
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from typing import Dict, Any

from backend.database.database import get_db
from backend.database.repository import FashionRepository
from backend.rl.ppo_agent_inference import get_agent

router = APIRouter(prefix="/analytics", tags=["Analytics"])
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "experiments", "results")


@router.get("/summary")
def get_analytics_summary(db: Session = Depends(get_db)):
    db_stats = FashionRepository.get_analytics_summary(db)
    agent = get_agent()
    agent_status = agent.get_status()

    return {
        "db_stats": db_stats,
        "agent_status": agent_status,
        "algorithm": "PPO (Proximal Policy Optimization)",
    }


@router.get("/experiments")
def get_experiment_results():
    summary_path = os.path.join(RESULTS_DIR, "all_results_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # Fallback to individual files if combined doesn't exist
    results = {}
    for exp_name in ["exp1_baseline_comparison", "exp2_preference_drift", "exp3_reward_ablation", "exp4_exploration"]:
        p = os.path.join(RESULTS_DIR, f"{exp_name}.json")
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                results[exp_name] = json.load(f)
    return results


@router.get("/recent-interactions")
def get_recent_interactions(limit: int = 20, db: Session = Depends(get_db)):
    return FashionRepository.get_recent_interactions(db, limit=limit)
