"""
FashionVerse — RL Agent Status & Inference Module
Provides a unified interface used by the FastAPI backend.
Loads the trained PPO model and exposes predict/recommend methods.
"""

import os, json
import numpy as np
from typing import Optional, Dict, List

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")


class RLAgent:
    """
    Production RL agent wrapper used by the FastAPI backend.
    Supports both trained PPO model and rule-based fallback.
    """

    def __init__(self):
        self._ppo = None
        self._is_loaded = False
        self._episode_count = 0
        self._total_reward = 0.0
        self._acceptance_count = 0
        self._interaction_count = 0
        self._algorithm = "PPO"
        self._training_data: Optional[Dict] = None

    def load_model(self, model_path: Optional[str] = None) -> bool:
        """Load trained PPO model. Returns True if successful."""
        if model_path is None:
            model_path = os.path.join(MODEL_DIR, "ppo_exp1.zip")

        if not os.path.exists(model_path):
            print(f"[RLAgent] No model found at {model_path}. Using fallback.")
            return False

        try:
            from backend.rl.ppo_agent import FashionPPOAgent
            self._ppo = FashionPPOAgent()
            self._ppo.load(model_path)
            self._is_loaded = True
            print(f"[RLAgent] Loaded PPO model from {model_path}")
            return True
        except Exception as e:
            print(f"[RLAgent] Load failed: {e}")
            return False

    def recommend(
        self,
        user_profile,
        constraints,
        n_candidates: int = 10,
    ) -> Dict:
        """
        Generate an outfit recommendation using the RL agent.
        Falls back to rule-based if model not loaded.
        """
        if self._is_loaded and self._ppo:
            return self._recommend_ppo(user_profile, constraints, n_candidates)
        return self._recommend_fallback(constraints)

    def _recommend_ppo(self, user_profile, constraints, n_candidates) -> Dict:
        from backend.rl.fashion_env import FashionEnv
        from backend.rl.state import FashionState
        from backend.rl.actions import ActionSpace
        from backend.fashion.catalog import get_catalog

        catalog = get_catalog()
        asp = ActionSpace(catalog, n_candidates)
        outfit = []
        budget_remaining = constraints.budget

        # Build initial state
        state = FashionState(
            user_profile_vec=user_profile.to_vector(),
            occasion=constraints.occasion,
            season=constraints.season,
            budget_total=constraints.budget,
            budget_remaining=budget_remaining,
            formality_target=float(constraints.formality_min + constraints.formality_max) / 2,
        )

        for step in range(6):
            candidates = asp.build_candidates(constraints, outfit)
            obs = state.encode()
            action = self._ppo.predict(obs, deterministic=True)
            decoded = asp.decode(action)

            if decoded.is_finish or not decoded.is_valid:
                break
            if decoded.item:
                v = constraints.validate_item(decoded.item, outfit)
                if v["valid"]:
                    outfit.append(decoded.item)
                    state.selected_items = outfit
                    state.budget_remaining -= decoded.item.price

        self._episode_count += 1
        self._interaction_count += 1

        from backend.fashion.compatibility import score_outfit, is_complete
        score = score_outfit(outfit, constraints.occasion) if outfit else {"overall": 0}

        return {
            "outfit": [{"item_id": i.item_id, "name": i.name, "category": i.category,
                        "price": i.price, "color": i.color, "style": i.style,
                        "image": i.image_placeholder} for i in outfit],
            "compatibility_score": score.get("overall", 0),
            "is_complete": is_complete(outfit),
            "total_price": sum(i.price for i in outfit),
            "agent": "PPO",
            "n_items": len(outfit),
        }

    def _recommend_fallback(self, constraints) -> Dict:
        from backend.fashion.outfit_generator import OutfitGenerator
        from backend.fashion.compatibility import score_outfit, is_complete
        gen = OutfitGenerator()
        outfit = gen.generate_rule_based(constraints)
        score = score_outfit(outfit, constraints.occasion) if outfit else {"overall": 0}
        return {
            "outfit": [{"item_id": i.item_id, "name": i.name, "category": i.category,
                        "price": i.price, "color": i.color, "style": i.style,
                        "image": i.image_placeholder} for i in outfit],
            "compatibility_score": score.get("overall", 0),
            "is_complete": is_complete(outfit),
            "total_price": sum(i.price for i in outfit),
            "agent": "RuleBased (fallback)",
            "n_items": len(outfit),
        }

    def record_feedback(self, feedback: str, reward: float):
        self._interaction_count += 1
        self._total_reward += reward
        if feedback in ("love", "like", "save", "purchase"):
            self._acceptance_count += 1

    def get_status(self) -> Dict:
        return {
            "algorithm": self._algorithm,
            "model_loaded": self._is_loaded,
            "total_interactions": self._interaction_count,
            "mean_reward": (self._total_reward / max(self._interaction_count, 1)),
            "acceptance_rate": (self._acceptance_count / max(self._interaction_count, 1)),
        }

    def load_training_results(self) -> Optional[Dict]:
        path = os.path.join(
            os.path.dirname(__file__), "..", "..", "experiments",
            "results", "exp1_baseline_comparison.json"
        )
        if os.path.exists(path):
            with open(path) as f:
                self._training_data = json.load(f)
        return self._training_data


# Singleton
_agent_instance: Optional[RLAgent] = None

def get_agent() -> RLAgent:
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = RLAgent()
        _agent_instance.load_model()
    return _agent_instance
