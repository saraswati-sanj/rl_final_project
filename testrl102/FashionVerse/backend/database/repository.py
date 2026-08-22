"""
FashionVerse — Database Repository
Data access layer for users, interactions, recommendations, and analytics.
"""

import json
from datetime import datetime
from typing import Optional, List, Dict
from sqlalchemy.orm import Session
from backend.database.models import UserModel, InteractionModel, RecommendationModel, MetricModel
from backend.user.user_profile import UserProfile


class FashionRepository:

    @staticmethod
    def get_or_create_user(db: Session, user_id: str, budget: int = 2500, gender: str = "unisex") -> UserProfile:
        user_record = db.query(UserModel).filter(UserModel.user_id == user_id).first()
        if not user_record:
            profile = UserProfile(user_id=user_id, budget=budget, gender=gender)
            user_record = UserModel(
                user_id=user_id,
                budget=budget,
                gender=gender,
                profile_json=json.dumps(profile.to_dict()),
            )
            db.add(user_record)
            db.commit()
            db.refresh(user_record)
            return profile

        profile_data = json.loads(user_record.profile_json)
        return UserProfile.from_dict(profile_data)

    @staticmethod
    def save_user_profile(db: Session, profile: UserProfile):
        user_record = db.query(UserModel).filter(UserModel.user_id == profile.user_id).first()
        if user_record:
            user_record.budget = profile.budget
            user_record.gender = profile.gender
            user_record.profile_json = json.dumps(profile.to_dict())
            user_record.updated_at = datetime.utcnow()
            db.commit()

    @staticmethod
    def record_interaction(
        db: Session,
        user_id: str,
        outfit_id: str,
        action: str,
        feedback: Optional[str] = None,
        reward: float = 0.0,
        items: Optional[List[dict]] = None,
    ) -> InteractionModel:
        record = InteractionModel(
            user_id=user_id,
            outfit_id=outfit_id,
            action=action,
            feedback=feedback,
            reward=reward,
            items_json=json.dumps(items) if items else None,
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        return record

    @staticmethod
    def save_recommendation(
        db: Session,
        user_id: str,
        request_text: Optional[str],
        occasion: Optional[str],
        budget: Optional[int],
        outfit: List[dict],
        compatibility_score: float,
        rl_agent: str = "PPO",
        explanation: Optional[str] = None,
    ) -> RecommendationModel:
        record = RecommendationModel(
            user_id=user_id,
            request_text=request_text,
            occasion=occasion,
            budget=budget,
            outfit_json=json.dumps(outfit),
            compatibility_score=compatibility_score,
            rl_agent=rl_agent,
            explanation=explanation,
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        return record

    @staticmethod
    def get_recent_interactions(db: Session, limit: int = 50) -> List[Dict]:
        rows = db.query(InteractionModel).order_by(InteractionModel.timestamp.desc()).limit(limit).all()
        return [
            {
                "id": r.id,
                "user_id": r.user_id,
                "outfit_id": r.outfit_id,
                "action": r.action,
                "feedback": r.feedback,
                "reward": r.reward,
                "timestamp": r.timestamp.isoformat() if r.timestamp else None,
            }
            for r in rows
        ]

    @staticmethod
    def get_analytics_summary(db: Session) -> Dict:
        total_users = db.query(UserModel).count()
        total_interactions = db.query(InteractionModel).count()
        total_recommendations = db.query(RecommendationModel).count()

        interactions = db.query(InteractionModel).all()
        feedback_counts = {}
        total_reward = 0.0
        positives = 0

        for it in interactions:
            if it.feedback:
                feedback_counts[it.feedback] = feedback_counts.get(it.feedback, 0) + 1
                if it.feedback in ("love", "like", "save", "purchase"):
                    positives += 1
            total_reward += (it.reward or 0.0)

        acceptance_rate = (positives / len(interactions)) if interactions else 0.0
        avg_reward = (total_reward / len(interactions)) if interactions else 0.0

        return {
            "total_users": total_users,
            "total_interactions": total_interactions,
            "total_recommendations": total_recommendations,
            "acceptance_rate": round(acceptance_rate, 3),
            "average_reward": round(avg_reward, 3),
            "feedback_distribution": feedback_counts,
        }
