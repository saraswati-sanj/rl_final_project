"""
FashionVerse — Database Models
SQLAlchemy ORM models storing users, interactions, recommendations, and metrics.
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean
from backend.database.database import Base


class UserModel(Base):
    __tablename__ = "users"

    user_id = Column(String(64), primary_key=True, index=True)
    budget = Column(Integer, default=2500)
    gender = Column(String(32), default="unisex")
    profile_json = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class InteractionModel(Base):
    __tablename__ = "interactions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(64), index=True, nullable=False)
    outfit_id = Column(String(128), nullable=False)
    action = Column(String(64), nullable=False)  # try_on, swap_item, remove_item, like, dislike, save, purchase
    feedback = Column(String(32), nullable=True)  # love, like, neutral, dislike, skip, save, purchase
    reward = Column(Float, default=0.0)
    items_json = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)


class RecommendationModel(Base):
    __tablename__ = "recommendations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(64), index=True, nullable=False)
    request_text = Column(Text, nullable=True)
    occasion = Column(String(64), nullable=True)
    budget = Column(Integer, nullable=True)
    outfit_json = Column(Text, nullable=False)
    compatibility_score = Column(Float, default=0.0)
    rl_agent = Column(String(64), default="PPO")
    explanation = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)


class MetricModel(Base):
    __tablename__ = "metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    episode = Column(Integer, nullable=False)
    reward = Column(Float, nullable=False)
    satisfaction = Column(Float, nullable=False)
    acceptance = Column(Float, nullable=False)
    algorithm = Column(String(64), default="PPO")
    timestamp = Column(DateTime, default=datetime.utcnow)
