"""
FashionVerse — Backend API Unit Tests
Tests: /chat, /recommend, /feedback, /avatar, /analytics, /user, /rl-status.

Run:
    cd FashionVerse
    python -m pytest tests/test_api.py -v
"""

import os, sys
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.main import app
from backend.database.database import init_db

client = TestClient(app)


@pytest.fixture(scope="session", autouse=True)
def setup_test_db():
    init_db()


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "FashionVerse" in data["name"]


def test_rl_status_endpoint():
    response = client.get("/rl-status")
    assert response.status_code == 200
    data = response.json()
    assert "algorithm" in data
    assert data["algorithm"] == "PPO"


def test_avatar_endpoints():
    r1 = client.get("/avatar/config?user_id=test_user")
    assert r1.status_code == 200
    assert "skin_tone" in r1.json()

    r2 = client.get("/avatar/materials")
    assert r2.status_code == 200
    assert "color_map" in r2.json()


def test_recommend_endpoint():
    payload = {
        "user_id": "test_user_rec",
        "occasion": "casual",
        "budget": 2500,
        "gender": "unisex",
    }
    response = client.post("/recommend", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "outfit" in data
    assert isinstance(data["outfit"], list)
    assert len(data["outfit"]) >= 2
    assert "compatibility_score" in data
    assert "explanation" in data


def test_chat_endpoint():
    payload = {
        "user_id": "test_user_chat",
        "message": "I need a semi-formal outfit for a college presentation under ₹2500.",
    }
    response = client.post("/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == "test_user_chat"
    assert "reply" in data
    assert "constraints" in data
    assert data["constraints"]["occasion"] in ("college", "semi_formal")
    assert data["constraints"]["budget"] == 2500
    assert "outfit" in data
    assert "explanation" in data


def test_feedback_endpoint_love():
    # 1. Get a recommendation first
    rec_res = client.post("/recommend", json={"user_id": "test_user_fb", "budget": 3000})
    outfit = rec_res.json()["outfit"]
    item_ids = [i["item_id"] for i in outfit]

    # 2. Submit explicit 'love' feedback
    fb_payload = {
        "user_id": "test_user_fb",
        "outfit_id": "outfit_001",
        "feedback": "love",
        "item_ids": item_ids,
        "occasion": "casual",
    }
    response = client.post("/feedback", json=fb_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["computed_reward"] > 0
    assert "updated_profile" in data


def test_feedback_endpoint_dislike():
    fb_payload = {
        "user_id": "test_user_dislike",
        "outfit_id": "outfit_002",
        "feedback": "dislike",
        "item_ids": [],
        "occasion": "formal",
    }
    response = client.post("/feedback", json=fb_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["computed_reward"] < 0


def test_user_endpoint():
    response = client.get("/user/test_user_profile")
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == "test_user_profile"
    assert "style_estimates" in data
    assert "acceptance_rate" in data


def test_analytics_endpoints():
    r1 = client.get("/analytics/summary")
    assert r1.status_code == 200
    data1 = r1.json()
    assert "db_stats" in data1

    r2 = client.get("/analytics/experiments")
    assert r2.status_code == 200
    data2 = r2.json()
    assert "exp1" in data2 or "exp1_baseline_comparison" in data2

    r3 = client.get("/analytics/recent-interactions")
    assert r3.status_code == 200
    assert isinstance(r3.json(), list)
