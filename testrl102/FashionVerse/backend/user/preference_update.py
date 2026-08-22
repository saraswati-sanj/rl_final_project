"""
FashionVerse — Preference Update Engine
Manages how the observable user profile is updated after each RL step.
Separate from user_profile.py to keep update logic testable in isolation.
"""

from typing import List, Optional
from backend.user.user_profile import UserProfile
from backend.user.user_simulator import FEEDBACK_VALUES


def update_preference(
    profile: UserProfile,
    items: List,
    feedback: str,
    learning_rate: float = 0.05,
) -> UserProfile:
    """
    Update a UserProfile based on outfit feedback.
    Returns the updated profile (mutates in-place and also returns it).

    Learning rule: Exponential Moving Average
        estimate_new = estimate_old + lr * (target - estimate_old)
    where target = 0.85 for positive feedback, 0.15 for negative.
    """
    profile.update_from_feedback(items, feedback, learning_rate)
    return profile


def compute_personalization_score(
    profile: UserProfile,
    true_prefs: Optional[dict] = None,
) -> float:
    """
    Measures how well the profile has converged toward true user preferences.
    If true_prefs not available (real user), falls back to like/dislike ratio.

    Returns a score in [0, 1].
    """
    if true_prefs is None:
        # Observable proxy: acceptance rate
        return profile.acceptance_rate()

    # Compare style estimates to true style preferences
    style_errors = []
    for style, true_val in true_prefs.get("style_prefs", {}).items():
        est = profile.style_estimates.get(style, 0.5)
        style_errors.append(abs(true_val - est))

    if not style_errors:
        return profile.acceptance_rate()

    avg_error = sum(style_errors) / len(style_errors)
    return max(0.0, 1.0 - avg_error * 2.0)
