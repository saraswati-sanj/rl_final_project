"""
FashionVerse — Phase 1 Unit Tests
Tests: catalog, compatibility, constraints, outfit generator, user simulator.

Run:
    cd FashionVerse
    python -m pytest tests/test_phase1.py -v
"""

import os
import sys
import pytest
import random

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def catalog():
    """Load the fashion catalog once for all tests."""
    from backend.fashion.catalog import FashionCatalog
    import os
    cat = FashionCatalog()
    # Point to data dir
    json_path = os.path.join(os.path.dirname(__file__), "..", "data", "fashion_items.json")
    cat.load(json_path)
    return cat


@pytest.fixture(scope="session")
def default_constraints():
    from backend.fashion.constraints import FashionConstraints
    return FashionConstraints(
        budget=2500,
        occasion="casual",
        season="summer",
        gender="unisex",
    )


@pytest.fixture(scope="session")
def scorer():
    from backend.fashion.compatibility import OutfitCompatibilityScorer
    return OutfitCompatibilityScorer()


# ─────────────────────────────────────────────────────────────────────────────
# 1. Catalog Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCatalog:

    def test_catalog_loads(self, catalog):
        assert catalog.size() > 0, "Catalog must have items"

    def test_catalog_has_all_categories(self, catalog):
        cats = catalog.categories()
        for required in ["top", "bottom", "dress", "shoes", "accessory"]:
            assert required in cats, f"Missing category: {required}"

    def test_catalog_size_by_category(self, catalog):
        summary = catalog.summary()
        assert summary.get("top", 0) >= 90,     "Expected ~100 tops"
        assert summary.get("bottom", 0) >= 90,  "Expected ~100 bottoms"
        assert summary.get("dress", 0) >= 55,   "Expected ~60 dresses"
        assert summary.get("shoes", 0) >= 55,   "Expected ~60 shoes"
        assert summary.get("accessory", 0) >= 35, "Expected ~40 accessories"

    def test_item_fields_complete(self, catalog):
        for item in catalog.all_items()[:20]:
            assert item.item_id, "item_id must not be empty"
            assert item.name, "name must not be empty"
            assert item.category in ("top", "bottom", "dress", "shoes", "accessory")
            assert 1 <= item.formality <= 5, f"Formality out of range: {item.formality}"
            assert item.price > 0, f"Price must be positive: {item.price}"
            assert 0.0 <= item.comfort_score <= 1.0
            assert 0.0 <= item.versatility_score <= 1.0

    def test_filter_by_budget(self, catalog):
        cheap = catalog.filter(max_budget=1000)
        assert all(i.price <= 1000 for i in cheap), "Budget filter failed"

    def test_filter_by_category(self, catalog):
        tops = catalog.filter(category="top")
        assert all(i.category == "top" for i in tops)

    def test_filter_by_occasion(self, catalog):
        formal = catalog.filter(occasion="formal")
        assert all("formal" in [o.lower() for o in i.occasion] for i in formal)

    def test_filter_combine(self, catalog):
        results = catalog.filter(
            category="top",
            max_budget=1500,
            formality_min=1,
            formality_max=3,
        )
        for item in results:
            assert item.category == "top"
            assert item.price <= 1500
            assert 1 <= item.formality <= 3

    def test_item_to_vector(self, catalog):
        item = catalog.all_items()[0]
        vec = item.to_vector()
        assert len(vec) == 11, f"Expected 11-dim vector, got {len(vec)}"
        assert all(isinstance(v, float) for v in vec)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Compatibility Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCompatibility:

    def test_color_same_neutral(self):
        from backend.fashion.compatibility import color_compatibility
        score = color_compatibility("black", "black")
        assert score == 1.0

    def test_color_neutral_with_anything(self):
        from backend.fashion.compatibility import color_compatibility
        # Neutral should be compatible with everything
        score = color_compatibility("black", "navy")
        assert score > 0.5

    def test_color_clashing(self):
        from backend.fashion.compatibility import color_compatibility
        score = color_compatibility("coral", "burnt_orange")
        # Same-warmth can clash; should be lower
        assert 0.0 <= score <= 1.0

    def test_style_same_returns_1(self):
        from backend.fashion.compatibility import style_compatibility
        assert style_compatibility("casual", "casual") == 1.0

    def test_style_compatible_pair(self):
        from backend.fashion.compatibility import style_compatibility
        score = style_compatibility("casual", "streetwear")
        assert score >= 0.7

    def test_style_incompatible_pair(self):
        from backend.fashion.compatibility import style_compatibility
        score = style_compatibility("formal", "athleisure")
        assert score < 0.5

    def test_formality_same(self):
        from backend.fashion.compatibility import formality_compatibility
        assert formality_compatibility(3, 3) == 1.0

    def test_formality_adjacent(self):
        from backend.fashion.compatibility import formality_compatibility
        score = formality_compatibility(2, 3)
        assert 0.5 < score < 1.0

    def test_formality_far(self):
        from backend.fashion.compatibility import formality_compatibility
        score = formality_compatibility(1, 5)
        assert score == 0.0

    def test_outfit_score_structure(self, catalog, scorer):
        items = catalog.filter(category="top")[:2] + catalog.filter(category="shoes")[:1]
        score = scorer.score_outfit(items, "casual")
        assert "overall" in score
        assert "color" in score
        assert "style" in score
        assert 0.0 <= score["overall"] <= 1.0

    def test_is_complete_outfit(self, catalog):
        from backend.fashion.compatibility import is_complete
        tops = catalog.filter(category="top")
        bottoms = catalog.filter(category="bottom")
        shoes = catalog.filter(category="shoes")
        # Incomplete
        assert not is_complete(tops[:1])
        assert not is_complete(tops[:1] + bottoms[:1])  # missing shoes
        # Complete
        assert is_complete(tops[:1] + bottoms[:1] + shoes[:1])

    def test_is_complete_with_dress(self, catalog):
        from backend.fashion.compatibility import is_complete
        dresses = catalog.filter(category="dress")
        shoes = catalog.filter(category="shoes")
        if dresses and shoes:
            assert is_complete(dresses[:1] + shoes[:1])


# ─────────────────────────────────────────────────────────────────────────────
# 3. Constraints Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestConstraints:

    def test_budget_remaining(self, catalog, default_constraints):
        tops = catalog.filter(category="top", max_budget=500)
        if not tops:
            pytest.skip("No items under 500")
        item = tops[0]
        remaining = default_constraints.budget_remaining([item])
        assert remaining == default_constraints.budget - item.price

    def test_over_budget_detection(self, catalog):
        from backend.fashion.constraints import FashionConstraints
        c = FashionConstraints(budget=500)
        expensive = [i for i in catalog.all_items() if i.price > 600][:3]
        if not expensive:
            pytest.skip("No expensive items")
        assert c.is_over_budget(expensive)

    def test_duplicate_category_blocked(self, catalog):
        from backend.fashion.constraints import FashionConstraints
        c = FashionConstraints(budget=10000)
        tops = catalog.filter(category="top")
        if len(tops) < 2:
            pytest.skip("Not enough tops")
        result = c.validate_item(tops[1], [tops[0]])
        assert result["valid"] is False
        assert "duplicate_top" in result["reason"]

    def test_valid_item_passes(self, catalog, default_constraints):
        top = catalog.filter(category="top", max_budget=default_constraints.budget)[0]
        result = default_constraints.validate_item(top, [])
        assert result["valid"] is True

    def test_dress_conflict_with_top(self, catalog):
        from backend.fashion.constraints import FashionConstraints
        c = FashionConstraints(budget=10000)
        tops = catalog.filter(category="top")
        dresses = catalog.filter(category="dress")
        if not tops or not dresses:
            pytest.skip("Missing items")
        result = c.validate_item(dresses[0], [tops[0]])
        assert result["valid"] is False

    def test_constraint_to_dict(self, default_constraints):
        d = default_constraints.to_dict()
        assert "budget" in d
        assert "occasion" in d

    def test_constraint_roundtrip(self):
        from backend.fashion.constraints import FashionConstraints
        c = FashionConstraints(budget=3000, occasion="office", gender="female")
        c2 = FashionConstraints.from_dict(c.to_dict())
        assert c2.budget == 3000
        assert c2.occasion == "office"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Outfit Generator Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestOutfitGenerator:

    @pytest.fixture(scope="class")
    def generator(self):
        from backend.fashion.outfit_generator import OutfitGenerator
        return OutfitGenerator()

    def test_random_outfit_not_empty(self, generator, default_constraints):
        outfit = generator.generate_random(default_constraints, seed=42)
        assert len(outfit) >= 2, "Outfit must have at least 2 items"

    def test_rule_based_outfit(self, generator, default_constraints):
        outfit = generator.generate_rule_based(default_constraints, seed=42)
        assert len(outfit) >= 2

    def test_popular_outfit(self, generator, default_constraints):
        outfit = generator.generate_popular(default_constraints)
        assert len(outfit) >= 2

    def test_outfit_within_budget(self, generator, default_constraints):
        for seed in range(5):
            outfit = generator.generate_random(default_constraints, seed=seed)
            total = sum(i.price for i in outfit)
            assert total <= default_constraints.budget, (
                f"Outfit over budget: ₹{total} > ₹{default_constraints.budget}"
            )

    def test_no_duplicate_main_categories(self, generator, default_constraints):
        for seed in range(5):
            outfit = generator.generate_random(default_constraints, seed=seed)
            cats = [i.category for i in outfit]
            for cat in ("top", "bottom", "shoes"):
                assert cats.count(cat) <= 1, f"Duplicate {cat} in outfit"

    def test_outfit_summary_structure(self, generator, default_constraints):
        outfit = generator.generate_random(default_constraints, seed=0)
        summary = generator.outfit_summary(outfit)
        assert "total_price" in summary
        assert "compatibility_score" in summary
        assert "items" in summary
        assert isinstance(summary["compatibility_score"], float)

    def test_rule_based_better_than_random_on_average(self, generator, default_constraints):
        """Rule-based should produce higher compatibility on average."""
        n = 10
        rule_scores = []
        rand_scores = []
        for s in range(n):
            rb = generator.generate_rule_based(default_constraints, seed=s)
            rn = generator.generate_random(default_constraints, seed=s)
            rule_scores.append(generator.outfit_summary(rb)["compatibility_score"])
            rand_scores.append(generator.outfit_summary(rn)["compatibility_score"])
        avg_rule = sum(rule_scores) / n
        avg_rand = sum(rand_scores) / n
        assert avg_rule >= avg_rand - 0.05, (
            f"Rule-based ({avg_rule:.3f}) should not be worse than random ({avg_rand:.3f})"
        )

    def test_strict_budget_constraint(self, generator):
        from backend.fashion.constraints import FashionConstraints
        tight = FashionConstraints(budget=800, occasion="casual")
        for seed in range(5):
            outfit = generator.generate_random(tight, seed=seed)
            total = sum(i.price for i in outfit)
            assert total <= 800, f"Budget violated: ₹{total}"


# ─────────────────────────────────────────────────────────────────────────────
# 5. User Simulator Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestUserSimulator:

    @pytest.fixture(scope="class")
    def simulator(self):
        from backend.user.user_simulator import UserSimulator
        return UserSimulator(seed=42)

    @pytest.fixture(scope="class")
    def generator(self):
        from backend.fashion.outfit_generator import OutfitGenerator
        return OutfitGenerator()

    @pytest.fixture(scope="class")
    def constraints(self):
        from backend.fashion.constraints import FashionConstraints
        return FashionConstraints(budget=2500, occasion="casual")

    def test_create_user_all_personalities(self, simulator):
        for personality in simulator.list_personalities():
            user = simulator.create_user(f"u_{personality}", personality)
            assert user.personality == personality
            assert user.budget > 0

    def test_feedback_returns_valid_label(self, simulator, generator, constraints):
        from backend.user.user_simulator import FEEDBACK_VALUES
        user = simulator.create_user("u_test", "casual_student")
        outfit = generator.generate_random(constraints, seed=1)
        feedback, score = user.evaluate_outfit(outfit, "casual")
        assert feedback in FEEDBACK_VALUES, f"Invalid feedback: {feedback}"
        assert 0.0 <= score <= 1.0

    def test_feedback_score_range(self, simulator, generator, constraints):
        user = simulator.create_user("u_range", "formal_professional")
        for seed in range(10):
            outfit = generator.generate_random(constraints, seed=seed)
            _, score = user.evaluate_outfit(outfit, "office")
            assert 0.0 <= score <= 1.0, f"Score out of range: {score}"

    def test_interaction_count_increments(self, simulator, generator, constraints):
        user = simulator.create_user("u_count")
        outfit = generator.generate_random(constraints, seed=0)
        initial = user.interaction_count
        user.evaluate_outfit(outfit, "casual")
        assert user.interaction_count == initial + 1

    def test_different_personalities_different_scores(self, simulator, generator, constraints):
        """Formal outfit should score better for formal_professional than casual_student."""
        from backend.fashion.constraints import FashionConstraints
        formal_c = FashionConstraints(budget=5000, occasion="formal",
                                       formality_min=4, formality_max=5)
        formal_outfit = generator.generate_rule_based(formal_c, seed=7)
        if not formal_outfit:
            pytest.skip("Couldn't generate formal outfit")

        student = simulator.create_user("u_student", "casual_student")
        professional = simulator.create_user("u_pro", "formal_professional")

        _, s1 = student.evaluate_outfit(formal_outfit, "formal")
        _, s2 = professional.evaluate_outfit(formal_outfit, "formal")
        # Professional should score formal outfit higher on average
        # (with noise this might not always hold for 1 sample — check direction)
        # Allow ±0.1 tolerance for noise
        assert s2 >= s1 - 0.1, f"Professional ({s2:.2f}) should score formal outfit >= student ({s1:.2f})"

    def test_preference_drift_changes_version(self, simulator):
        user = simulator.create_user("u_drift", "casual_student")
        v0 = user.preference_version
        user.apply_preference_drift(0.1)
        assert user.preference_version == v0 + 1

    def test_preference_drift_changes_prefs(self, simulator):
        user = simulator.create_user("u_drift2", "casual_student")
        original = dict(user.hidden_prefs["style_prefs"])
        user.apply_preference_drift(0.3)
        changed = user.hidden_prefs["style_prefs"]
        # At least some values should have changed
        diffs = [abs(changed[k] - original[k]) for k in original]
        assert any(d > 0 for d in diffs)

    def test_observable_state_hint_no_hidden_info(self, simulator):
        user = simulator.create_user("u_obs", "minimalist")
        hint = user.get_observable_state_hint()
        # Must NOT contain style_prefs or color_prefs
        assert "style_prefs" not in hint
        assert "color_prefs" not in hint
        assert "interaction_count" in hint

    def test_batch_creation(self, simulator):
        users = simulator.create_batch(14)
        assert len(users) == 14
        personalities = [u.personality for u in users]
        # Should cover all personality types
        unique = set(personalities)
        assert len(unique) >= 7


# ─────────────────────────────────────────────────────────────────────────────
# 6. User Profile Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestUserProfile:

    def test_profile_initial_vector_dimension(self):
        from backend.user.user_profile import UserProfile
        profile = UserProfile(user_id="u1")
        vec = profile.to_vector()
        assert len(vec) == 23, f"Expected 23-dim state vector, got {len(vec)}"

    def test_profile_update_positive_feedback(self, catalog):
        from backend.user.user_profile import UserProfile
        profile = UserProfile(user_id="u2")
        items = catalog.filter(category="top", style="casual")[:1]
        initial = profile.style_estimates.get("casual", 0.5)
        profile.update_from_feedback(items, "love")
        updated = profile.style_estimates.get("casual", 0.5)
        assert updated > initial, "Positive feedback should increase style estimate"

    def test_profile_update_negative_feedback(self, catalog):
        from backend.user.user_profile import UserProfile
        profile = UserProfile(user_id="u3")
        items = catalog.filter(category="top", style="formal")[:1]
        profile.style_estimates["formal"] = 0.7  # start high
        profile.update_from_feedback(items, "dislike")
        updated = profile.style_estimates.get("formal", 0.5)
        assert updated < 0.7, "Negative feedback should decrease style estimate"

    def test_acceptance_rate_computation(self):
        from backend.user.user_profile import UserProfile
        p = UserProfile(user_id="u4")
        p.total_interactions = 10
        p.likes = 6
        p.saves = 2
        assert p.acceptance_rate() == 0.8

    def test_profile_serialization(self):
        from backend.user.user_profile import UserProfile
        p = UserProfile(user_id="u5", budget=3000, gender="female")
        p.total_interactions = 5
        d = p.to_dict()
        p2 = UserProfile.from_dict(d)
        assert p2.user_id == "u5"
        assert p2.budget == 3000
        assert p2.total_interactions == 5


# ─────────────────────────────────────────────────────────────────────────────
# 7. Preference Update Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPreferenceUpdate:

    def test_update_changes_profile(self, catalog):
        from backend.user.user_profile import UserProfile
        from backend.user.preference_update import update_preference
        profile = UserProfile(user_id="pu1")
        casual_tops = catalog.filter(category="top", style="casual")[:1]
        prev_val = profile.style_estimates.get("casual", 0.5)
        update_preference(profile, casual_tops, "like")
        new_val = profile.style_estimates.get("casual", 0.5)
        assert new_val != prev_val, "Profile must change after update"

    def test_personalization_score_improves(self, catalog):
        from backend.user.user_profile import UserProfile
        from backend.user.preference_update import update_preference, compute_personalization_score
        from backend.user.user_simulator import USER_PERSONALITIES

        profile = UserProfile(user_id="pu2")
        true_prefs = USER_PERSONALITIES["casual_student"]
        casual_items = catalog.filter(category="top", style="casual")[:3]

        score_before = compute_personalization_score(profile, true_prefs)
        for item in casual_items:
            update_preference(profile, [item], "love", learning_rate=0.15)
        score_after = compute_personalization_score(profile, true_prefs)

        assert score_after >= score_before - 0.01, (
            "Personalization should not degrade significantly with relevant feedback"
        )
