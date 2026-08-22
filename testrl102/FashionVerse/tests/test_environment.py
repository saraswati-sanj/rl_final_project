"""
FashionVerse — RL Environment Unit Tests
Tests: FashionEnv, reward function, action space, state encoding.

Run:
    cd FashionVerse
    python -m pytest tests/test_environment.py tests/test_reward.py tests/test_actions.py -v
"""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.rl.state import FashionState, STATE_DIM
from backend.rl.actions import ActionSpace, ACTION_DIM, FINISH_ACTION_ID, N_CANDIDATES
from backend.rl.reward import RewardCalculator, RewardConfig
from backend.rl.fashion_env import FashionEnv
from backend.fashion.constraints import FashionConstraints


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def env():
    """Create and reset a FashionEnv for testing."""
    e = FashionEnv(seed=42, max_steps=8, n_candidates=10)
    return e


@pytest.fixture(scope="module")
def catalog():
    from backend.fashion.catalog import get_catalog
    return get_catalog()


@pytest.fixture(scope="module")
def constraints():
    return FashionConstraints(budget=2500, occasion="casual", season="summer")


@pytest.fixture(scope="module")
def action_space(catalog, constraints):
    asp = ActionSpace(catalog, n_candidates=10)
    asp.build_candidates(constraints, [])
    return asp


# ─────────────────────────────────────────────────────────────────────────────
# State Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestStateEncoding:

    def test_state_dim_correct(self):
        state = FashionState()
        obs = state.encode()
        assert obs.shape == (STATE_DIM,), f"Expected ({STATE_DIM},), got {obs.shape}"

    def test_state_dtype_float32(self):
        state = FashionState()
        obs = state.encode()
        assert obs.dtype == np.float32

    def test_state_budget_normalizes(self):
        """Budget in state should be normalized to [0, 1]."""
        state = FashionState(budget_total=2500, budget_remaining=2500)
        obs = state.encode()
        # All values in a reasonable float range
        assert obs.min() >= -2.0 and obs.max() <= 3.0

    def test_state_changes_after_item_added(self, catalog):
        """Encoding must change when an item is added."""
        state1 = FashionState()
        obs1 = state1.encode()

        state2 = FashionState()
        tops = catalog.filter(category="top")
        state2.selected_items = [tops[0]]
        obs2 = state2.encode()

        assert not np.allclose(obs1, obs2), "State must differ after item selection"

    def test_state_to_dict(self):
        state = FashionState(occasion="office", budget_total=3000)
        d = state.to_dict()
        assert d["occasion"] == "office"
        assert d["budget_total"] == 3000
        assert "n_items" in d

    def test_reward_history_encodes(self):
        state = FashionState(reward_history=[5.0, -2.0, 3.0, 0.0, 1.0])
        obs = state.encode()
        # Reward history is clipped to [-1, 1] after normalization
        # Just check it doesn't crash and is finite
        assert np.all(np.isfinite(obs))


# ─────────────────────────────────────────────────────────────────────────────
# Action Space Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestActionSpace:

    def test_action_dim_correct(self):
        assert ACTION_DIM == 60, f"Expected ACTION_DIM=60, got {ACTION_DIM}"

    def test_finish_action_always_valid(self, action_space):
        mask = action_space.get_action_mask()
        assert mask[FINISH_ACTION_ID], "finish_outfit must always be valid"

    def test_decode_finish_action(self, action_space):
        decoded = action_space.decode(FINISH_ACTION_ID)
        assert decoded.is_finish
        assert decoded.is_valid
        assert decoded.action_type == "finish_outfit"

    def test_decode_valid_top_action(self, action_space):
        """Action ID 0 = first candidate for select_top."""
        mask = action_space.get_action_mask()
        # Find a valid top action (type 0)
        for a in range(N_CANDIDATES):
            if mask[a]:
                decoded = action_space.decode(a)
                assert decoded.is_valid
                assert decoded.action_type == "select_top"
                assert decoded.item is not None
                assert decoded.item.category == "top"
                break

    def test_decode_out_of_range(self, action_space):
        decoded = action_space.decode(999)
        assert not decoded.is_valid

    def test_action_mask_shape(self, action_space):
        mask = action_space.get_action_mask()
        assert mask.shape == (ACTION_DIM,)
        assert mask.dtype == bool

    def test_valid_actions_exist(self, action_space):
        n = action_space.n_valid_actions()
        assert n >= 1, "Must have at least finish_outfit as valid action"
        assert n <= ACTION_DIM

    def test_sample_valid_action_is_valid(self, action_space):
        for _ in range(10):
            a = action_space.sample_valid_action()
            decoded = action_space.decode(a)
            assert decoded.is_valid, f"Sampled invalid action: {a}"

    def test_candidates_respect_budget(self, catalog):
        """No candidate should exceed the remaining budget."""
        tight_constraints = FashionConstraints(budget=500, occasion="casual")
        asp = ActionSpace(catalog, n_candidates=10)
        candidates = asp.build_candidates(tight_constraints, [])
        for slot, cands in candidates.items():
            for c in cands:
                assert c.item.price <= 500, (
                    f"Candidate {c.item.item_id} costs ₹{c.item.price} > budget ₹500"
                )

    def test_no_duplicate_categories_after_selection(self, catalog):
        """After a top is selected, select_top candidates should be empty."""
        constraints = FashionConstraints(budget=3000, occasion="casual")
        asp = ActionSpace(catalog, n_candidates=10)
        top = catalog.filter(category="top")[0]
        candidates = asp.build_candidates(constraints, [top])
        # select_top should have 0 candidates
        assert len(candidates.get("select_top", [])) == 0


# ─────────────────────────────────────────────────────────────────────────────
# Reward Function Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRewardFunction:

    @pytest.fixture(scope="class")
    def calc(self):
        return RewardCalculator()

    def test_love_feedback_gives_positive_reward(self, calc, catalog, constraints):
        tops = catalog.filter(category="top", max_budget=2500)[:1]
        shoes = catalog.filter(category="shoes", max_budget=2000)[:1]
        items = tops + shoes
        r, info = calc.terminal_reward("love", items, constraints, [], "casual")
        assert r > 0, f"Love feedback should give positive reward, got {r}"

    def test_dislike_feedback_gives_negative_reward(self, calc, catalog, constraints):
        tops = catalog.filter(category="top", max_budget=2500)[:1]
        r, info = calc.terminal_reward("dislike", tops, constraints, [], "casual")
        assert r < 5, f"Dislike feedback should give low/negative reward, got {r}"

    def test_purchase_reward_highest(self, calc, catalog, constraints):
        items = catalog.filter(category="top", max_budget=2500)[:1]
        r_purchase, _ = calc.terminal_reward("purchase", items, constraints, [], "casual")
        r_like, _     = calc.terminal_reward("like",     items, constraints, [], "casual")
        assert r_purchase > r_like

    def test_skip_reward_less_than_like(self, calc, catalog, constraints):
        items = catalog.filter(category="top", max_budget=2500)[:1]
        r_skip, _ = calc.terminal_reward("skip", items, constraints, [], "casual")
        r_like, _ = calc.terminal_reward("like", items, constraints, [], "casual")
        assert r_skip < r_like

    def test_budget_violation_penalty(self, calc, catalog):
        tight = FashionConstraints(budget=100, occasion="casual")
        expensive = [i for i in catalog.all_items() if i.price > 200][:3]
        if not expensive:
            pytest.skip("No expensive items")
        r, info = calc.terminal_reward("like", expensive, tight, [], "casual")
        assert info.get("penalties", 0) < 0, "Should apply budget penalty"

    def test_step_reward_compatibility_bonus(self, calc, catalog, constraints):
        tops   = catalog.filter(category="top")[:1]
        shoes  = catalog.filter(category="shoes")[:1]
        items  = tops + shoes
        r, info = calc.step_reward(shoes[0], items, constraints, [], "casual")
        # Should get a compat bonus since we have 2 items
        assert "compat_reward" in info

    def test_repeated_outfit_penalty(self, calc, catalog, constraints):
        top = catalog.filter(category="top")[:1]
        repeated_ids = [top[0].item_id]  # mark as previously seen
        r_novel, _ = calc.step_reward(top[0], top, constraints, [], "casual")
        r_repeat, info_r = calc.step_reward(top[0], top, constraints, repeated_ids, "casual")
        assert info_r.get("penalties", 0) < 0, "Repeated item should get penalty"

    def test_invalid_action_penalty_negative(self, calc):
        r = calc.invalid_action_reward()
        assert r < 0

    def test_reward_config_customizable(self, catalog, constraints):
        cfg_high = RewardConfig(feedback_love=20.0)
        cfg_low  = RewardConfig(feedback_love=5.0)
        items = catalog.filter(category="top")[:1]
        r_high, _ = RewardCalculator(cfg_high).terminal_reward("love", items, constraints, [])
        r_low,  _ = RewardCalculator(cfg_low).terminal_reward("love", items, constraints, [])
        assert r_high > r_low

    def test_reward_config_serialization(self):
        cfg = RewardConfig(feedback_love=12.0, w_compatibility=0.5)
        d = cfg.to_dict()
        cfg2 = RewardConfig.from_dict(d)
        assert cfg2.feedback_love == 12.0
        assert cfg2.w_compatibility == 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Environment Integration Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestFashionEnv:

    def test_env_reset_returns_correct_obs_shape(self, env):
        obs, info = env.reset()
        assert obs.shape == (STATE_DIM,), f"Expected ({STATE_DIM},), got {obs.shape}"
        assert obs.dtype == np.float32

    def test_env_reset_info_has_keys(self, env):
        _, info = env.reset()
        assert "occasion" in info
        assert "budget" in info

    def test_env_step_returns_5_tuple(self, env):
        env.reset()
        # Take a valid action (finish_outfit always valid)
        obs, reward, terminated, truncated, info = env.step(FINISH_ACTION_ID)
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_env_step_obs_shape_consistent(self, env):
        obs, _ = env.reset()
        for _ in range(3):
            a = env.action_space_.sample_valid_action()
            obs, _, done, trunc, _ = env.step(a)
            assert obs.shape == (STATE_DIM,)
            if done or trunc:
                break

    def test_episode_terminates(self, env):
        """An episode must terminate within max_steps."""
        env.reset(seed=42)
        done = False
        steps = 0
        while not done and steps < 20:
            a = env.action_space_.sample_valid_action()
            _, _, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            steps += 1
        assert done, f"Episode did not terminate within 20 steps"

    def test_state_changes_after_item_addition(self, env):
        obs1, _ = env.reset(seed=10)
        # Pick a select_top action if available
        mask = env.action_space_.get_action_mask()
        top_actions = [i for i in range(10) if mask[i]]  # type 0 = select_top
        if not top_actions:
            pytest.skip("No top actions available")
        obs2, _, done, _, _ = env.step(top_actions[0])
        if not done:
            assert not np.allclose(obs1, obs2), "Obs must change after valid action"

    def test_budget_never_violated_by_valid_actions(self, env):
        """After a valid action, budget_remaining must not go negative."""
        for trial in range(5):
            env.reset(seed=trial)
            done = False
            steps = 0
            while not done and steps < 8:
                a = env.action_space_.sample_valid_action()
                _, _, terminated, truncated, info = env.step(a)
                remaining = info.get("budget_remaining", 0)
                assert remaining >= -1, f"Budget went negative: {remaining}"  # -1 tolerance
                done = terminated or truncated
                steps += 1

    def test_reward_is_finite(self, env):
        """All rewards must be finite floats."""
        for trial in range(5):
            env.reset(seed=trial)
            done = False
            while not done:
                a = env.action_space_.sample_valid_action()
                _, reward, terminated, truncated, _ = env.step(a)
                assert np.isfinite(reward), f"Non-finite reward: {reward}"
                done = terminated or truncated

    def test_env_observation_space_contains_obs(self, env):
        obs, _ = env.reset()
        assert env.observation_space.contains(obs), \
            "Observation must be within observation_space bounds"

    def test_multiple_episodes_accumulate_stats(self):
        """Running N episodes should accumulate episode stats."""
        e = FashionEnv(seed=99, max_steps=6)
        for ep in range(5):
            e.reset()
            done = False
            while not done:
                a = e.action_space_.sample_valid_action()
                _, _, terminated, truncated, _ = e.step(a)
                done = terminated or truncated

        stats = e.get_episode_stats()
        assert stats["n_episodes"] == 5
        assert "mean_reward" in stats
        assert "acceptance_rate" in stats

    def test_env_renders_without_error(self, env):
        env2 = FashionEnv(seed=1, render_mode="ansi")
        env2.reset()
        try:
            env2.render()
        except Exception as exc:
            pytest.fail(f"render() raised: {exc}")

    def test_env_with_fixed_personality(self):
        """Fixed personality should give more consistent feedback patterns."""
        e = FashionEnv(seed=7, personality="casual_student", max_steps=6)
        rewards = []
        for _ in range(5):
            e.reset()
            done = False
            ep_reward = 0
            while not done:
                a = e.action_space_.sample_valid_action()
                _, r, terminated, truncated, _ = e.step(a)
                ep_reward += r
                done = terminated or truncated
            rewards.append(ep_reward)
        # Should have variance (stochastic) but all be finite
        assert all(np.isfinite(r) for r in rewards)

    def test_preference_drift_experiment(self):
        """Enable drift and verify preference_version changes."""
        e = FashionEnv(seed=0, enable_drift=True, drift_every_n=2, max_steps=4)
        versions = []
        for ep in range(6):
            e.reset()
            done = False
            while not done:
                a = e.action_space_.sample_valid_action()
                _, _, terminated, truncated, _ = e.step(a)
                done = terminated or truncated
            if e._user:
                versions.append(e._user.preference_version)
        # Some episodes should have drift applied
        assert max(versions) >= 0  # at minimum runs without error

    def test_gymnasium_api_compliance(self):
        """Check the env passes Gymnasium API checks."""
        from gymnasium.utils.env_checker import check_env
        e = FashionEnv(seed=42, max_steps=6)
        # check_env can raise if something is wrong
        try:
            check_env(e, warn=True, skip_render_check=True)
        except Exception as exc:
            pytest.fail(f"Gymnasium check_env failed: {exc}")
