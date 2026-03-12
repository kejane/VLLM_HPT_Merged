"""Tests for vllm_hpt.tuning.strategies module."""

import math

import optuna
import pytest

from vllm_hpt.tuning.params import SamplingParams
from vllm_hpt.tuning.strategies import STRATEGY_REGISTRY, create_strategy
from vllm_hpt.tuning.strategies.base import SearchStrategy
from vllm_hpt.tuning.strategies.cmaes import CMAESStrategy
from vllm_hpt.tuning.strategies.gp import GPStrategy
from vllm_hpt.tuning.strategies.grid import GridSearchStrategy
from vllm_hpt.tuning.strategies.tpe import TPEStrategy

optuna.logging.set_verbosity(optuna.logging.WARNING)

_has_torch = True
try:
    import torch  # noqa: F401
except ImportError:
    _has_torch = False

requires_torch = pytest.mark.skipif(not _has_torch, reason="torch not installed (optional: uv sync --extra gp)")


# ---------------------------------------------------------------------------
# SearchStrategy ABC
# ---------------------------------------------------------------------------

class TestSearchStrategyABC:
    """SearchStrategy cannot be instantiated directly."""

    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            SearchStrategy(study_name="x", storage_path="x.db")


# ---------------------------------------------------------------------------
# create_strategy factory
# ---------------------------------------------------------------------------

class TestCreateStrategy:
    """Factory function returns correct types and rejects unknowns."""

    @pytest.mark.parametrize(
        "name,expected_type",
        [
            ("tpe", TPEStrategy),
            pytest.param("gp", GPStrategy, marks=requires_torch),
            ("cmaes", CMAESStrategy),
            ("grid", GridSearchStrategy),
        ],
    )
    def test_valid_names(self, tmp_path, name, expected_type):
        db = str(tmp_path / f"{name}.db")
        strategy = create_strategy(name, study_name=f"test_{name}", storage_path=db, seed=42)
        assert isinstance(strategy, expected_type)

    def test_invalid_name_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown strategy"):
            create_strategy("bogus", study_name="x", storage_path=str(tmp_path / "x.db"))

    def test_registry_has_four_entries(self):
        assert set(STRATEGY_REGISTRY.keys()) == {"tpe", "gp", "cmaes", "grid"}


# ---------------------------------------------------------------------------
# TPE Strategy
# ---------------------------------------------------------------------------

class TestTPEStrategy:
    """TPE suggest -> report -> suggest cycle."""

    def test_suggest_report_cycle(self, tmp_path):
        db = str(tmp_path / "tpe_test.db")
        strategy = create_strategy("tpe", study_name="test_tpe", storage_path=db, seed=42)

        params = strategy.suggest_next_params()
        assert isinstance(params, SamplingParams)

        strategy.report_result(params, 0.75)
        assert strategy.n_completed_trials() == 1
        assert strategy.best_score == 0.75
        assert strategy.best_params is not None

    def test_multiple_rounds(self, tmp_path):
        db = str(tmp_path / "tpe_multi.db")
        strategy = create_strategy("tpe", study_name="test_tpe_multi", storage_path=db, seed=42)

        scores = [0.5, 0.8, 0.6]
        for score in scores:
            p = strategy.suggest_next_params()
            strategy.report_result(p, score)

        assert strategy.n_completed_trials() == 3
        assert strategy.best_score == 0.8


# ---------------------------------------------------------------------------
# GP Strategy
# ---------------------------------------------------------------------------

@requires_torch
class TestGPStrategy:
    """GP suggest -> report -> suggest cycle."""

    def test_suggest_report_cycle(self, tmp_path):
        db = str(tmp_path / "gp_test.db")
        strategy = create_strategy("gp", study_name="test_gp", storage_path=db, seed=42)

        params = strategy.suggest_next_params()
        assert isinstance(params, SamplingParams)

        strategy.report_result(params, 0.65)
        assert strategy.n_completed_trials() == 1
        assert strategy.best_score == 0.65
        assert strategy.best_params is not None

    def test_multiple_rounds(self, tmp_path):
        db = str(tmp_path / "gp_multi.db")
        strategy = create_strategy("gp", study_name="test_gp_multi", storage_path=db, seed=42)

        scores = [0.4, 0.9, 0.7]
        for score in scores:
            p = strategy.suggest_next_params()
            strategy.report_result(p, score)

        assert strategy.n_completed_trials() == 3
        assert strategy.best_score == 0.9


# ---------------------------------------------------------------------------
# CMA-ES Strategy
# ---------------------------------------------------------------------------

class TestCMAESStrategy:
    """CMA-ES suggest -> report -> suggest cycle."""

    def test_suggest_report_cycle(self, tmp_path):
        db = str(tmp_path / "cmaes_test.db")
        strategy = create_strategy("cmaes", study_name="test_cmaes", storage_path=db, seed=42)

        params = strategy.suggest_next_params()
        assert isinstance(params, SamplingParams)

        strategy.report_result(params, 0.55)
        assert strategy.n_completed_trials() == 1
        assert strategy.best_score == 0.55
        assert strategy.best_params is not None

    def test_multiple_rounds(self, tmp_path):
        db = str(tmp_path / "cmaes_multi.db")
        strategy = create_strategy("cmaes", study_name="test_cmaes_multi", storage_path=db, seed=42)

        scores = [0.3, 0.7, 0.5]
        for score in scores:
            p = strategy.suggest_next_params()
            strategy.report_result(p, score)

        assert strategy.n_completed_trials() == 3
        assert strategy.best_score == 0.7


# ---------------------------------------------------------------------------
# Grid Strategy
# ---------------------------------------------------------------------------

class TestGridStrategy:
    """Grid search with correct total combos and grid_values parameter."""

    def test_suggest_report_cycle(self, tmp_path):
        db = str(tmp_path / "grid_test.db")
        strategy = create_strategy("grid", study_name="test_grid", storage_path=db, seed=42, grid_values=2)

        params = strategy.suggest_next_params()
        assert isinstance(params, SamplingParams)

        strategy.report_result(params, 0.60)
        assert strategy.n_completed_trials() == 1
        assert strategy.best_score == 0.60
        assert strategy.best_params is not None

    def test_grid_values_affects_combos(self, tmp_path):
        db = str(tmp_path / "grid_combos.db")
        strategy = create_strategy("grid", study_name="test_grid_combos", storage_path=db, grid_values=2)
        assert isinstance(strategy, GridSearchStrategy)
        total = math.prod(len(v) for v in strategy._grid_search_space.values())
        assert total >= 2

    def test_grid_values_3_more_combos(self, tmp_path):
        db2 = str(tmp_path / "grid2.db")
        db3 = str(tmp_path / "grid3.db")
        s2 = create_strategy("grid", study_name="g2", storage_path=db2, grid_values=2)
        s3 = create_strategy("grid", study_name="g3", storage_path=db3, grid_values=3)
        total2 = math.prod(len(v) for v in s2._grid_search_space.values())
        total3 = math.prod(len(v) for v in s3._grid_search_space.values())
        assert total3 > total2


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestStrategyPersistence:
    """Strategy state persists across instances via SQLite."""

    def test_persistence_across_instances(self, tmp_path):
        db = str(tmp_path / "persist.db")
        study_name = "test_persist"

        s1 = create_strategy("tpe", study_name=study_name, storage_path=db, seed=42)
        p = s1.suggest_next_params()
        s1.report_result(p, 0.85)
        assert s1.n_completed_trials() == 1

        s2 = create_strategy("tpe", study_name=study_name, storage_path=db, seed=42)
        assert s2.n_completed_trials() == 1
        assert s2.best_score == 0.85
