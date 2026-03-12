"""Tests for vllm_hpt.orchestrator.checkpoint module."""

import json

import pytest

from vllm_hpt.orchestrator.checkpoint import Checkpoint, CheckpointManager
from vllm_hpt.tuning.history import HistoryEntry, OptimizationHistory
from vllm_hpt.tuning.params import SamplingParams


def _make_checkpoint(run_id: str = "test_run", current_round: int = 3) -> Checkpoint:
    """Create a checkpoint with sample data."""
    history = OptimizationHistory()
    history.entries.append(
        HistoryEntry(
            round_num=1,
            params=SamplingParams(temperature=0.8, top_p=0.95),
            train_accuracy=0.65,
            validation_accuracy=0.60,
        )
    )
    history.entries.append(
        HistoryEntry(
            round_num=2,
            params=SamplingParams(temperature=1.0, top_p=0.85),
            train_accuracy=0.72,
            validation_accuracy=0.68,
        )
    )
    return Checkpoint(
        run_id=run_id,
        current_round=current_round,
        total_rounds=10,
        history=history,
        best_params=SamplingParams(temperature=1.0, top_p=0.85),
        best_validation_accuracy=0.68,
    )


# ---------------------------------------------------------------------------
# Save / Load roundtrip
# ---------------------------------------------------------------------------

class TestCheckpointSaveLoad:
    """Test save and load roundtrip."""

    def test_save_creates_file(self, tmp_path):
        mgr = CheckpointManager()
        cp = _make_checkpoint()
        filepath = str(tmp_path / "cp.json")
        mgr.save(cp, filepath)
        assert (tmp_path / "cp.json").exists()

    def test_save_load_roundtrip(self, tmp_path):
        mgr = CheckpointManager()
        cp = _make_checkpoint()
        filepath = str(tmp_path / "cp.json")
        mgr.save(cp, filepath)

        loaded = mgr.load(filepath)
        assert loaded.run_id == cp.run_id
        assert loaded.current_round == cp.current_round
        assert loaded.total_rounds == cp.total_rounds
        assert loaded.best_validation_accuracy == cp.best_validation_accuracy
        assert loaded.best_params.temperature == cp.best_params.temperature
        assert loaded.best_params.top_p == cp.best_params.top_p

    def test_history_preserved(self, tmp_path):
        mgr = CheckpointManager()
        cp = _make_checkpoint()
        filepath = str(tmp_path / "cp.json")
        mgr.save(cp, filepath)

        loaded = mgr.load(filepath)
        assert len(loaded.history.entries) == 2
        assert loaded.history.entries[0].round_num == 1
        assert loaded.history.entries[0].train_accuracy == 0.65
        assert loaded.history.entries[1].validation_accuracy == 0.68

    def test_params_preserved_in_history(self, tmp_path):
        mgr = CheckpointManager()
        cp = _make_checkpoint()
        filepath = str(tmp_path / "cp.json")
        mgr.save(cp, filepath)

        loaded = mgr.load(filepath)
        entry = loaded.history.entries[0]
        assert entry.params.temperature == 0.8
        assert entry.params.top_p == 0.95

    def test_json_is_valid(self, tmp_path):
        """Saved file should be valid JSON."""
        mgr = CheckpointManager()
        cp = _make_checkpoint()
        filepath = str(tmp_path / "cp.json")
        mgr.save(cp, filepath)

        with open(filepath) as f:
            data = json.load(f)
        assert data["run_id"] == "test_run"
        assert isinstance(data["history"], list)

    def test_creates_parent_dirs(self, tmp_path):
        mgr = CheckpointManager()
        cp = _make_checkpoint()
        filepath = str(tmp_path / "deep" / "nested" / "cp.json")
        mgr.save(cp, filepath)
        assert (tmp_path / "deep" / "nested" / "cp.json").exists()


# ---------------------------------------------------------------------------
# Load errors
# ---------------------------------------------------------------------------

class TestCheckpointLoadErrors:
    """Test error handling on load."""

    def test_file_not_found(self, tmp_path):
        mgr = CheckpointManager()
        with pytest.raises(FileNotFoundError):
            mgr.load(str(tmp_path / "nonexistent.json"))

    def test_invalid_json(self, tmp_path):
        filepath = tmp_path / "bad.json"
        filepath.write_text("{invalid json")
        mgr = CheckpointManager()
        with pytest.raises(Exception):
            mgr.load(str(filepath))

    def test_missing_keys(self, tmp_path):
        filepath = tmp_path / "incomplete.json"
        filepath.write_text(json.dumps({"run_id": "x"}))
        mgr = CheckpointManager()
        with pytest.raises((ValueError, KeyError)):
            mgr.load(str(filepath))


# ---------------------------------------------------------------------------
# auto_save_path
# ---------------------------------------------------------------------------

class TestAutoSavePath:
    """Test auto_save_path generation."""

    def test_format(self):
        mgr = CheckpointManager()
        path = mgr.auto_save_path("abc123")
        assert path == "checkpoints/run_abc123.json"

    def test_different_ids(self):
        mgr = CheckpointManager()
        assert mgr.auto_save_path("a") != mgr.auto_save_path("b")


# ---------------------------------------------------------------------------
# find_latest
# ---------------------------------------------------------------------------

class TestFindLatest:
    """Test finding the latest checkpoint."""

    def test_no_checkpoints(self, tmp_path):
        mgr = CheckpointManager()
        assert mgr.find_latest(str(tmp_path)) is None

    def test_nonexistent_dir(self):
        mgr = CheckpointManager()
        assert mgr.find_latest("/nonexistent/path") is None

    def test_finds_latest(self, tmp_path):
        import time
        mgr = CheckpointManager()

        cp1 = _make_checkpoint("run1", 1)
        cp2 = _make_checkpoint("run2", 5)

        mgr.save(cp1, str(tmp_path / "run1.json"))
        time.sleep(0.05)
        mgr.save(cp2, str(tmp_path / "run2.json"))

        latest = mgr.find_latest(str(tmp_path))
        assert latest is not None
        assert "run2.json" in latest


# ---------------------------------------------------------------------------
# strategy_name field
# ---------------------------------------------------------------------------

class TestStrategyNameField:

    def test_strategy_name_saved(self, tmp_path):
        mgr = CheckpointManager()
        cp = _make_checkpoint()
        cp.strategy_name = "gp"
        filepath = str(tmp_path / "cp.json")
        mgr.save(cp, filepath)
        loaded = mgr.load(filepath)
        assert loaded.strategy_name == "gp"

    def test_strategy_name_default(self, tmp_path):
        mgr = CheckpointManager()
        cp = _make_checkpoint()
        filepath = str(tmp_path / "cp.json")
        mgr.save(cp, filepath)
        with open(filepath) as f:
            data = json.load(f)
        del data["strategy_name"]
        with open(filepath, "w") as f:
            json.dump(data, f)
        loaded = mgr.load(filepath)
        assert loaded.strategy_name == "tpe"

    def test_study_path(self):
        mgr = CheckpointManager()
        assert mgr.study_path("run123") == "checkpoints/run123_study.db"
