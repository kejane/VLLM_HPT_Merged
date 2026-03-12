"""Tests for vllm_hpt.tuning.history module."""

import pytest

from vllm_hpt.tuning.history import HistoryEntry, OptimizationHistory
from vllm_hpt.tuning.params import SamplingParams


def _entry(round_num: int, train: float, val: float = None) -> HistoryEntry:
    return HistoryEntry(
        round_num=round_num,
        params=SamplingParams(temperature=0.5 + round_num * 0.1),
        train_accuracy=train,
        validation_accuracy=val,
    )


class TestOptimizationHistory:

    def test_add_entry(self):
        h = OptimizationHistory()
        h.add_entry(_entry(1, 0.5, 0.4))
        assert len(h.entries) == 1

    def test_get_best_validation(self):
        h = OptimizationHistory()
        h.add_entry(_entry(1, 0.5, 0.4))
        h.add_entry(_entry(2, 0.6, 0.7))
        h.add_entry(_entry(3, 0.8, 0.6))
        best = h.get_best(by="validation")
        assert best.round_num == 2
        assert best.validation_accuracy == 0.7

    def test_get_best_train(self):
        h = OptimizationHistory()
        h.add_entry(_entry(1, 0.5, 0.4))
        h.add_entry(_entry(2, 0.9, 0.3))
        best = h.get_best(by="train")
        assert best.round_num == 2
        assert best.train_accuracy == 0.9

    def test_get_best_empty(self):
        h = OptimizationHistory()
        assert h.get_best() is None

    def test_get_best_no_validation(self):
        h = OptimizationHistory()
        h.add_entry(_entry(1, 0.5))
        assert h.get_best(by="validation") is None

    def test_get_top_k(self):
        h = OptimizationHistory()
        for i in range(10):
            h.add_entry(_entry(i, 0.1 * i, 0.1 * i))
        top = h.get_top_k(k=3, by="validation")
        assert len(top) == 3
        assert top[0].round_num == 9

    def test_get_top_k_empty(self):
        h = OptimizationHistory()
        assert h.get_top_k() == []

    def test_get_top_k_fewer_than_k(self):
        h = OptimizationHistory()
        h.add_entry(_entry(1, 0.5, 0.4))
        top = h.get_top_k(k=5)
        assert len(top) == 1


class TestHistorySerialization:

    def test_to_dict(self):
        h = OptimizationHistory()
        h.add_entry(_entry(1, 0.5, 0.4))
        d = h.to_dict()
        assert isinstance(d, list)
        assert len(d) == 1
        assert d[0]["round_num"] == 1
        assert d[0]["train_accuracy"] == 0.5

    def test_from_dict_roundtrip(self):
        h = OptimizationHistory()
        h.add_entry(_entry(1, 0.5, 0.4))
        h.add_entry(_entry(2, 0.7, 0.65))
        d = h.to_dict()

        h2 = OptimizationHistory.from_dict(d)
        assert len(h2.entries) == 2
        assert h2.entries[0].round_num == 1
        assert h2.entries[1].validation_accuracy == 0.65
        assert isinstance(h2.entries[0].params, SamplingParams)

    def test_from_dict_empty(self):
        h = OptimizationHistory.from_dict([])
        assert len(h.entries) == 0


class TestFormatForPrompt:

    def test_empty_history(self):
        h = OptimizationHistory()
        result = h.format_for_prompt()
        assert "No optimization history" in result

    def test_formats_entries(self):
        h = OptimizationHistory()
        h.add_entry(_entry(1, 0.5, 0.4))
        result = h.format_for_prompt()
        assert "Round 1" in result
        assert "train_accuracy=0.5000" in result

    def test_entries_without_validation_still_shown(self):
        h = OptimizationHistory()
        h.add_entry(_entry(1, 0.5))
        result = h.format_for_prompt()
        assert "Round 1" in result
        assert "train_accuracy=0.5000" in result
