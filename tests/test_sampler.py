"""Tests for vllm_hpt.exam.sampler module."""

import pytest

from vllm_hpt.exam.loader import Question
from vllm_hpt.exam.sampler import sample_mini_exam


def _make_questions(n: int) -> list[Question]:
    return [
        Question(
            id=f"Q{i}",
            question=f"Question {i}?",
            choices=["A. x", "B. y", "C. z", "D. w"],
            answer_key="A",
            original_label="A",
        )
        for i in range(n)
    ]


class TestSampleMiniExam:

    def test_samples_correct_count(self):
        questions = _make_questions(500)
        sampled = sample_mini_exam(questions, n=200, seed=42)
        assert len(sampled) == 200

    def test_returns_all_when_n_exceeds_total(self):
        questions = _make_questions(50)
        sampled = sample_mini_exam(questions, n=200)
        assert len(sampled) == 50

    def test_returns_all_when_n_equals_total(self):
        questions = _make_questions(200)
        sampled = sample_mini_exam(questions, n=200)
        assert len(sampled) == 200

    def test_seed_reproducibility(self):
        questions = _make_questions(500)
        s1 = sample_mini_exam(questions, n=100, seed=42)
        s2 = sample_mini_exam(questions, n=100, seed=42)
        assert [q.id for q in s1] == [q.id for q in s2]

    def test_different_seeds_different_results(self):
        questions = _make_questions(500)
        s1 = sample_mini_exam(questions, n=100, seed=1)
        s2 = sample_mini_exam(questions, n=100, seed=2)
        assert [q.id for q in s1] != [q.id for q in s2]

    def test_no_duplicates(self):
        questions = _make_questions(500)
        sampled = sample_mini_exam(questions, n=200, seed=42)
        ids = [q.id for q in sampled]
        assert len(ids) == len(set(ids))

    def test_returns_question_objects(self):
        questions = _make_questions(50)
        sampled = sample_mini_exam(questions, n=10, seed=1)
        for q in sampled:
            assert isinstance(q, Question)
