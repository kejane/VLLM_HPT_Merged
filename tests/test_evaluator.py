"""Tests for vllm_hpt.evaluation.evaluator module."""

import pytest

from vllm_hpt.evaluation.evaluator import (
    EvaluationResult,
    WrongQuestion,
    evaluate,
)
from vllm_hpt.exam.loader import Question


def _make_question(qid: str, answer_key: str) -> Question:
    """Helper to create a Question with minimal fields."""
    return Question(
        id=qid,
        question=f"Question {qid}?",
        choices=["A. opt1", "B. opt2", "C. opt3", "D. opt4"],
        answer_key=answer_key,
        original_label=answer_key,
    )


# ---------------------------------------------------------------------------
# evaluate() - correct answers
# ---------------------------------------------------------------------------

class TestEvaluateCorrect:
    """Test evaluate with correct answers."""

    def test_all_correct(self):
        questions = [_make_question("1", "A"), _make_question("2", "B")]
        outputs = ["The answer is A", "The answer is B"]
        result = evaluate(questions, outputs)
        assert result.accuracy == 1.0
        assert result.correct_count == 2
        assert result.total_count == 2
        assert result.wrong_questions == []
        assert result.parse_failure_count == 0

    def test_single_correct(self):
        questions = [_make_question("1", "C")]
        outputs = ["C."]
        result = evaluate(questions, outputs)
        assert result.accuracy == 1.0
        assert result.correct_count == 1


# ---------------------------------------------------------------------------
# evaluate() - wrong answers
# ---------------------------------------------------------------------------

class TestEvaluateWrong:
    """Test evaluate with wrong answers."""

    def test_all_wrong(self):
        questions = [_make_question("1", "A"), _make_question("2", "B")]
        outputs = ["The answer is B", "The answer is A"]
        result = evaluate(questions, outputs)
        assert result.accuracy == 0.0
        assert result.correct_count == 0
        assert len(result.wrong_questions) == 2

    def test_mixed_correct_wrong(self):
        questions = [
            _make_question("1", "A"),
            _make_question("2", "B"),
            _make_question("3", "C"),
            _make_question("4", "D"),
        ]
        outputs = [
            "The answer is A",  # correct
            "The answer is C",  # wrong
            "The answer is C",  # correct
            "The answer is A",  # wrong
        ]
        result = evaluate(questions, outputs)
        assert result.accuracy == 0.5
        assert result.correct_count == 2
        assert len(result.wrong_questions) == 2

    def test_wrong_question_details(self):
        questions = [_make_question("Q1", "A")]
        outputs = ["The answer is B"]
        result = evaluate(questions, outputs)
        wq = result.wrong_questions[0]
        assert isinstance(wq, WrongQuestion)
        assert wq.question_id == "Q1"
        assert wq.model_answer == "B"
        assert wq.correct_answer == "A"
        assert wq.model_output_raw == "The answer is B"


# ---------------------------------------------------------------------------
# evaluate() - unparseable answers
# ---------------------------------------------------------------------------

class TestEvaluateUnparseable:
    """Test evaluate with unparseable outputs."""

    def test_unparseable_counted_as_wrong(self):
        questions = [_make_question("1", "A")]
        outputs = ["I have no idea what the answer might be"]
        result = evaluate(questions, outputs)
        assert result.accuracy == 0.0
        assert result.parse_failure_count == 1
        assert len(result.wrong_questions) == 1
        assert result.wrong_questions[0].model_answer is None

    def test_mixed_with_unparseable(self):
        questions = [
            _make_question("1", "A"),
            _make_question("2", "B"),
            _make_question("3", "C"),
        ]
        outputs = [
            "The answer is A",  # correct
            "gibberish",        # unparseable -> wrong
            "The answer is D",  # wrong
        ]
        result = evaluate(questions, outputs)
        assert result.correct_count == 1
        assert result.total_count == 3
        assert result.parse_failure_count == 1
        assert len(result.wrong_questions) == 2


# ---------------------------------------------------------------------------
# evaluate() - edge cases
# ---------------------------------------------------------------------------

class TestEvaluateEdgeCases:
    """Test evaluate edge cases."""

    def test_length_mismatch_raises(self):
        questions = [_make_question("1", "A")]
        outputs = ["A", "B"]
        with pytest.raises(ValueError, match="Mismatch"):
            evaluate(questions, outputs)

    def test_empty_lists(self):
        result = evaluate([], [])
        assert result.accuracy == 0.0
        assert result.total_count == 0


# ---------------------------------------------------------------------------
# EvaluationResult.summary()
# ---------------------------------------------------------------------------

class TestEvaluationResultSummary:
    """Test summary generation."""

    def test_summary_contains_key_info(self):
        result = EvaluationResult(
            accuracy=0.75,
            correct_count=3,
            total_count=4,
            wrong_questions=[
                WrongQuestion(
                    question_id="Q1",
                    question_text="What?",
                    choices=["A. x"],
                    model_answer="B",
                    correct_answer="A",
                    model_output_raw="B",
                )
            ],
            parse_failure_count=0,
        )
        s = result.summary()
        assert "75.00%" in s
        assert "Total Questions: 4" in s
        assert "Correct Answers: 3" in s
        assert "Q1" in s
