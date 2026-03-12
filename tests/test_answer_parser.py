"""Tests for vllm_hpt.evaluation.answer_parser module."""

import pytest

from vllm_hpt.evaluation.answer_parser import (
    AnswerParserStats,
    parse_answer,
    parse_answers,
)


# ---------------------------------------------------------------------------
# parse_answer - Level 1: "answer is [A-D]"
# ---------------------------------------------------------------------------

class TestParseAnswerLevel1:
    """Level 1: Match 'answer is [A-D]' pattern."""

    def test_the_best_answer_is(self):
        assert parse_answer("The best answer is A") == "A"

    def test_answer_is_b(self):
        assert parse_answer("The answer is B") == "B"

    def test_case_insensitive(self):
        assert parse_answer("the ANSWER IS c") == "C"

    def test_answer_is_with_extra_text(self):
        assert parse_answer("I think the answer is D because...") == "D"

    def test_answer_is_with_whitespace(self):
        assert parse_answer("answer  is  A") == "A"

    def test_strict_colon_format(self):
        assert parse_answer("The best answer is: D") == "D"

    def test_strict_colon_format_correct_answer(self):
        assert parse_answer("The correct answer is: B") == "B"

    def test_strict_colon_format_with_markdown_letter(self):
        assert parse_answer("The best answer is: **C**") == "C"

    def test_markdown_bold_letter(self):
        assert parse_answer("The best answer is **D**.") == "D"

    def test_markdown_bold_choice(self):
        assert parse_answer("The best answer is **Choice B**.") == "B"

    def test_correct_answer_markdown_choice(self):
        assert parse_answer("The correct answer is **Choice C**.") == "C"


# ---------------------------------------------------------------------------
# parse_answer - Level 2: standalone [A-D] followed by period
# ---------------------------------------------------------------------------

class TestParseAnswerLevel2:
    """Level 2: Match standalone [A-D] followed by period."""

    def test_letter_dot(self):
        assert parse_answer("B. is the correct choice") == "B"

    def test_letter_dot_end(self):
        assert parse_answer("The correct one is C. ") == "C"

    def test_letter_dot_newline(self):
        assert parse_answer("D.\n") == "D"


# ---------------------------------------------------------------------------
# parse_answer - Level 3: first [A-D] at start
# ---------------------------------------------------------------------------

class TestParseAnswerLevel3:
    """Level 3: Match first [A-D] at start of output."""

    def test_starts_with_letter(self):
        assert parse_answer("A") == "A"

    def test_starts_with_letter_and_text(self):
        assert parse_answer("B is my choice") == "B"

    def test_leading_whitespace(self):
        assert parse_answer("  C") == "C"

    def test_boxed_answer(self):
        assert parse_answer(r"\boxed{C}") == "C"


# ---------------------------------------------------------------------------
# parse_answer - None return
# ---------------------------------------------------------------------------

class TestParseAnswerNone:
    """Test cases where parsing should return None."""

    def test_empty_string(self):
        assert parse_answer("") is None

    def test_no_answer_label(self):
        assert parse_answer("I don't know the answer") is None

    def test_lowercase_only_in_middle(self):
        """Lowercase letters in the middle shouldn't match level 3."""
        assert parse_answer("the letter a is nice") is None

    def test_none_input(self):
        """None-like empty input."""
        assert parse_answer("") is None

    def test_only_numbers(self):
        assert parse_answer("12345") is None


# ---------------------------------------------------------------------------
# parse_answers (batch)
# ---------------------------------------------------------------------------

class TestParseAnswers:
    """Test batch parsing with statistics."""

    def test_batch_parsing(self):
        outputs = [
            "The answer is A",
            "B.",
            "C",
            "The best answer is **Choice D**.",
            r"\boxed{B}",
            "no answer here",
        ]
        answers, stats = parse_answers(outputs)
        assert answers == ["A", "B", "C", "D", "B", None]
        assert stats.successful == 5
        assert stats.failed == 1
        assert stats.total_parsed == 6

    def test_empty_list(self):
        answers, stats = parse_answers([])
        assert answers == []
        assert stats.total_parsed == 0

    def test_all_failures(self):
        outputs = ["xyz", "123", "no match"]
        answers, stats = parse_answers(outputs)
        assert all(a is None for a in answers)
        assert stats.failed == 3
        assert stats.successful == 0


# ---------------------------------------------------------------------------
# AnswerParserStats
# ---------------------------------------------------------------------------

class TestAnswerParserStats:
    """Test stats tracking."""

    def test_initial_state(self):
        stats = AnswerParserStats()
        assert stats.total_parsed == 0
        assert stats.successful == 0
        assert stats.failed == 0
        assert stats.level_counts == {}

    def test_record_success(self):
        stats = AnswerParserStats()
        stats.record_success(1)
        stats.record_success(1)
        stats.record_success(2)
        assert stats.total_parsed == 3
        assert stats.successful == 3
        assert stats.level_counts == {1: 2, 2: 1}

    def test_record_failure(self):
        stats = AnswerParserStats()
        stats.record_failure()
        assert stats.total_parsed == 1
        assert stats.failed == 1

    def test_level_tracking_in_batch(self):
        """Verify level counts are tracked correctly in batch parsing."""
        outputs = [
            "The best answer is: A",   # Level 1
            "The best answer is **Choice B**.",  # Level 2
            r"\boxed{C}",  # Level 3
            "D.",  # Level 4
            "A is my choice",  # Level 5
        ]
        _, stats = parse_answers(outputs)
        assert stats.level_counts.get(1, 0) == 1
        assert stats.level_counts.get(2, 0) == 1
        assert stats.level_counts.get(3, 0) == 1
        assert stats.level_counts.get(4, 0) == 1
        assert stats.level_counts.get(5, 0) == 1
