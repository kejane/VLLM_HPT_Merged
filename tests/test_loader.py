"""Tests for vllm_hpt.exam.loader module."""

import json
import pytest

from vllm_hpt.exam.loader import Question, load_dataset, load_all_datasets, _normalize_label


# ---------------------------------------------------------------------------
# _normalize_label
# ---------------------------------------------------------------------------

class TestNormalizeLabel:
    """Tests for label normalization (1234 -> ABCD)."""

    def test_numeric_labels(self):
        assert _normalize_label("1") == "A"
        assert _normalize_label("2") == "B"
        assert _normalize_label("3") == "C"
        assert _normalize_label("4") == "D"

    def test_alpha_labels_passthrough(self):
        assert _normalize_label("A") == "A"
        assert _normalize_label("B") == "B"
        assert _normalize_label("C") == "C"
        assert _normalize_label("D") == "D"

    def test_unknown_label_passthrough(self):
        assert _normalize_label("E") == "E"
        assert _normalize_label("X") == "X"


# ---------------------------------------------------------------------------
# Question dataclass
# ---------------------------------------------------------------------------

class TestQuestion:
    """Tests for the Question dataclass."""

    def test_creation(self):
        q = Question(
            id="Q1",
            question="What is 2+2?",
            choices=["A. 3", "B. 4", "C. 5", "D. 6"],
            answer_key="B",
            original_label="B",
        )
        assert q.id == "Q1"
        assert q.answer_key == "B"
        assert len(q.choices) == 4

    def test_str_representation(self):
        q = Question(
            id="Q1",
            question="What is 2+2?",
            choices=["A. 3", "B. 4"],
            answer_key="B",
            original_label="B",
        )
        s = str(q)
        assert "Q1" in s
        assert "What is 2+2?" in s
        assert "Answer: B" in s


# ---------------------------------------------------------------------------
# load_dataset with real data
# ---------------------------------------------------------------------------

class TestLoadDataset:
    """Tests for load_dataset using real ARC-Challenge data."""

    DATA_DIR = "data/ai2_arc/ARC-Challenge"

    def test_load_train(self):
        questions = load_dataset(f"{self.DATA_DIR}/train.jsonl")
        assert len(questions) > 0
        # All questions should have valid answer keys
        for q in questions:
            assert q.answer_key in ("A", "B", "C", "D")
            assert len(q.choices) >= 2
            assert q.id

    def test_load_validation(self):
        questions = load_dataset(f"{self.DATA_DIR}/validation.jsonl")
        assert len(questions) > 0

    def test_load_test(self):
        questions = load_dataset(f"{self.DATA_DIR}/test.jsonl")
        assert len(questions) > 0

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_dataset("nonexistent/path.jsonl")

    def test_choices_formatted(self):
        """Choices should be formatted as 'X. text'."""
        questions = load_dataset(f"{self.DATA_DIR}/train.jsonl")
        q = questions[0]
        for choice in q.choices:
            # Should match pattern like "A. some text"
            assert choice[1] == ".", f"Choice not formatted correctly: {choice}"

    def test_load_from_tmp_file(self, tmp_path):
        """Test loading from a custom JSONL file."""
        data = {
            "id": "test_001",
            "question": "What color is the sky?",
            "choices": {
                "text": ["Red", "Blue", "Green", "Yellow"],
                "label": ["1", "2", "3", "4"],
            },
            "answerKey": "2",
        }
        filepath = tmp_path / "test.jsonl"
        filepath.write_text(json.dumps(data) + "\n")

        questions = load_dataset(str(filepath))
        assert len(questions) == 1
        q = questions[0]
        assert q.id == "test_001"
        # Numeric label "2" should be normalized to "B"
        assert q.answer_key == "B"
        assert q.original_label == "2"
        assert q.choices[0].startswith("A.")
        assert q.choices[1].startswith("B.")

    def test_skips_invalid_json(self, tmp_path):
        """Lines with invalid JSON should be skipped."""
        filepath = tmp_path / "bad.jsonl"
        good = json.dumps({
            "id": "ok",
            "question": "Q?",
            "choices": {"text": ["a", "b"], "label": ["A", "B"]},
            "answerKey": "A",
        })
        filepath.write_text(f"{{bad json\n{good}\n")

        questions = load_dataset(str(filepath))
        assert len(questions) == 1
        assert questions[0].id == "ok"

    def test_skips_missing_fields(self, tmp_path):
        """Lines missing required fields should be skipped."""
        filepath = tmp_path / "missing.jsonl"
        filepath.write_text(json.dumps({"id": "no_question"}) + "\n")

        questions = load_dataset(str(filepath))
        assert len(questions) == 0


# ---------------------------------------------------------------------------
# load_all_datasets
# ---------------------------------------------------------------------------

class TestLoadAllDatasets:
    """Tests for load_all_datasets."""

    def test_loads_all_splits(self):
        datasets = load_all_datasets()
        assert "train" in datasets
        assert "validation" in datasets
        assert "test" in datasets
        assert len(datasets["train"]) > 0

    def test_missing_dir_returns_empty(self, tmp_path):
        datasets = load_all_datasets(str(tmp_path / "nonexistent"))
        assert datasets["train"] == []
        assert datasets["validation"] == []
        assert datasets["test"] == []
