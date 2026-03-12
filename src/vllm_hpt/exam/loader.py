"""Exam dataset loader for ARC-Challenge JSONL format."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from vllm_hpt.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Question:
    """Structured representation of an exam question."""

    id: str
    question: str
    choices: List[str]  # Formatted as ["A. xxx", "B. xxx", ...]
    answer_key: str  # Normalized to A/B/C/D
    original_label: str  # Original label from data (for debugging)

    def __str__(self) -> str:
        """Human-readable string representation."""
        choices_str = "\n  ".join(self.choices)
        return (
            f"Question {self.id}:\n"
            f"  {self.question}\n"
            f"  {choices_str}\n"
            f"  Answer: {self.answer_key}"
        )


def _normalize_label(label: str) -> str:
    """Normalize label from "1234" format to "ABCD" format.

    Args:
        label: Original label (e.g., "1", "2", "A", "B")

    Returns:
        Normalized label in A/B/C/D format
    """
    label_map = {"1": "A", "2": "B", "3": "C", "4": "D"}
    return label_map.get(label, label)


def load_dataset(filepath: str) -> List[Question]:
    """Load a single JSONL dataset file.

    Args:
        filepath: Path to JSONL file

    Returns:
        List of Question objects

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {filepath}")

    questions = []
    skipped = 0

    logger.info(f"Loading dataset from {filepath}")

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)

                # Validate required fields
                if not all(key in data for key in ["id", "question", "choices", "answerKey"]):
                    logger.warning(
                        f"Line {line_num}: Missing required fields, skipping",
                        line_num=line_num,
                        data_keys=list(data.keys()),
                    )
                    skipped += 1
                    continue

                # Extract choices
                choices_data = data["choices"]
                if not isinstance(choices_data, dict) or "text" not in choices_data or "label" not in choices_data:
                    logger.warning(
                        f"Line {line_num}: Invalid choices format, skipping",
                        line_num=line_num,
                    )
                    skipped += 1
                    continue

                choice_texts = choices_data["text"]
                choice_labels = choices_data["label"]

                if len(choice_texts) != len(choice_labels):
                    logger.warning(
                        f"Line {line_num}: Mismatched choice texts and labels, skipping",
                        line_num=line_num,
                    )
                    skipped += 1
                    continue

                # Format choices as "A. text", "B. text", etc.
                formatted_choices = [
                    f"{_normalize_label(label)}. {text}"
                    for label, text in zip(choice_labels, choice_texts)
                ]

                # Normalize answer key
                original_answer = data["answerKey"]
                normalized_answer = _normalize_label(original_answer)

                question = Question(
                    id=data["id"],
                    question=data["question"],
                    choices=formatted_choices,
                    answer_key=normalized_answer,
                    original_label=original_answer,
                )
                questions.append(question)

            except json.JSONDecodeError as e:
                logger.warning(
                    f"Line {line_num}: JSON decode error, skipping",
                    line_num=line_num,
                    error=str(e),
                )
                skipped += 1
            except Exception as e:
                logger.warning(
                    f"Line {line_num}: Unexpected error, skipping",
                    line_num=line_num,
                    error=str(e),
                )
                skipped += 1

    logger.info(
        f"Loaded {len(questions)} questions from {filepath}",
        loaded=len(questions),
        skipped=skipped,
    )

    return questions


def load_all_datasets(data_dir: str = "data/ai2_arc/ARC-Challenge") -> Dict[str, List[Question]]:
    """Load all dataset splits (train, validation, test).

    Args:
        data_dir: Directory containing the dataset files

    Returns:
        Dictionary mapping split name to list of questions
        {"train": [...], "validation": [...], "test": [...]}
    """
    data_path = Path(data_dir)
    splits = ["train", "validation", "test"]
    datasets = {}

    logger.info(f"Loading all datasets from {data_dir}")

    for split in splits:
        filepath = data_path / f"{split}.jsonl"
        try:
            datasets[split] = load_dataset(str(filepath))
        except FileNotFoundError:
            logger.warning(f"Split '{split}' not found at {filepath}, skipping")
            datasets[split] = []

    total_questions = sum(len(qs) for qs in datasets.values())
    logger.info(
        f"Loaded all datasets: {total_questions} total questions",
        train=len(datasets.get("train", [])),
        validation=len(datasets.get("validation", [])),
        test=len(datasets.get("test", [])),
    )

    return datasets
