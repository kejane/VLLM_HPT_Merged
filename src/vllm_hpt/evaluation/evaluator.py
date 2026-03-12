"""Evaluator module for comparing model answers against correct answers.

This module evaluates model performance by comparing parsed answers to correct
answer keys, calculating accuracy, and generating detailed wrong question reports.
"""

from dataclasses import dataclass
from typing import List, Optional

from vllm_hpt.evaluation.answer_parser import parse_answers
from vllm_hpt.exam.loader import Question
from vllm_hpt.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class WrongQuestion:
    """Details about a question the model answered incorrectly."""
    
    question_id: str
    question_text: str
    choices: List[str]
    model_answer: Optional[str]  # Parsed answer, None if parse failed
    correct_answer: str
    model_output_raw: str  # Raw model output


@dataclass
class EvaluationResult:
    """Results from evaluating model answers against correct answers."""
    
    accuracy: float
    correct_count: int
    total_count: int
    wrong_questions: List[WrongQuestion]
    parse_failure_count: int
    
    def summary(self) -> str:
        """Generate a human-readable summary of evaluation results.
        
        Returns:
            Formatted summary string with key metrics
        """
        summary_lines = [
            "=" * 60,
            "EVALUATION SUMMARY",
            "=" * 60,
            f"Total Questions: {self.total_count}",
            f"Correct Answers: {self.correct_count}",
            f"Wrong Answers: {len(self.wrong_questions)}",
            f"Parse Failures: {self.parse_failure_count}",
            f"Accuracy: {self.accuracy:.2%}",
            "=" * 60,
        ]
        
        if self.wrong_questions:
            summary_lines.append(f"\nWrong Questions ({len(self.wrong_questions)}):")
            for wq in self.wrong_questions[:10]:  # Show first 10
                model_ans = wq.model_answer if wq.model_answer else "PARSE_FAILED"
                summary_lines.append(
                    f"  - Q{wq.question_id}: Model={model_ans}, Correct={wq.correct_answer}"
                )
            if len(self.wrong_questions) > 10:
                summary_lines.append(f"  ... and {len(self.wrong_questions) - 10} more")
        
        return "\n".join(summary_lines)


def evaluate(questions: List[Question], raw_outputs: List[str]) -> EvaluationResult:
    """Evaluate model answers against correct answers.
    
    Args:
        questions: List of Question objects with correct answer keys
        raw_outputs: List of raw model output strings (same order as questions)
        
    Returns:
        EvaluationResult with accuracy metrics and wrong question details
        
    Raises:
        ValueError: If questions and raw_outputs have different lengths
    """
    if len(questions) != len(raw_outputs):
        raise ValueError(
            f"Mismatch: {len(questions)} questions but {len(raw_outputs)} outputs"
        )
    
    logger.info(
        "Starting evaluation",
        total_questions=len(questions)
    )
    
    # Parse all raw outputs
    parsed_answers, parse_stats = parse_answers(raw_outputs)
    
    # Compare answers and build wrong question list
    correct_count = 0
    wrong_questions = []
    
    for idx, (question, parsed_answer, raw_output) in enumerate(
        zip(questions, parsed_answers, raw_outputs)
    ):
        is_correct = parsed_answer == question.answer_key
        
        if is_correct:
            correct_count += 1
        else:
            # Record wrong question
            wrong_question = WrongQuestion(
                question_id=question.id,
                question_text=question.question,
                choices=question.choices,
                model_answer=parsed_answer,
                correct_answer=question.answer_key,
                model_output_raw=raw_output,
            )
            wrong_questions.append(wrong_question)
            
            logger.debug(
                "Wrong answer detected",
                question_id=question.id,
                model_answer=parsed_answer,
                correct_answer=question.answer_key,
            )
    
    # Calculate accuracy
    total_count = len(questions)
    accuracy = correct_count / total_count if total_count > 0 else 0.0
    
    result = EvaluationResult(
        accuracy=accuracy,
        correct_count=correct_count,
        total_count=total_count,
        wrong_questions=wrong_questions,
        parse_failure_count=parse_stats.failed,
    )
    
    logger.info(
        "Evaluation complete",
        accuracy=f"{accuracy:.2%}",
        correct=correct_count,
        wrong=len(wrong_questions),
        parse_failures=parse_stats.failed,
        total=total_count,
    )
    
    return result


def format_wrong_questions_for_agent(
    wrong_questions: List[WrongQuestion],
    max_questions: int = 5,
    output_truncate_length: int = 500,
) -> str:
    """Format wrong questions for tuner-agent prompts."""
    if not wrong_questions:
        return "No wrong questions to display."

    lines = [
        f"Wrong Questions ({len(wrong_questions)} total, showing first {max_questions}):",
        "",
    ]

    for idx, wq in enumerate(wrong_questions[:max_questions], start=1):
        model_ans = wq.model_answer if wq.model_answer else "PARSE_FAILED"
        lines.extend(
            [
                f"--- Question {idx} (ID: {wq.question_id}) ---",
                f"Question: {wq.question_text}",
                "",
                "Choices:",
            ]
        )

        for choice in wq.choices:
            lines.append(f"  {choice}")

        truncated = len(wq.model_output_raw) > output_truncate_length
        output_text = wq.model_output_raw[:output_truncate_length]

        lines.extend(
            [
                "",
                f"Model Answer: {model_ans}",
                f"Correct Answer: {wq.correct_answer}",
                "",
                "Raw Model Output:",
                f"  {output_text}{'...' if truncated else ''}",
                "",
            ]
        )

    if len(wrong_questions) > max_questions:
        lines.append(
            f"... and {len(wrong_questions) - max_questions} more wrong questions"
        )

    return "\n".join(lines)
