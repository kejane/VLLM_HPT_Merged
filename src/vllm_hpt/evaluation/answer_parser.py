"""Answer parser for extracting answer labels from model outputs.

This module provides multi-level regex-based parsing to extract answer labels
(A/B/C/D) from raw model output text with fallback strategies.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from vllm_hpt.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class AnswerParserStats:
    """Statistics for answer parsing operations."""
    
    total_parsed: int = 0
    successful: int = 0
    failed: int = 0
    level_counts: dict = field(default_factory=dict)
    
    def record_success(self, level: int) -> None:
        """Record a successful parse at the given level."""
        self.total_parsed += 1
        self.successful += 1
        self.level_counts[level] = self.level_counts.get(level, 0) + 1
    
    def record_failure(self) -> None:
        """Record a failed parse."""
        self.total_parsed += 1
        self.failed += 1


def parse_answer(raw_output: str) -> Optional[str]:
    """Parse answer label from raw model output using multi-level regex fallback.
    
    Extraction levels:
    1. Match strict final-answer forms such as "The best answer is: D"
    2. Match natural answer phrases, including markdown/Choice variants
    3. Match boxed answers such as "\\boxed{C}"
    4. Match standalone [A-D] followed by period or end of string (e.g., "A." or "B")
    5. Match first [A-D] character at the start of the output
    
    Args:
        raw_output: Raw text output from the model
        
    Returns:
        Extracted answer label (A/B/C/D) or None if parsing fails
    """
    if not raw_output:
        return None

    strict_patterns = [
        r'(?:final\s+answer|the\s+best\s+answer|the\s+correct\s+answer)\s+is\s*[:：]\s*([A-D])\b',
        r'(?:final\s+answer|the\s+best\s+answer|the\s+correct\s+answer)\s+is\s*[:：]\s*\*\*([A-D])\*\*',
    ]
    for pattern in strict_patterns:
        match = re.search(pattern, raw_output, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    
    natural_patterns = [
        r'answer\s+is\s+\*\*(?:choice\s+)?([A-D])\*\*',
        r'answer\s+is\s+(?:choice\s+)?([A-D])\b',
        r'correct\s+answer\s+is\s+\*\*(?:choice\s+)?([A-D])\*\*',
        r'correct\s+answer\s+is\s+(?:choice\s+)?([A-D])\b',
    ]
    for pattern in natural_patterns:
        match = re.search(pattern, raw_output, re.IGNORECASE)
        if match:
            return match.group(1).upper()

    match = re.search(r'\\boxed\{\s*([A-D])\s*\}', raw_output, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    match = re.search(r'\b([A-D])\.(?:\s|$)', raw_output)
    if match:
        return match.group(1).upper()
    
    match = re.match(r'^([A-D])', raw_output.strip())
    if match:
        return match.group(1).upper()
    
    return None


def _detect_parse_level(raw_output: str) -> Tuple[Optional[str], Optional[int]]:
    """Return (answer, level) for a single raw output, or (None, None) on failure."""
    strict_patterns = [
        r'(?:final\s+answer|the\s+best\s+answer|the\s+correct\s+answer)\s+is\s*[:：]\s*([A-D])\b',
        r'(?:final\s+answer|the\s+best\s+answer|the\s+correct\s+answer)\s+is\s*[:：]\s*\*\*([A-D])\*\*',
    ]
    for pattern in strict_patterns:
        m = re.search(pattern, raw_output, re.IGNORECASE)
        if m:
            return m.group(1).upper(), 1

    natural_patterns = [
        r'answer\s+is\s+\*\*(?:choice\s+)?([A-D])\*\*',
        r'answer\s+is\s+(?:choice\s+)?([A-D])\b',
        r'correct\s+answer\s+is\s+\*\*(?:choice\s+)?([A-D])\*\*',
        r'correct\s+answer\s+is\s+(?:choice\s+)?([A-D])\b',
    ]
    for pattern in natural_patterns:
        m = re.search(pattern, raw_output, re.IGNORECASE)
        if m:
            return m.group(1).upper(), 2

    m = re.search(r'\\boxed\{\s*([A-D])\s*\}', raw_output, re.IGNORECASE)
    if m:
        return m.group(1).upper(), 3

    m = re.search(r'\b([A-D])\.(?:\s|$)', raw_output)
    if m:
        return m.group(1).upper(), 4

    m = re.match(r'^([A-D])', raw_output.strip())
    if m:
        return m.group(1).upper(), 5

    return None, None


def parse_answers(raw_outputs: List[str]) -> Tuple[List[Optional[str]], AnswerParserStats]:
    """Parse multiple answer labels from raw model outputs.
    
    Args:
        raw_outputs: List of raw text outputs from the model
        
    Returns:
        Tuple of (parsed answers list, statistics)
    """
    stats = AnswerParserStats()
    parsed_answers = []

    for idx, raw_output in enumerate(raw_outputs):
        answer: Optional[str] = None
        parse_level: Optional[int] = None

        if raw_output:
            answer, parse_level = _detect_parse_level(raw_output)

        if answer is not None and parse_level is not None:
            stats.record_success(parse_level)
        else:
            stats.record_failure()
            logger.warning(
                "parse_answer_failed",
                index=idx,
                output_preview=raw_output[:100] if raw_output else None,
            )

        parsed_answers.append(answer)

    return parsed_answers, stats
