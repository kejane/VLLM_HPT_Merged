"""Mini-exam sampler for hyperparameter tuning rounds.

This module provides functionality to randomly sample a subset of questions
from the training set to create a "mini-exam" for each tuning round.
"""

import random
from typing import List, Optional

import structlog

from vllm_hpt.exam.loader import Question

logger = structlog.get_logger(__name__)


def sample_mini_exam(
    questions: List[Question],
    n: int = 200,
    seed: Optional[int] = None,
) -> List[Question]:
    """Sample N questions randomly from the question list.
    
    Args:
        questions: List of Question objects to sample from
        n: Number of questions to sample (default: 200)
        seed: Random seed for reproducibility (default: None)
    
    Returns:
        List of sampled Question objects
    """
    if seed is not None:
        random.seed(seed)
    
    total_questions = len(questions)
    
    if n >= total_questions:
        logger.warning(
            "Requested sample size >= total questions, returning all",
            requested=n,
            total=total_questions,
            seed=seed,
        )
        return questions
    
    sampled = random.sample(questions, n)
    
    logger.info(
        "Sampled mini-exam questions",
        requested=n,
        returned=len(sampled),
        total=total_questions,
        seed=seed,
    )
    
    return sampled
