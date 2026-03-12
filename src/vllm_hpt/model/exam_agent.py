"""Exam agent for answering multiple choice questions using LLM."""

import asyncio
from typing import List, Optional, Callable, Coroutine, Any

from vllm_hpt.exam.loader import Question
from vllm_hpt.model.client import LLMClient
from vllm_hpt.tuning.params import SamplingParams
from vllm_hpt.utils.logger import get_logger

logger = get_logger(__name__)


ProgressCallback = Callable[[int, int], Coroutine[Any, Any, None]]


class ExamAgent:
    """Agent that uses LLM to answer exam questions with batch processing."""

    def __init__(self, client: LLMClient, concurrency: int = 5):
        """Initialize the exam agent.

        Args:
            client: LLMClient instance for making API calls
            concurrency: Maximum number of concurrent API calls (default: 5)
        """
        self.client = client
        self.concurrency = concurrency
        logger.info(
            "exam_agent_initialized",
            concurrency=concurrency,
        )

    def _build_prompt(self, question: Question) -> str:
        """Build prompt for a multiple choice question.

        Args:
            question: Question object with question text and choices

        Returns:
            Formatted prompt string
        """
        # Format choices with newlines
        choices_text = "\n".join(question.choices)

        prompt = f"""Answer the following multiple choice question by selecting the best answer.

Question: {question.question}
Choices:
{choices_text}

Output requirements:
1. The first line must be exactly in the form: The best answer is: X
2. Replace X with exactly one of A, B, C, or D
3. Do not wrap the answer letter in markdown, bold, brackets, or LaTeX
4. Do not write "Choice A" or similar on the first line
5. You may provide a brief explanation only after the first line

The best answer is:"""

        return prompt

    async def answer_question(
        self, question: Question, params: SamplingParams
    ) -> str:
        """Answer a single question using the LLM.

        Args:
            question: Question object to answer
            params: Sampling parameters for the LLM

        Returns:
            Raw model output (answer text)
        """
        try:
            prompt = self._build_prompt(question)
            messages = [{"role": "user", "content": prompt}]

            # Call LLM with sampling params
            response = await self.client.chat(
                messages=messages,
                sampling_params=params.to_api_dict(),
            )

            if not response:
                logger.warning(
                    "question_answer_empty",
                    question_id=question.id,
                )
                return ""

            logger.debug(
                "question_answered",
                question_id=question.id,
                response_length=len(response),
            )

            return response

        except Exception as e:
            logger.error(
                "question_answer_failed",
                question_id=question.id,
                error=str(e),
            )
            return ""

    async def answer_questions(
        self, 
        questions: List[Question], 
        params: SamplingParams,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> List[str]:
        """Answer multiple questions with concurrency control.

        Args:
            questions: List of Question objects to answer
            params: Sampling parameters for the LLM
            progress_callback: Optional async callback(current, total) for progress updates

        Returns:
            List of raw model outputs (one per question, in same order)
        """
        total = len(questions)
        completed = 0
        logger.info(
            "batch_answering_started",
            total_questions=total,
            concurrency=self.concurrency,
        )

        semaphore = asyncio.Semaphore(self.concurrency)

        async def answer_with_semaphore(i: int, q: Question) -> str:
            nonlocal completed
            async with semaphore:
                logger.debug(
                    "answering_question",
                    progress=f"{i + 1}/{total}",
                    question_id=q.id,
                )
                result = await self.answer_question(q, params)
                completed += 1
                if progress_callback:
                    await progress_callback(completed, total)
                return result

        tasks = [
            answer_with_semaphore(i, question)
            for i, question in enumerate(questions)
        ]

        answers = await asyncio.gather(*tasks)

        logger.info(
            "batch_answering_completed",
            total_questions=total,
            successful=sum(1 for a in answers if a),
            failed=sum(1 for a in answers if not a),
        )

        return answers
