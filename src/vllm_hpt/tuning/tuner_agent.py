"""Tuner agent that uses an LLM to suggest new sampling parameters.

Analyzes evaluation results and optimization history to intelligently
propose parameter adjustments. Falls back to random perturbation on failure.
"""

import json
import re
from typing import Optional

from vllm_hpt.evaluation.evaluator import EvaluationResult, format_wrong_questions_for_agent
from vllm_hpt.model.client import LLMClient
from vllm_hpt.tuning.history import OptimizationHistory
from vllm_hpt.tuning.params import PARAM_RANGES, SamplingParams, random_perturbation
from vllm_hpt.utils.logger import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = (
    "You are a hyperparameter tuning expert specializing in sampling parameter optimization "
    "for autoregressive large language models (LLMs).\n\n"
    "## Background: How LLM Text Generation Works\n\n"
    "LLMs generate text token-by-token. At each step the model produces a probability distribution "
    "over its entire vocabulary. Sampling parameters control HOW the next token is selected from "
    "this distribution, directly affecting answer quality, consistency, and diversity.\n\n"
    "The generation pipeline at each step is:\n"
    "1. Model outputs raw logits for every token in the vocabulary.\n"
    "2. repetition_penalty modifies logits of previously generated tokens (multiplicative penalty).\n"
    "3. temperature scales the logits: logit_i / temperature. This reshapes the probability curve.\n"
    "4. top_k filters to keep only the K highest-probability tokens (hard cutoff).\n"
    "5. top_p (nucleus sampling) filters to keep the smallest set of tokens whose cumulative "
    "probability >= top_p (soft, adaptive cutoff).\n"
    "6. A token is randomly sampled from the remaining filtered distribution.\n"
    "7. Generation stops when max_tokens is reached or an end-of-sequence token is produced.\n\n"
    "## Your Task\n\n"
    "You are tuning these sampling parameters for a model answering multiple-choice science questions "
    "(ARC-Challenge dataset, choices A/B/C/D). The goal is to MAXIMIZE answer accuracy. "
    "Analyze the evaluation results, wrong question patterns, and optimization history provided, "
    "then suggest improved parameter values.\n\n"
    "IMPORTANT: Respond with ONLY a valid JSON object. No explanation text outside the JSON."
)


class TunerAgent:
    """LLM-powered agent that suggests new sampling parameters.

    Uses an LLM to analyze evaluation results, optimization history,
    and wrong question samples to propose better parameter configurations.
    Falls back to random perturbation if the LLM response cannot be parsed.
    """

    def __init__(
        self,
        client: LLMClient,
        max_retries: int = 3,
        wrong_sample_size: int = 5,
        output_truncate_length: int = 500,
    ) -> None:
        """Initialize the tuner agent.

        Args:
            client: LLMClient configured for the tuner model.
            max_retries: Maximum parse/retry attempts before falling back.
            wrong_sample_size: Max wrong questions to include in tuner prompt.
            output_truncate_length: Max chars of model output per wrong question.
        """
        self.client = client
        self.max_retries = max_retries
        self.wrong_sample_size = wrong_sample_size
        self.output_truncate_length = output_truncate_length
        self.last_prompt: Optional[str] = None
        self.last_response: Optional[str] = None
        logger.info(
            "tuner_agent_initialized",
            model=client.model,
            max_retries=max_retries,
            wrong_sample_size=wrong_sample_size,
            output_truncate_length=output_truncate_length,
        )

    def _build_prompt(
        self,
        current_params: SamplingParams,
        eval_result: EvaluationResult,
        history: OptimizationHistory,
    ) -> str:
        """Build the tuner prompt with all context sections.

        Args:
            current_params: Current sampling parameters.
            eval_result: Evaluation results from the current round.
            history: Full optimization history (train accuracy only).

        Returns:
            Formatted prompt string for the tuner LLM.
        """
        sections: list[str] = []

        # Section 1: Parameter constraints with detailed explanations
        current_dict = current_params.model_dump()

        param_descriptions = {
            "temperature": (
                "Controls randomness by scaling logits before softmax. "
                "logit_i_scaled = logit_i / temperature. "
                "Lower values (→0) sharpen the distribution making the model more deterministic and confident "
                "(greedy at 0). Higher values flatten the distribution increasing randomness and diversity. "
                "For factual multiple-choice questions, lower temperature (0.0-0.5) typically improves accuracy "
                "by favoring the model's most confident answer. Too high causes random guessing."
            ),
            "top_p": (
                "Nucleus sampling: keeps the smallest set of tokens whose cumulative probability >= top_p, "
                "then re-normalizes. Acts as an adaptive cutoff — when the model is confident (one token "
                "dominates), few tokens pass; when uncertain, more tokens pass. "
                "Lower top_p (e.g. 0.5) restricts to high-confidence tokens only. "
                "Higher top_p (e.g. 0.95) allows more diversity. "
                "For multiple-choice, moderate-to-low top_p helps the model commit to its best answer."
            ),
            "top_k": (
                "Hard cutoff: keeps only the top K highest-probability tokens, discards the rest. "
                "-1 disables top_k filtering entirely (all tokens remain candidates). "
                "Small K (e.g. 5-10) forces the model to choose among very few candidates. "
                "Large K (e.g. 50-100) is more permissive. "
                "For short, structured answers like A/B/C/D, a smaller top_k can reduce noise."
            ),
            "repetition_penalty": (
                "Penalizes tokens that have already appeared in the generated text. "
                "Applied as: logit_i = logit_i / penalty (if logit_i > 0) or logit_i * penalty (if < 0). "
                "1.0 = no penalty. Values > 1.0 discourage repetition. "
                "For short multiple-choice answers, repetition is rarely an issue, so values near 1.0 are typical. "
                "If the model produces repetitive filler text before answering, increase slightly (1.05-1.2)."
            ),
            "max_tokens": (
                "Maximum number of tokens the model can generate in its response. "
                "If the answer is cut off before the model states its choice, accuracy drops to 0 for that question. "
                "Multiple-choice answers are short (typically 1-50 tokens), but some models produce chain-of-thought "
                "reasoning before the answer. Set high enough to avoid truncation (256-1024), "
                "but not excessively high to waste compute."
            ),
        }

        constraints_lines = ["## Sampling Parameters (current values and valid ranges)", ""]
        for param_name, (min_val, max_val) in PARAM_RANGES.items():
            current_val = current_dict[param_name]
            desc = param_descriptions.get(param_name, "")
            constraints_lines.append(
                f"### {param_name}\n"
                f"- Current value: {current_val}\n"
                f"- Valid range: [{min_val}, {max_val}]\n"
                f"- Effect: {desc}"
            )
        sections.append("\n\n".join(constraints_lines))

        # Section 2: Optimization history
        history_text = history.format_for_prompt(top_k=5)
        sections.append(f"## Optimization History\n\n{history_text}")

        # Section 3: Current round results
        results_lines = [
            "## Current Round Results",
            "",
            f"- Train Accuracy: {eval_result.accuracy:.4f} ({eval_result.correct_count}/{eval_result.total_count})",
            f"- Parse Failures: {eval_result.parse_failure_count}",
        ]
        sections.append("\n".join(results_lines))

        # Section 4: Sample wrong questions
        if eval_result.wrong_questions:
            wrong_text = format_wrong_questions_for_agent(
                eval_result.wrong_questions,
                max_questions=self.wrong_sample_size,
                output_truncate_length=self.output_truncate_length,
            )
            sections.append(f"## Sample Wrong Questions\n\n{wrong_text}")

        # Section 5: Task instruction with sampling-aware guidance
        task_instruction = (
            "## Analysis & Decision Guide\n\n"
            "Examine the wrong answers and optimization history above. Diagnose the failure mode, "
            "then adjust parameters accordingly:\n\n"
            "### Diagnosis → Action Map\n\n"
            "**Symptom: Model gives random/inconsistent answers across similar questions**\n"
            "- Root cause: Distribution too flat, sampling introduces too much noise.\n"
            "- Action: LOWER temperature (toward 0.0-0.3), LOWER top_p (toward 0.5-0.8), LOWER top_k.\n\n"
            "**Symptom: Model always picks the same wrong answer (e.g., always 'A')**\n"
            "- Root cause: Distribution too peaked on a biased token, or model is overconfident.\n"
            "- Action: SLIGHTLY INCREASE temperature (0.3-0.7), INCREASE top_p to allow alternatives.\n\n"
            "**Symptom: Many parse failures (model output doesn't contain A/B/C/D)**\n"
            "- Root cause: Answer truncated (max_tokens too low) or model generates irrelevant text.\n"
            "- Action: INCREASE max_tokens. If model rambles, LOWER temperature to make it more direct.\n\n"
            "**Symptom: Model repeats phrases or loops before answering**\n"
            "- Root cause: Repetition in generation loop.\n"
            "- Action: INCREASE repetition_penalty (1.05-1.2).\n\n"
            "**Symptom: Accuracy plateaued, small changes don't help**\n"
            "- Root cause: Local optimum in parameter space.\n"
            "- Action: Try a LARGER change in temperature or top_p to explore a different region.\n\n"
            "### Key Principles\n"
            "- For factual multiple-choice, LOWER randomness generally helps (low temperature + low top_p).\n"
            "- temperature and top_p interact: both low = very deterministic; avoid setting both to extreme values.\n"
            "- Only change 1-2 parameters at a time to isolate effects.\n"
            "- Check if the best historical accuracy used similar parameter ranges — converge toward what works.\n\n"
            "Based on your analysis, suggest new parameter values that you believe will improve accuracy."
        )
        sections.append(task_instruction)

        # Section 6: Output format
        output_format = (
            "## Output Format\n\n"
            "Respond with ONLY a valid JSON object containing the new parameter values. "
            "Do not include any explanation or text outside the JSON.\n\n"
            "Example:\n"
            "```json\n"
            '{"temperature": 0.5, "top_p": 0.85, "top_k": 40, '
            '"repetition_penalty": 1.1, "max_tokens": 512}\n'
            "```"
        )
        sections.append(output_format)

        return "\n\n".join(sections)

    def _parse_response(self, response: str) -> Optional[SamplingParams]:
        """Parse LLM response into SamplingParams.

        Handles responses with or without markdown code blocks.

        Args:
            response: Raw LLM response text.

        Returns:
            Parsed SamplingParams or None if parsing fails.
        """
        # Try to extract JSON from markdown code blocks first
        code_block_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", response, re.DOTALL)
        if code_block_match:
            json_str = code_block_match.group(1).strip()
        else:
            # Try to find raw JSON object
            json_match = re.search(r"\{[^{}]*\}", response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0).strip()
            else:
                logger.warning("parse_no_json_found", response_preview=response[:200])
                return None

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(
                "parse_json_decode_error",
                error=str(e),
                json_str_preview=json_str[:200],
            )
            return None

        if not isinstance(data, dict):
            logger.warning("parse_not_dict", data_type=type(data).__name__)
            return None

        # Filter to only known parameter keys
        known_keys = set(PARAM_RANGES.keys())
        filtered = {k: v for k, v in data.items() if k in known_keys}

        if not filtered:
            logger.warning("parse_no_valid_params", keys=list(data.keys()))
            return None

        try:
            # SamplingParams validators will auto-clamp values
            params = SamplingParams(**filtered)
            return params
        except Exception as e:
            logger.warning("parse_params_creation_error", error=str(e))
            return None

    async def suggest_params(
        self,
        current_params: SamplingParams,
        eval_result: EvaluationResult,
        history: OptimizationHistory,
    ) -> SamplingParams:
        """Suggest new sampling parameters based on evaluation results.

        Builds a prompt with current context, sends it to the tuner LLM,
        and parses the response. Retries on parse failure, and falls back
        to random perturbation if all retries are exhausted.

        Args:
            current_params: Current sampling parameters.
            eval_result: Evaluation results from the current round.
            history: Full optimization history (train accuracy only visible to tuner).

        Returns:
            New SamplingParams (from LLM suggestion or random perturbation).
        """
        prompt = self._build_prompt(
            current_params, eval_result, history
        )
        self.last_prompt = prompt
        self.last_response = None

        logger.info(
            "tuner_prompt_built",
            prompt_length=len(prompt),
            current_accuracy=eval_result.accuracy,
            current_params=current_params.model_dump(),
            prompt_content=prompt,
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        for attempt in range(1, self.max_retries + 1):
            try:
                response = await self.client.chat(
                    messages=messages,
                    sampling_params={"temperature": 0.7, "max_tokens": 2048},
                )

                logger.info(
                    "tuner_response_received",
                    attempt=attempt,
                    response_length=len(response),
                    response_full=response,
                )
                self.last_response = response

                parsed = self._parse_response(response)
                if parsed is not None:
                    logger.info(
                        "tuner_params_suggested",
                        attempt=attempt,
                        params=parsed.model_dump(),
                    )
                    return parsed

                logger.warning(
                    "tuner_parse_failed",
                    attempt=attempt,
                    max_retries=self.max_retries,
                )

            except Exception as e:
                logger.error(
                    "tuner_api_error",
                    attempt=attempt,
                    error=str(e),
                )

        # All retries exhausted — fall back to random perturbation
        logger.warning(
            "tuner_fallback_to_random",
            max_retries=self.max_retries,
            current_params=current_params.model_dump(),
        )
        fallback_params = random_perturbation(current_params)
        logger.info(
            "tuner_fallback_params",
            params=fallback_params.model_dump(),
        )
        return fallback_params
