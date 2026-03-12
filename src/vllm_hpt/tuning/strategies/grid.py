"""Grid search strategy with exhaustive parameter space exploration."""

import math
from typing import Optional

import optuna

from vllm_hpt.tuning.params import PARAM_RANGES, SamplingParams
from vllm_hpt.tuning.strategies.base import SearchStrategy
from vllm_hpt.utils.logger import get_logger

logger = get_logger(__name__)


class GridSearchStrategy(SearchStrategy):
    """Search strategy using Optuna's GridSampler for exhaustive search."""

    def __init__(
        self,
        study_name: str,
        storage_path: str,
        seed: Optional[int] = None,
        grid_values: int = 3,
        grid_config: Optional[dict] = None,
    ):
        self._current_trial: Optional[optuna.trial.Trial] = None
        self._grid_values = grid_values
        self._grid_config = grid_config
        self._grid_search_space = self._build_grid_search_space()

        total = math.prod(len(v) for v in self._grid_search_space.values())
        logger.info("grid_total_combos", total=total)

        super().__init__(study_name=study_name, storage_path=storage_path, seed=seed)

    def _build_grid_search_space(self) -> dict:
        if self._grid_config is not None:
            return self._grid_config

        space = {}
        for name, (low, high) in PARAM_RANGES.items():
            n = self._grid_values
            if isinstance(low, int) and isinstance(high, int):
                step = max(1, (high - low) // max(1, n - 1))
                values = sorted(set(list(range(low, high + 1, step))[:n]))
                if high not in values:
                    values.append(high)
                space[name] = values
            else:
                space[name] = [round(low + i * (high - low) / max(1, n - 1), 6) for i in range(n)]
        return space

    def _create_sampler(self) -> optuna.samplers.BaseSampler:
        return optuna.samplers.GridSampler(self._grid_search_space)

    def suggest_next_params(self) -> SamplingParams:
        trial = self._study.ask()
        self._current_trial = trial
        params = SamplingParams(
            temperature=trial.suggest_float("temperature", 0.0, 2.0),
            top_p=trial.suggest_float("top_p", 0.0, 1.0),
            top_k=trial.suggest_int("top_k", -1, 100),
            repetition_penalty=trial.suggest_float("repetition_penalty", 1.0, 2.0),
            max_tokens=trial.suggest_int("max_tokens", 1, 10240),
        )
        logger.info("params_suggested", strategy=self.__class__.__name__, params=params.model_dump())
        return params

    def report_result(self, params: SamplingParams, score: float) -> None:
        if self._current_trial is None:
            logger.warning("report_result_no_trial")
            return
        self._study.tell(self._current_trial, score)
        logger.info("result_reported", score=score, trial=self._current_trial.number)
        self._current_trial = None
