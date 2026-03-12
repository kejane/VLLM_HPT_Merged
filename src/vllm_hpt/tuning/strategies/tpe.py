"""TPE (Tree-structured Parzen Estimator) search strategy."""

from typing import Optional

import optuna

from vllm_hpt.tuning.params import SamplingParams
from vllm_hpt.tuning.strategies.base import SearchStrategy
from vllm_hpt.utils.logger import get_logger

logger = get_logger(__name__)


class TPEStrategy(SearchStrategy):
    """Search strategy using Optuna's TPE sampler."""

    def __init__(
        self,
        study_name: str,
        storage_path: str,
        seed: Optional[int] = None,
    ):
        self._current_trial: Optional[optuna.trial.Trial] = None
        super().__init__(study_name=study_name, storage_path=storage_path, seed=seed)

    def _create_sampler(self) -> optuna.samplers.BaseSampler:
        return optuna.samplers.TPESampler(
            n_startup_trials=10,
            multivariate=True,
            seed=self._seed,
        )

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
