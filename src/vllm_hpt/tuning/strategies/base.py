"""Abstract base class for hyperparameter search strategies."""

from abc import ABC, abstractmethod
from typing import Optional

import optuna

from vllm_hpt.tuning.params import PARAM_RANGES, SamplingParams
from vllm_hpt.utils.logger import get_logger

logger = get_logger(__name__)

# Define search space using PARAM_RANGES
SEARCH_SPACE = {
    "temperature": {
        "type": "float",
        "low": PARAM_RANGES["temperature"][0],
        "high": PARAM_RANGES["temperature"][1],
    },
    "top_p": {
        "type": "float",
        "low": PARAM_RANGES["top_p"][0],
        "high": PARAM_RANGES["top_p"][1],
    },
    "top_k": {
        "type": "int",
        "low": PARAM_RANGES["top_k"][0],
        "high": PARAM_RANGES["top_k"][1],
    },
    "repetition_penalty": {
        "type": "float",
        "low": PARAM_RANGES["repetition_penalty"][0],
        "high": PARAM_RANGES["repetition_penalty"][1],
    },
    "max_tokens": {
        "type": "int",
        "low": PARAM_RANGES["max_tokens"][0],
        "high": PARAM_RANGES["max_tokens"][1],
    },
}


class SearchStrategy(ABC):
    """Abstract base class for hyperparameter search strategies using Optuna's ask-and-tell API."""

    def __init__(self, study_name: str, storage_path: str, seed: Optional[int] = None):
        """
        Initialize the search strategy.

        Args:
            study_name: Name of the Optuna study.
            storage_path: Path to SQLite database for persistence.
            seed: Random seed for reproducibility.
        """
        self._study_name = study_name
        self._storage_path = storage_path
        self._seed = seed

        # 确保上级目录存在，防止 SQLite 报 unable to open database file
        import os

        os.makedirs(os.path.dirname(storage_path), exist_ok=True)

        # Create storage URL
        storage_url = f"sqlite:///{storage_path}"

        # Initialize Optuna study
        self._study = optuna.create_study(
            study_name=study_name,
            storage=storage_url,
            direction="maximize",
            load_if_exists=True,
            sampler=self._create_sampler(),
        )

        logger.info(
            "Initialized search strategy",
            strategy=self.__class__.__name__,
            study_name=study_name,
            storage=storage_url,
            n_trials=len(self._study.trials),
        )

    @abstractmethod
    def _create_sampler(self) -> optuna.samplers.BaseSampler:
        """
        Create the Optuna sampler for this strategy.

        Returns:
            Optuna sampler instance.
        """
        pass

    @abstractmethod
    def suggest_next_params(self) -> SamplingParams:
        """
        Ask the optimizer for the next set of parameters to try.

        Returns:
            SamplingParams instance with suggested parameters.
        """
        pass

    @abstractmethod
    def report_result(self, params: SamplingParams, score: float) -> None:
        """
        Tell the optimizer the result of evaluating a parameter set.

        Args:
            params: The parameters that were evaluated.
            score: The score achieved (higher is better).
        """
        pass

    @property
    def best_params(self) -> Optional[SamplingParams]:
        """
        Get the best parameters found so far.

        Returns:
            SamplingParams instance with best parameters, or None if no trials completed.
        """
        if self.n_completed_trials() == 0:
            return None

        return SamplingParams(**self._study.best_params)

    @property
    def best_score(self) -> Optional[float]:
        """
        Get the best score achieved so far.

        Returns:
            Best score, or None if no trials completed.
        """
        if self.n_completed_trials() == 0:
            return None

        return self._study.best_value

    def n_completed_trials(self) -> int:
        """
        Get the number of completed trials.

        Returns:
            Number of completed trials.
        """
        return len(
            [
                t
                for t in self._study.trials
                if t.state == optuna.trial.TrialState.COMPLETE
            ]
        )

    @property
    def study_name(self) -> str:
        """
        Get the Optuna study name.

        Returns:
            Study name.
        """
        return self._study_name
