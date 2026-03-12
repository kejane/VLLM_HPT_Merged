"""Optimization history tracking for parameter tuning."""

from dataclasses import dataclass
from typing import Any, List, Optional, TYPE_CHECKING

from vllm_hpt.tuning.params import SamplingParams
from vllm_hpt.utils.logger import get_logger

if TYPE_CHECKING:
    import optuna

logger = get_logger(__name__)


@dataclass
class HistoryEntry:
    """Single optimization history entry."""

    round_num: int
    params: SamplingParams
    train_accuracy: float
    validation_accuracy: Optional[float] = None


class OptimizationHistory:
    """Tracks parameter-score pairs across optimization rounds."""

    def __init__(self) -> None:
        self.entries: List[HistoryEntry] = []

    def add_entry(self, entry: HistoryEntry) -> None:
        """Add new history entry."""
        self.entries.append(entry)
        logger.info(
            "history_entry_added",
            round_num=entry.round_num,
            train_accuracy=entry.train_accuracy,
            validation_accuracy=entry.validation_accuracy,
        )

    def get_best(self, by: str = "validation") -> Optional[HistoryEntry]:
        """
        Get best entry by accuracy metric.

        Args:
            by: Metric to use ("validation" or "train").

        Returns:
            Best entry or None if no entries.
        """
        if not self.entries:
            return None

        if by == "validation":
            valid_entries = [e for e in self.entries if e.validation_accuracy is not None]
            if not valid_entries:
                return None
            return max(valid_entries, key=lambda e: e.validation_accuracy or 0.0)
        else:
            return max(self.entries, key=lambda e: e.train_accuracy)

    def get_top_k(self, k: int = 5, by: str = "validation") -> List[HistoryEntry]:
        """
        Get top-k entries sorted by accuracy.

        Args:
            k: Number of entries to return.
            by: Metric to use ("validation" or "train").

        Returns:
            List of top-k entries, sorted descending by accuracy.
        """
        if not self.entries:
            return []

        if by == "validation":
            valid_entries = [e for e in self.entries if e.validation_accuracy is not None]
            sorted_entries = sorted(
                valid_entries, key=lambda e: e.validation_accuracy or 0.0, reverse=True
            )
        else:
            sorted_entries = sorted(
                self.entries, key=lambda e: e.train_accuracy, reverse=True
            )

        return sorted_entries[:k]

    def format_for_prompt(self, top_k: int = 5) -> str:
        """Format history for tuner agent prompt (train accuracy only, no validation)."""
        if not self.entries:
            return "No optimization history available yet."

        sorted_entries = sorted(
            self.entries, key=lambda e: e.train_accuracy, reverse=True
        )
        top_entries = sorted_entries[:top_k]

        lines = []
        for entry in top_entries:
            params_dict = entry.params.model_dump()
            params_str = ", ".join(f"{k}: {v}" for k, v in params_dict.items())
            lines.append(
                f"Round {entry.round_num}: train_accuracy={entry.train_accuracy:.4f}, params={{{params_str}}}"
            )

        return "\n".join(lines)

    def to_dict(self) -> List[dict]:
        """
        Serialize history to dict for checkpoint saving.

        Returns:
            List of entry dictionaries.
        """
        return [
            {
                "round_num": e.round_num,
                "params": e.params.model_dump(),
                "train_accuracy": e.train_accuracy,
                "validation_accuracy": e.validation_accuracy,
            }
            for e in self.entries
        ]

    @classmethod
    def from_dict(cls, data: List[dict]) -> "OptimizationHistory":
        """
        Deserialize history from checkpoint data.

        Args:
            data: List of entry dictionaries.

        Returns:
            OptimizationHistory instance.
        """
        history = cls()
        for entry_dict in data:
            entry = HistoryEntry(
                round_num=entry_dict["round_num"],
                params=SamplingParams(**entry_dict["params"]),
                train_accuracy=entry_dict["train_accuracy"],
                validation_accuracy=entry_dict.get("validation_accuracy"),
            )
            history.entries.append(entry)

        logger.info("history_restored", num_entries=len(history.entries))
        return history

    @classmethod
    def from_study(cls, study: "optuna.study.Study") -> "OptimizationHistory":
        """
        Reconstruct history from Optuna study's completed trials.

        Args:
            study: Optuna study object with completed trials.

        Returns:
            OptimizationHistory instance with entries from completed trials.
        """
        import optuna

        history = cls()
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                entry = HistoryEntry(
                    round_num=trial.number + 1,  # Optuna is 0-indexed, rounds are 1-indexed
                    params=SamplingParams(**trial.params),
                    train_accuracy=trial.value,
                    validation_accuracy=None,  # Optuna doesn't track this separately
                )
                history.entries.append(entry)

        logger.info("history_from_study", num_entries=len(history.entries))
        return history
