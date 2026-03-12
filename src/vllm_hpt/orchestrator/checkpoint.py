"""Checkpoint management for saving and restoring tuning state."""

import json
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from vllm_hpt.tuning.history import OptimizationHistory
from vllm_hpt.tuning.params import SamplingParams
from vllm_hpt.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Checkpoint:
    """Checkpoint data for resuming tuning runs."""

    run_id: str
    current_round: int
    total_rounds: int
    history: OptimizationHistory
    best_params: SamplingParams
    best_validation_accuracy: float
    tuning_mode: str = "tpe"
    strategy_name: str = "tpe"
    random_state: Optional[Any] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())


class CheckpointManager:
    """Manages checkpoint saving and loading."""

    def save(self, checkpoint: Checkpoint, filepath: str) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.updated_at = datetime.now().isoformat()

        data = {
            "run_id": checkpoint.run_id,
            "current_round": checkpoint.current_round,
            "total_rounds": checkpoint.total_rounds,
            "history": checkpoint.history.to_dict(),
            "best_params": checkpoint.best_params.model_dump(),
            "best_validation_accuracy": checkpoint.best_validation_accuracy,
            "tuning_mode": checkpoint.tuning_mode,
            "strategy_name": checkpoint.strategy_name,
            "random_state": checkpoint.random_state,
            "created_at": checkpoint.created_at,
            "updated_at": checkpoint.updated_at,
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(
            "checkpoint_saved",
            filepath=filepath,
            run_id=checkpoint.run_id,
            current_round=checkpoint.current_round,
        )

    def load(self, filepath: str) -> Checkpoint:
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

        with open(filepath, "r") as f:
            data = json.load(f)

        try:
            # Backwards-compatible: old checkpoints may only have strategy_name
            tuning_mode = data.get("tuning_mode", data.get("strategy_name", "tpe"))
            strategy_name = data.get(
                "strategy_name", tuning_mode if tuning_mode != "a2a" else "tpe"
            )

            checkpoint = Checkpoint(
                run_id=data["run_id"],
                current_round=data["current_round"],
                total_rounds=data["total_rounds"],
                history=OptimizationHistory.from_dict(data["history"]),
                best_params=SamplingParams(**data["best_params"]),
                best_validation_accuracy=data["best_validation_accuracy"],
                tuning_mode=tuning_mode,
                strategy_name=strategy_name,
                random_state=data.get("random_state"),
                created_at=data.get("created_at", datetime.now().isoformat()),
                updated_at=data.get("updated_at", datetime.now().isoformat()),
            )

            # Random state is restored by runner.resume() after all setup
            # completes, not here, to avoid applying it twice.

            logger.info(
                "checkpoint_loaded",
                filepath=filepath,
                run_id=checkpoint.run_id,
                current_round=checkpoint.current_round,
            )
            return checkpoint

        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"Invalid checkpoint data in {filepath}: {e}")

    def auto_save_path(self, run_id: str) -> str:
        return f"checkpoints/run_{run_id}.json"

    def study_path(self, run_id: str) -> str:
        return f"checkpoints/{run_id}_study.db"

    def find_latest(self, checkpoint_dir: str = "checkpoints") -> Optional[str]:
        path = Path(checkpoint_dir)
        if not path.exists():
            return None

        checkpoint_files = list(path.glob("*.json"))
        if not checkpoint_files:
            return None

        checkpoint_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        latest = str(checkpoint_files[0])
        logger.info("latest_checkpoint_found", filepath=latest)
        return latest
