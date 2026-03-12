"""Factory function for creating search strategy instances."""

from typing import Optional

from vllm_hpt.tuning.strategies.base import SearchStrategy
from vllm_hpt.tuning.strategies.cmaes import CMAESStrategy
from vllm_hpt.tuning.strategies.gp import GPStrategy
from vllm_hpt.tuning.strategies.grid import GridSearchStrategy
from vllm_hpt.tuning.strategies.tpe import TPEStrategy
from vllm_hpt.utils.logger import get_logger

logger = get_logger(__name__)

STRATEGY_REGISTRY: dict[str, type[SearchStrategy]] = {
    "tpe": TPEStrategy,
    "gp": GPStrategy,
    "cmaes": CMAESStrategy,
    "grid": GridSearchStrategy,
}


def create_strategy(
    name: str,
    study_name: str,
    storage_path: str,
    seed: Optional[int] = None,
    **kwargs,
) -> SearchStrategy:
    if name not in STRATEGY_REGISTRY:
        available = ", ".join(STRATEGY_REGISTRY.keys()) if STRATEGY_REGISTRY else "none"
        raise ValueError(
            f"Unknown strategy: {name}. Available strategies: {available}"
        )
    
    strategy_class = STRATEGY_REGISTRY[name]
    return strategy_class(
        study_name=study_name,
        storage_path=storage_path,
        seed=seed,
        **kwargs,
    )


__all__ = ["create_strategy", "SearchStrategy", "STRATEGY_REGISTRY"]
