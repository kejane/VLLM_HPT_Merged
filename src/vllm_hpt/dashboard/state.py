"""Real-time state management for dashboard WebSocket broadcasting."""

import asyncio
import json
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

_dashboard_state: Optional["DashboardState"] = None


@dataclass
class ModelStatus:
    name: str
    status: str = "unknown"  # "ok", "error", "unknown"
    last_check: Optional[str] = None
    error_message: Optional[str] = None


@dataclass 
class DashboardState:
    mode: str = "idle"  # "idle", "tuning", "evaluating", "validating", "waiting_params", "eval_done"
    
    current_round: int = 0
    total_rounds: int = 0
    
    current_question: int = 0
    total_questions: int = 0
    
    current_params: dict = field(default_factory=dict)
    
    accuracy_history: list[dict] = field(default_factory=list)
    
    # Evaluate-specific fields
    eval_result: Optional[dict] = None  # {accuracy, correct, total, parse_failures, wrong_sample}
    eval_result_file: Optional[str] = None  # Path to saved result file
    
    exam_model: ModelStatus = field(default_factory=lambda: ModelStatus(name="exam_agent"))
    
    strategy_name: Optional[str] = None
    trial_number: Optional[int] = None
    optuna_study: Optional[Any] = field(default=None, repr=False)  # optuna.study.Study reference
    
    last_update: Optional[str] = None
    
    _subscribers: set = field(default_factory=set, repr=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    _params_event: threading.Event = field(default_factory=threading.Event, repr=False)
    _pending_params: Optional[dict] = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "current_round": self.current_round,
            "total_rounds": self.total_rounds,
            "current_question": self.current_question,
            "total_questions": self.total_questions,
            "current_params": self.current_params,
            "accuracy_history": self.accuracy_history,
            "eval_result": self.eval_result,
            "eval_result_file": self.eval_result_file,
            "exam_model": {
                "name": self.exam_model.name,
                "status": self.exam_model.status,
                "last_check": self.exam_model.last_check,
                "error_message": self.exam_model.error_message,
            },
            "strategy_name": self.strategy_name,
            "trial_number": self.trial_number,
            "last_update": self.last_update,
        }

    async def subscribe(self, websocket) -> None:
        async with self._lock:
            self._subscribers.add(websocket)

    async def unsubscribe(self, websocket) -> None:
        async with self._lock:
            self._subscribers.discard(websocket)

    async def broadcast(self) -> None:
        self.last_update = datetime.now().isoformat()
        message = json.dumps(self.to_dict())
        async with self._lock:
            dead = set()
            for ws in self._subscribers:
                try:
                    await ws.send_text(message)
                except Exception:
                    dead.add(ws)
            self._subscribers -= dead

    async def update(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if hasattr(self, key) and not key.startswith("_"):
                setattr(self, key, value)
        await self.broadcast()

    def submit_params(self, params: dict) -> None:
        self._pending_params = params
        self._params_event.set()

    async def wait_for_params(self) -> dict:
        while not self._params_event.is_set():
            await asyncio.sleep(0.2)
        self._params_event.clear()
        params = self._pending_params or {}
        self._pending_params = None
        return params

    def reset(self) -> None:
        self.mode = "idle"
        self.current_round = 0
        self.total_rounds = 0
        self.current_question = 0
        self.total_questions = 0
        self.current_params = {}
        self.accuracy_history = []
        self.eval_result = None
        self.eval_result_file = None
        self.exam_model = ModelStatus(name="exam_agent")
        self.strategy_name = None
        self.trial_number = None
        self.optuna_study = None


def get_dashboard_state() -> DashboardState:
    global _dashboard_state
    if _dashboard_state is None:
        _dashboard_state = DashboardState()
    return _dashboard_state
