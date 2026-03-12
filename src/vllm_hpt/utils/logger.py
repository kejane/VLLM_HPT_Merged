"""Logging configuration using structlog for JSON-formatted logs.

IMPORTANT: structlog is configured at module load time to use stdlib LoggerFactory.
This ensures that loggers created before setup_logging() is called will still
route through stdlib logging and ultimately to the file handlers.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import structlog

_current_log_file: Optional[str] = None
_conversation_log_file: Optional[str] = None

# Configure structlog at module load time with stdlib factory.
# This ensures ALL loggers (even those created before setup_logging()) 
# will route through stdlib logging infrastructure.
# The actual file handlers are added later by setup_logging().
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_logger_name,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer(),
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=False,
)


def setup_logging(log_level: str = "INFO", run_id: Optional[str] = None) -> str:
    """Configure stdlib logging handlers for file output.
    
    This function sets up the file handlers for stdlib logging.
    structlog is already configured at module load time to use stdlib factory,
    so this just needs to configure where stdlib logs go.
    
    Args:
        log_level: Logging level (default: INFO)
        run_id: Optional run ID for log filename. If provided, logs to logs/run_{run_id}.log
        
    Returns:
        Path to the log file being used.
    """
    global _current_log_file, _conversation_log_file
    
    if run_id:
        log_file = f"logs/run_{run_id}.log"
        _conversation_log_file = f"logs/conversation_{run_id}.log"
    else:
        log_file = "logs/vllm_hpt.log"
        _conversation_log_file = "logs/conversation.log"
    
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _current_log_file = log_file

    for noisy_logger in ("httpx", "openai", "httpcore", "urllib3", "aiohttp", "asyncio"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    
    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, log_level.upper()),
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, mode="a"),
        ],
        force=True,
    )
    
    return log_file


def get_logger(module_name: str) -> structlog.stdlib.BoundLogger:
    """Get a logger instance bound to a module name.
    
    The returned logger will route through stdlib logging infrastructure,
    which means logs will go to whatever handlers are configured
    (including file handlers set up by setup_logging()).
    """
    return structlog.get_logger(module_name).bind(module=module_name)


def log_conversation(
    round_num: int,
    timestamp: datetime,
    prompt_to_tuner: str,
    tuner_response: str,
    current_params: dict,
    suggested_params: dict,
    accuracy: float,
    wrong_questions: list,
    history_summary: str,
    duration_ms: float,
) -> None:
    """Write human-readable conversation log to separate file.
    
    This creates a formatted, easy-to-read log of the agent dialogue
    for debugging and understanding the tuning process.
    """
    if _conversation_log_file is None:
        return
    
    separator = "=" * 80
    section_sep = "-" * 40
    
    lines = [
        separator,
        f"ROUND {round_num} | {timestamp.isoformat()}",
        separator,
        "",
        f"[ACCURACY] {accuracy:.2%}",
        f"[DURATION] {duration_ms:.0f}ms",
        "",
        section_sep,
        "CURRENT PARAMS",
        section_sep,
    ]
    
    for k, v in current_params.items():
        lines.append(f"  {k}: {v}")
    
    lines.extend([
        "",
        section_sep,
        f"WRONG QUESTIONS ({len(wrong_questions)} total)",
        section_sep,
    ])
    
    for wq in wrong_questions[:10]:
        lines.append(f"  [{wq.get('id', 'N/A')}] model={wq.get('model', '?')} correct={wq.get('correct', '?')}")
    if len(wrong_questions) > 10:
        lines.append(f"  ... and {len(wrong_questions) - 10} more")
    
    lines.extend([
        "",
        section_sep,
        "HISTORY SUMMARY",
        section_sep,
        history_summary,
        "",
        section_sep,
        ">>> MESSAGE TO TUNER AGENT",
        section_sep,
        prompt_to_tuner,
        "",
        section_sep,
        "<<< TUNER AGENT RESPONSE",
        section_sep,
        tuner_response,
        "",
        section_sep,
        "SUGGESTED PARAMS",
        section_sep,
    ])
    
    for k, v in suggested_params.items():
        lines.append(f"  {k}: {v}")
    
    lines.extend(["", ""])
    
    with open(_conversation_log_file, "a", encoding="utf-8") as f:
        f.write("\n".join(lines))
