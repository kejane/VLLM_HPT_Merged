"""Main tuning runner that orchestrates the complete HPT workflow."""

import asyncio
import random
import signal
from datetime import datetime
from typing import Optional

from vllm_hpt.config import Settings
from vllm_hpt.dashboard.state import get_dashboard_state
from vllm_hpt.evaluation.evaluator import EvaluationResult, evaluate
from vllm_hpt.exam.loader import Question, load_all_datasets
from vllm_hpt.exam.sampler import sample_mini_exam
from vllm_hpt.model.client import LLMClient
from vllm_hpt.model.exam_agent import ExamAgent
from vllm_hpt.orchestrator.checkpoint import Checkpoint, CheckpointManager
from vllm_hpt.tuning.history import HistoryEntry, OptimizationHistory
from vllm_hpt.tuning.params import SamplingParams
from vllm_hpt.tuning.strategies import create_strategy
from vllm_hpt.tuning.strategies.base import SearchStrategy

from vllm_hpt.tuning.tuner_agent import TunerAgent
from vllm_hpt.utils.cache import ResponseCache
from vllm_hpt.utils.logger import get_logger, log_conversation, setup_logging

logger = get_logger(__name__)

TRADITIONAL_STRATEGIES = {"tpe", "gp", "cmaes", "grid"}


class TuningRunner:
    """Core orchestrator for the hyperparameter tuning workflow.

    Runs the complete loop: sample -> answer -> evaluate -> tune -> checkpoint.
    Supports validation at intervals, resume from checkpoint, and graceful
    Ctrl+C handling.
    """

    def __init__(
        self,
        settings: Settings,
        rounds: int = 20,
        mini_exam_size: int = 200,
        validation_interval: int = 3,
        data_dir: str = "data/ai2_arc/ARC-Challenge",
        concurrency: int = 5,
        cache_enabled: bool = True,
        tuning_mode: str = "tpe",
        seed: Optional[int] = None,
        grid_values: int = 3,
        wrong_sample_size: int = 5,
        output_truncate_length: int = 500,
        enable_thinking: Optional[bool] = None,
    ) -> None:
        self.settings = settings
        self.rounds = rounds
        self.mini_exam_size = mini_exam_size
        self.validation_interval = validation_interval
        self.data_dir = data_dir
        self.concurrency = concurrency
        self.cache_enabled = cache_enabled
        self.tuning_mode = tuning_mode.lower()
        self.strategy_name = self.tuning_mode if self.tuning_mode in TRADITIONAL_STRATEGIES else "tpe"
        self.seed = seed
        self.grid_values = grid_values
        self.wrong_sample_size = wrong_sample_size
        self.output_truncate_length = output_truncate_length
        # --enable-thinking is only applicable in a2a mode because traditional strategies don't use the tuner model
        self.enable_thinking = enable_thinking if self.tuning_mode == "a2a" else None

        self._interrupted = False
        self._checkpoint_manager = CheckpointManager()
        self._run_id: Optional[str] = None
        self._current_params: Optional[SamplingParams] = None
        self._best_params: Optional[SamplingParams] = None
        self._best_validation_accuracy: float = 0.0
        self._history: Optional[OptimizationHistory] = None
        self._strategy: Optional[SearchStrategy] = None

        logger.info(
            "tuning_runner_initialized",
            tuning_mode=self.tuning_mode,
            rounds=rounds,
            mini_exam_size=mini_exam_size,
            validation_interval=validation_interval,
            data_dir=data_dir,
            concurrency=concurrency,
            cache_enabled=cache_enabled,
            enable_thinking=enable_thinking,
        )

    def _setup_signal_handler(self) -> None:
        """Register SIGINT handler for graceful Ctrl+C shutdown."""

        def _handle_sigint(signum: int, frame: object) -> None:
            logger.warning("sigint_received", message="Ctrl+C detected, saving checkpoint and exiting")
            self._interrupted = True
            raise KeyboardInterrupt()

        signal.signal(signal.SIGINT, _handle_sigint)

    def _generate_run_id(self) -> str:
        """Generate a timestamp-based run ID."""
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def _create_exam_client(self, cache: Optional[ResponseCache] = None) -> LLMClient:
        exam_config = self.settings.get_exam_agent_config()
        return LLMClient(
            base_url=exam_config["base_url"],
            api_key=exam_config["api_key"],
            model=exam_config["model"],
            cache=cache,
            enable_thinking=self.enable_thinking,
        )

    def _create_tuner_client(self) -> LLMClient:
        tuner_config = self.settings.get_tuner_agent_config()
        return LLMClient(
            base_url=tuner_config["base_url"],
            api_key=tuner_config["api_key"],
            model=tuner_config["model"],
            enable_thinking=self.enable_thinking,
        )

    def _save_checkpoint(self, current_round: int) -> None:
        """Save current state to checkpoint.

        Args:
            current_round: The round number just completed.
        """
        if self._run_id is None or self._history is None or self._best_params is None:
            logger.warning("checkpoint_skip", reason="state not initialized")
            return

        checkpoint = Checkpoint(
            run_id=self._run_id,
            current_round=current_round,
            total_rounds=self.rounds,
            history=self._history,
            best_params=self._best_params,
            best_validation_accuracy=self._best_validation_accuracy,
            tuning_mode=self.tuning_mode,
            strategy_name=self.strategy_name,
            random_state=random.getstate(),
        )

        filepath = self._checkpoint_manager.auto_save_path(self._run_id)
        self._checkpoint_manager.save(checkpoint, filepath)
        logger.info("checkpoint_saved", round=current_round, filepath=filepath)

    async def _run_validation(
        self,
        exam_agent: ExamAgent,
        validation_questions: list[Question],
        params: SamplingParams,
        round_num: int,
    ) -> float:
        """Run validation on the full validation set.

        Args:
            exam_agent: ExamAgent instance.
            validation_questions: Full validation question set.
            params: Current sampling parameters.
            round_num: Current round number.

        Returns:
            Validation accuracy as a float.
        """
        dashboard = get_dashboard_state()
        await dashboard.update(
            mode="validating",
            current_question=0,
            total_questions=len(validation_questions),
        )

        logger.info(
            "validation_started",
            round=round_num,
            num_questions=len(validation_questions),
        )

        async def on_val_progress(current: int, total: int) -> None:
            await dashboard.update(current_question=current, total_questions=total)

        val_outputs = await exam_agent.answer_questions(
            validation_questions, params, progress_callback=on_val_progress
        )
        val_result = evaluate(validation_questions, val_outputs)

        logger.info(
            "validation_completed",
            round=round_num,
            accuracy=f"{val_result.accuracy:.4f}",
            correct=val_result.correct_count,
            total=val_result.total_count,
            parse_failures=val_result.parse_failure_count,
        )

        await dashboard.update(mode="tuning")

        return val_result.accuracy

    async def _run_loop(
        self,
        exam_agent: ExamAgent,
        tuner_agent: Optional["TunerAgent"],
        train_questions: list[Question],
        validation_questions: list[Question],
        start_round: int = 1,
    ) -> None:
        """Execute the main tuning loop.

        Args:
            exam_agent: ExamAgent for answering questions.
            tuner_agent: TunerAgent for A2A mode, or None for Traditional mode.
            train_questions: Training question set.
            validation_questions: Validation question set.
            start_round: Round to start from (for resume support).
        """
        assert self._current_params is not None
        assert self._history is not None
        assert self._strategy is not None or tuner_agent is not None

        dashboard = get_dashboard_state()

        for round_num in range(start_round, self.rounds + 1):
            if self._interrupted:
                logger.warning("loop_interrupted", round=round_num)
                self._save_checkpoint(round_num - 1)
                break

            await dashboard.update(
                mode="tuning",
                current_round=round_num,
                total_rounds=self.rounds,
                current_params=self._current_params.model_dump(),
                current_question=0,
                total_questions=self.mini_exam_size,
            )

            logger.info(
                "round_started",
                round=round_num,
                total_rounds=self.rounds,
                params=self._current_params.model_dump(),
            )

            # Step 1: Sample mini-exam from training set
            mini_exam = sample_mini_exam(train_questions, n=self.mini_exam_size)

            # Step 2: Exam agent answers questions
            async def on_progress(current: int, total: int) -> None:
                await dashboard.update(current_question=current, total_questions=total)

            try:
                raw_outputs = await exam_agent.answer_questions(
                    mini_exam, self._current_params, progress_callback=on_progress
                )
                from vllm_hpt.dashboard.state import ModelStatus
                dashboard.exam_model = ModelStatus(
                    name="exam_agent", status="ok", last_check=datetime.now().isoformat()
                )
            except Exception as e:
                from vllm_hpt.dashboard.state import ModelStatus
                dashboard.exam_model = ModelStatus(
                    name="exam_agent", status="error",
                    last_check=datetime.now().isoformat(), error_message=str(e)
                )
                raise

            # Step 3: Evaluate answers
            eval_result = evaluate(mini_exam, raw_outputs)

            logger.info(
                "round_train_result",
                round=round_num,
                accuracy=f"{eval_result.accuracy:.4f}",
                correct=eval_result.correct_count,
                total=eval_result.total_count,
                parse_failures=eval_result.parse_failure_count,
                wrong_sample=[
                    {
                        "id": wq.question_id,
                        "model": wq.model_answer,
                        "correct": wq.correct_answer,
                    }
                    for wq in eval_result.wrong_questions[:self.wrong_sample_size]
                ] if self.tuning_mode == "a2a" else [],
            )

            # Step 4: Create history entry
            entry = HistoryEntry(
                round_num=round_num,
                params=self._current_params,
                train_accuracy=eval_result.accuracy,
            )

            # Step 5: Validation check
            validation_accuracy: Optional[float] = None
            if round_num % self.validation_interval == 0 and validation_questions:
                validation_accuracy = await self._run_validation(
                    exam_agent, validation_questions, self._current_params, round_num
                )
                entry.validation_accuracy = validation_accuracy

                # Track best validation
                if validation_accuracy > self._best_validation_accuracy:
                    self._best_validation_accuracy = validation_accuracy
                    self._best_params = self._current_params
                    logger.info(
                        "new_best_validation",
                        round=round_num,
                        accuracy=f"{validation_accuracy:.4f}",
                        params=self._best_params.model_dump(),
                    )

            # Step 6: Add to history
            self._history.add_entry(entry)

            accuracy_history = [
                {
                    "round": e.round_num,
                    "train": e.train_accuracy,
                    "validation": e.validation_accuracy,
                }
                for e in self._history.entries
            ]
            await dashboard.update(
                accuracy_history=accuracy_history,
                current_question=self.mini_exam_size,
            )

            # Step 7: Get next params
            if self.tuning_mode == "a2a":
                assert tuner_agent is not None
                tuner_start_time = datetime.now()
                new_params = await tuner_agent.suggest_params(
                    self._current_params,
                    eval_result,
                    self._history,
                )
                tuner_end_time = datetime.now()
                tuner_duration_ms = (tuner_end_time - tuner_start_time).total_seconds() * 1000

                logger.info(
                    "round_conversation_summary",
                    round=round_num,
                    tuner_start_time=tuner_start_time.isoformat(),
                    tuner_end_time=tuner_end_time.isoformat(),
                    tuner_duration_ms=tuner_duration_ms,
                    current_accuracy=eval_result.accuracy,
                    current_params=self._current_params.model_dump(),
                    suggested_params=new_params.model_dump(),
                    wrong_count=len(eval_result.wrong_questions),
                    history_entries=len(self._history.entries),
                )

                log_conversation(
                    round_num=round_num,
                    timestamp=tuner_start_time,
                    prompt_to_tuner=tuner_agent.last_prompt or "",
                    tuner_response=tuner_agent.last_response or "",
                    current_params=self._current_params.model_dump(),
                    suggested_params=new_params.model_dump(),
                    accuracy=eval_result.accuracy,
                    wrong_questions=[
                        {"id": wq.question_id, "model": wq.model_answer, "correct": wq.correct_answer}
                        for wq in eval_result.wrong_questions[:self.wrong_sample_size]
                    ],
                    history_summary=self._history.format_for_prompt(top_k=5),
                    duration_ms=tuner_duration_ms,
                )

                logger.info(
                    "params_updated",
                    round=round_num,
                    old_params=self._current_params.model_dump(),
                    new_params=new_params.model_dump(),
                )
                self._current_params = new_params
            else:
                assert self._strategy is not None
                self._strategy.report_result(self._current_params, eval_result.accuracy)
                if round_num < self.rounds:
                    self._current_params = self._strategy.suggest_next_params()
                    logger.info("strategy_suggested_params", round=round_num, params=self._current_params.model_dump())

            # Step 8: Save checkpoint
            self._save_checkpoint(round_num)

            logger.info(
                "round_completed",
                round=round_num,
                total_rounds=self.rounds,
                best_validation_accuracy=f"{self._best_validation_accuracy:.4f}",
            )

    def _log_final_report(self) -> None:
        """Log the final summary report."""
        assert self._history is not None
        assert self._best_params is not None

        report_lines = [
            "",
            "=" * 70,
            "TUNING RUN COMPLETE",
            "=" * 70,
            f"Run ID: {self._run_id}",
            f"Total Rounds: {self.rounds}",
            f"Best Validation Accuracy: {self._best_validation_accuracy:.4f}",
            f"Best Params: {self._best_params.model_dump()}",
        ]

        # Show top-5 history entries
        top_entries = self._history.get_top_k(k=5, by="validation")
        if top_entries:
            report_lines.append("")
            report_lines.append("TOP 5 ROUNDS (by validation accuracy):")
            for entry in top_entries:
                val_acc = f"{entry.validation_accuracy:.4f}" if entry.validation_accuracy is not None else "N/A"
                report_lines.append(
                    f"  Round {entry.round_num}: train={entry.train_accuracy:.4f}, "
                    f"val={val_acc}, params={entry.params.model_dump()}"
                )

        report_lines.append("=" * 70)
        report = "\n".join(report_lines)

        logger.info("final_report", report=report)

    async def run(self) -> None:
        """Main entry point for a new tuning run.

        Executes the complete workflow:
        1. Setup logging, load datasets, create agents
        2. Initialize params and history
        3. Run main tuning loop
        4. Final test evaluation with best params
        """
        self._run_id = self._generate_run_id()
        setup_logging(run_id=self._run_id)
        self._setup_signal_handler()

        logger.info("tuning_run_starting", rounds=self.rounds, run_id=self._run_id)

        # Load datasets
        datasets = load_all_datasets(self.data_dir)
        train_questions = datasets.get("train", [])
        validation_questions = datasets.get("validation", [])

        if not train_questions:
            logger.error("no_train_questions", data_dir=self.data_dir)
            raise ValueError(f"No training questions found in {self.data_dir}")

        logger.info(
            "datasets_loaded",
            train=len(train_questions),
            validation=len(validation_questions),
        )

        # Create cache and clients
        cache = ResponseCache(enabled=self.cache_enabled) if self.cache_enabled else None
        exam_client = self._create_exam_client(cache=cache)

        # Create agents
        exam_agent = ExamAgent(exam_client, concurrency=self.concurrency)

        tuner_agent: Optional[TunerAgent] = None
        if self.tuning_mode == "a2a":
            tuner_client = self._create_tuner_client()
            tuner_agent = TunerAgent(
                tuner_client,
                wrong_sample_size=self.wrong_sample_size,
                output_truncate_length=self.output_truncate_length,
            )
            self._strategy = None
            self._current_params = SamplingParams()
        else:
            self._strategy = create_strategy(
                name=self.strategy_name,
                study_name=self._run_id,
                storage_path=f"checkpoints/{self._run_id}_study.db",
                seed=self.seed,
                **( {"grid_values": self.grid_values} if self.strategy_name == "grid" else {}),
            )
            self._current_params = self._strategy.suggest_next_params()

        self._best_params = SamplingParams()
        self._best_validation_accuracy = 0.0
        self._history = OptimizationHistory()

        logger.info(
            "run_initialized",
            run_id=self._run_id,
            initial_params=self._current_params.model_dump(),
        )

        try:
            await self._run_loop(
                exam_agent=exam_agent,
                tuner_agent=tuner_agent,
                train_questions=train_questions,
                validation_questions=validation_questions,
                start_round=1,
            )
        except KeyboardInterrupt:
            logger.warning("interrupted_saving_checkpoint")
            if self._run_id and self._history:
                self._save_checkpoint(max(1, len(self._history.entries)))
            raise

        # If validation never triggered (rounds < validation_interval),
        # fall back to the round with best train accuracy.
        if self._best_validation_accuracy == 0.0 and self._history and self._history.entries:
            best_entry = max(self._history.entries, key=lambda e: e.train_accuracy)
            self._best_params = best_entry.params
            logger.info(
                "best_params_from_train_fallback",
                round=best_entry.round_num,
                train_accuracy=f"{best_entry.train_accuracy:.4f}",
                params=best_entry.params.model_dump(),
            )

        self._log_final_report()

        if cache is not None:
            logger.info("cache_stats", **cache.stats())

    async def resume(self, checkpoint_path: str) -> None:
        """Resume a tuning run from a checkpoint.

        Args:
            checkpoint_path: Path to checkpoint JSON file.

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist.
        """
        checkpoint = self._checkpoint_manager.load(checkpoint_path)
        self._run_id = checkpoint.run_id

        setup_logging(run_id=self._run_id)
        self._setup_signal_handler()

        logger.info("tuning_run_resuming", checkpoint_path=checkpoint_path, run_id=self._run_id)

        self._history = checkpoint.history
        self._best_params = checkpoint.best_params
        self._best_validation_accuracy = checkpoint.best_validation_accuracy
        self.rounds = checkpoint.total_rounds

        checkpoint_mode = checkpoint.tuning_mode
        if checkpoint_mode == "a2a":
            self.tuning_mode = "a2a"
        else:
            self.tuning_mode = checkpoint_mode
            self.strategy_name = checkpoint.strategy_name

        if checkpoint.random_state is not None:
            try:
                random.setstate(tuple(checkpoint.random_state))
            except Exception as e:
                logger.warning("random_state_restore_failed", error=str(e))

        start_round = checkpoint.current_round + 1

        logger.info(
            "checkpoint_restored",
            run_id=self._run_id,
            resuming_from_round=start_round,
            total_rounds=self.rounds,
            best_validation_accuracy=f"{self._best_validation_accuracy:.4f}",
            history_entries=len(self._history.entries),
        )

        datasets = load_all_datasets(self.data_dir)
        train_questions = datasets.get("train", [])
        validation_questions = datasets.get("validation", [])

        if not train_questions:
            logger.error("no_train_questions", data_dir=self.data_dir)
            raise ValueError(f"No training questions found in {self.data_dir}")

        cache = ResponseCache(enabled=self.cache_enabled) if self.cache_enabled else None
        exam_client = self._create_exam_client(cache=cache)
        exam_agent = ExamAgent(exam_client, concurrency=self.concurrency)

        tuner_agent: Optional[TunerAgent] = None
        if self.tuning_mode == "a2a":
            tuner_client = self._create_tuner_client()
            tuner_agent = TunerAgent(
                tuner_client,
                wrong_sample_size=self.wrong_sample_size,
                output_truncate_length=self.output_truncate_length,
            )
            self._strategy = create_strategy(
                name="tpe",
                study_name=self._run_id,
                storage_path=f"checkpoints/{self._run_id}_study.db",
                seed=self.seed,
            )
            if self._history.entries:
                self._current_params = self._history.entries[-1].params
            else:
                self._current_params = self._best_params
        else:
            if self.strategy_name != checkpoint.strategy_name:
                logger.warning(
                    "strategy_mismatch",
                    checkpoint_strategy=checkpoint.strategy_name,
                    requested_strategy=self.strategy_name,
                )
            self._strategy = create_strategy(
                name=self.strategy_name,
                study_name=self._run_id,
                storage_path=f"checkpoints/{self._run_id}_study.db",
                seed=self.seed,
                **( {"grid_values": self.grid_values} if self.strategy_name == "grid" else {}),
            )
            self._current_params = self._strategy.suggest_next_params()

        try:
            await self._run_loop(
                exam_agent=exam_agent,
                tuner_agent=tuner_agent,
                train_questions=train_questions,
                validation_questions=validation_questions,
                start_round=start_round,
            )
        except KeyboardInterrupt:
            logger.warning("interrupted_saving_checkpoint")
            if self._run_id and self._history:
                self._save_checkpoint(max(1, len(self._history.entries)))
            raise

        self._log_final_report()

        if cache is not None:
            logger.info("cache_stats", **cache.stats())

    async def evaluate_with_params(self, params_dict: dict) -> None:
        """Evaluate the model with specific parameters on the test split only.

        Sends progress updates to the dashboard, prints results to terminal,
        and saves results to a file.

        Args:
            params_dict: Dictionary of sampling parameter values.
        """
        eval_run_id = f"eval_{self._generate_run_id()}"
        setup_logging(run_id=eval_run_id)

        logger.info("evaluate_with_params_started", params=params_dict, run_id=eval_run_id)

        params = SamplingParams(**params_dict)
        dashboard = get_dashboard_state()

        await dashboard.update(
            mode="evaluating",
            current_params=params.model_dump(),
            current_question=0,
            total_questions=0,
            eval_result=None,
            eval_result_file=None,
        )

        datasets = load_all_datasets(self.data_dir)
        test_questions = datasets.get("test", [])

        if not test_questions:
            logger.error("no_test_questions", data_dir=self.data_dir)
            raise ValueError(f"No test questions found in {self.data_dir}")

        exam_client = self._create_exam_client()
        exam_agent = ExamAgent(exam_client, concurrency=self.concurrency)

        await dashboard.update(
            total_questions=len(test_questions),
        )

        logger.info(
            "evaluating_split",
            split="test",
            num_questions=len(test_questions),
            params=params.model_dump(),
        )

        async def on_progress(current: int, total: int) -> None:
            await dashboard.update(current_question=current, total_questions=total)

        try:
            outputs = await exam_agent.answer_questions(
                test_questions, params, progress_callback=on_progress
            )
            from vllm_hpt.dashboard.state import ModelStatus
            dashboard.exam_model = ModelStatus(
                name="exam_agent", status="ok", last_check=datetime.now().isoformat()
            )
        except Exception as e:
            from vllm_hpt.dashboard.state import ModelStatus
            dashboard.exam_model = ModelStatus(
                name="exam_agent", status="error",
                last_check=datetime.now().isoformat(), error_message=str(e)
            )
            await dashboard.update(mode="idle")
            raise

        result = evaluate(test_questions, outputs)

        logger.info(
            "split_result",
            split="test",
            accuracy=f"{result.accuracy:.4f}",
            correct=result.correct_count,
            total=result.total_count,
            parse_failures=result.parse_failure_count,
        )
        print(f"\nTEST SET:")
        print(result.summary())

        wrong_sample = [
            {
                "id": wq.question_id,
                "model": wq.model_answer,
                "correct": wq.correct_answer,
            }
            for wq in result.wrong_questions[:10]
        ]

        eval_result_dict = {
            "accuracy": result.accuracy,
            "correct": result.correct_count,
            "total": result.total_count,
            "parse_failures": result.parse_failure_count,
            "wrong_sample": wrong_sample,
        }

        result_file = self._save_eval_result(params, result, eval_run_id)

        await dashboard.update(
            mode="eval_done",
            eval_result=eval_result_dict,
            eval_result_file=result_file,
        )

    def _save_eval_result(
        self, params: SamplingParams, result: EvaluationResult, run_id: str
    ) -> str:
        """Save evaluation result and parameters to a text file.

        Args:
            params: The sampling parameters used.
            result: The evaluation result.
            run_id: The evaluation run ID.

        Returns:
            Path to the saved result file.
        """
        import os

        params_str = "_".join(f"{k}{v}" for k, v in sorted(params.model_dump().items()))
        filename = f"result_{params_str}.txt"

        os.makedirs("results", exist_ok=True)
        filepath = os.path.join("results", filename)

        lines = [
            f"Evaluation Run: {run_id}",
            f"Timestamp: {datetime.now().isoformat()}",
            "",
            "Parameters:",
        ]
        for k, v in sorted(params.model_dump().items()):
            lines.append(f"  {k}: {v}")

        lines.extend([
            "",
            "Results (test split):",
            f"  Accuracy: {result.accuracy:.4f} ({result.accuracy:.2%})",
            f"  Correct: {result.correct_count}",
            f"  Total: {result.total_count}",
            f"  Parse Failures: {result.parse_failure_count}",
            f"  Wrong: {len(result.wrong_questions)}",
        ])

        if result.wrong_questions:
            lines.extend(["", f"Wrong Questions (first 10):"])
            for wq in result.wrong_questions[:10]:
                model_ans = wq.model_answer if wq.model_answer else "PARSE_FAILED"
                lines.append(f"  Q{wq.question_id}: Model={model_ans}, Correct={wq.correct_answer}")

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        logger.info("eval_result_saved", filepath=filepath)
        print(f"\nResult saved to: {filepath}")
        return filepath

    async def evaluate_interactive(self) -> None:
        """Wait for parameters from the dashboard UI, then run evaluation."""
        dashboard = get_dashboard_state()
        await dashboard.update(mode="waiting_params")

        logger.info("waiting_for_params_from_ui")
        print("Waiting for parameters from dashboard UI...")

        params_dict = await dashboard.wait_for_params()

        logger.info("params_received_from_ui", params=params_dict)
        print(f"Received parameters: {params_dict}")

        await self.evaluate_with_params(params_dict)
