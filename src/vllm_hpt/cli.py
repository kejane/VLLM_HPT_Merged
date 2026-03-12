"""CLI entry point for vLLM HPT using Typer.

Provides commands for running, resuming, and evaluating the tuning workflow.
"""

import asyncio
import json
import sys
import threading
import webbrowser
from typing import Optional

import typer
from pydantic import ValidationError

from vllm_hpt.config import Settings
from vllm_hpt.orchestrator.runner import TuningRunner

app = typer.Typer(
    name="vllm-hpt",
    help="vLLM Hyperparameter Tuning - Automated sampling parameter optimization",
    add_completion=False,
)


def _start_dashboard_server(port: int) -> None:
    from vllm_hpt.dashboard.server import run_server
    run_server(port=port)


@app.command()
def run(
    tuning_mode: str = typer.Option("tpe", "--tuning-mode", help="Tuning strategy: a2a (LLM-based) or traditional (tpe|gp|cmaes|grid)"),
    rounds: int = typer.Option(20, "--rounds", help="Number of tuning rounds"),
    mini_exam_size: int = typer.Option(200, "--mini-exam-size", help="Questions per mini-exam"),
    validation_interval: int = typer.Option(3, "--validation-interval", help="Validate every N rounds"),
    concurrency: int = typer.Option(5, "--concurrency", help="Concurrent API requests"),
    data_dir: str = typer.Option("data/ai2_arc/ARC-Challenge", "--data-dir", help="Data directory"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Disable API response caching"),
    grid_values: int = typer.Option(3, "--grid-values", help="Values per parameter for grid search"),
    seed: Optional[int] = typer.Option(None, "--seed", help="Random seed for search strategy"),
    wrong_sample_size: int = typer.Option(5, "--wrong-sample-size", help="Wrong questions shown to tuner agent per round (a2a only)"),
    output_truncate_length: int = typer.Option(500, "--output-truncate-length", help="Max chars of model output per wrong question (a2a only)"),
    enable_thinking: Optional[bool] = typer.Option(None, "--enable-thinking/--disable-thinking", help="Explicitly enable or disable model thinking/reasoning mode if supported by the API"),
    ui: bool = typer.Option(False, "--ui", help="Launch web dashboard for real-time monitoring"),
    ui_port: int = typer.Option(8501, "--ui-port", help="Dashboard server port"),
) -> None:
    """Run a new hyperparameter tuning session."""
    try:
        settings = Settings()
    except ValidationError as e:
        typer.echo("Error: Failed to load settings. Check your environment variables or .env file.", err=True)
        typer.echo(f"\nDetails:\n{e}", err=True)
        raise typer.Exit(code=1)

    valid_modes = ["a2a", "tpe", "gp", "cmaes", "grid"]
    if tuning_mode.lower() not in valid_modes:
        typer.echo(f"Error: Invalid --tuning-mode '{tuning_mode}'. Choose from: {', '.join(valid_modes)}", err=True)
        raise typer.Exit(code=1)

    runner = TuningRunner(
        settings=settings,
        rounds=rounds,
        mini_exam_size=mini_exam_size,
        validation_interval=validation_interval,
        data_dir=data_dir,
        concurrency=concurrency,
        cache_enabled=not no_cache,
        tuning_mode=tuning_mode.lower(),
        seed=seed,
        grid_values=grid_values,
        wrong_sample_size=wrong_sample_size,
        output_truncate_length=output_truncate_length,
        enable_thinking=enable_thinking,
    )

    if ui:
        server_thread = threading.Thread(target=_start_dashboard_server, args=(ui_port,), daemon=True)
        server_thread.start()
        typer.echo(f"Dashboard running at http://127.0.0.1:{ui_port}")
        webbrowser.open(f"http://127.0.0.1:{ui_port}")

    try:
        asyncio.run(runner.run())
    except KeyboardInterrupt:
        typer.echo("\nInterrupted by user. Checkpoint saved.", err=True)
        raise typer.Exit(code=130)
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def resume(
    checkpoint: str = typer.Option(..., "--checkpoint", help="Path to checkpoint file"),
    tuning_mode: Optional[str] = typer.Option(None, "--tuning-mode", help="Override tuning mode (a2a|tpe|gp|cmaes|grid); defaults to mode saved in checkpoint"),
    rounds: Optional[int] = typer.Option(None, "--rounds", help="Override total rounds"),
    mini_exam_size: Optional[int] = typer.Option(None, "--mini-exam-size", help="Override mini-exam size"),
    validation_interval: Optional[int] = typer.Option(None, "--validation-interval", help="Override validation interval"),
    concurrency: Optional[int] = typer.Option(None, "--concurrency", help="Override concurrency"),
    data_dir: Optional[str] = typer.Option(None, "--data-dir", help="Override data directory"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Disable API response caching"),
    wrong_sample_size: int = typer.Option(5, "--wrong-sample-size", help="Wrong questions shown to tuner agent (a2a only)"),
    output_truncate_length: int = typer.Option(500, "--output-truncate-length", help="Max chars of model output per wrong question (a2a only)"),
    enable_thinking: Optional[bool] = typer.Option(None, "--enable-thinking/--disable-thinking", help="Explicitly enable or disable model thinking/reasoning mode if supported by the API"),
    ui: bool = typer.Option(False, "--ui", help="Launch web dashboard for real-time monitoring"),
    ui_port: int = typer.Option(8501, "--ui-port", help="Dashboard server port"),
) -> None:
    """Resume a hyperparameter tuning session from a checkpoint."""
    try:
        settings = Settings()
    except ValidationError as e:
        typer.echo("Error: Failed to load settings. Check your environment variables or .env file.", err=True)
        typer.echo(f"\nDetails:\n{e}", err=True)
        raise typer.Exit(code=1)

    if tuning_mode is not None:
        valid_modes = ["a2a", "tpe", "gp", "cmaes", "grid"]
        if tuning_mode.lower() not in valid_modes:
            typer.echo(f"Error: Invalid --tuning-mode '{tuning_mode}'. Choose from: {', '.join(valid_modes)}", err=True)
            raise typer.Exit(code=1)

    runner = TuningRunner(
        settings=settings,
        rounds=rounds or 20,
        mini_exam_size=mini_exam_size or 200,
        validation_interval=validation_interval or 3,
        data_dir=data_dir or "data/ai2_arc/ARC-Challenge",
        concurrency=concurrency or 5,
        cache_enabled=not no_cache,
        tuning_mode=tuning_mode.lower() if tuning_mode else "tpe",
        wrong_sample_size=wrong_sample_size,
        output_truncate_length=output_truncate_length,
        enable_thinking=enable_thinking,
    )

    if ui:
        server_thread = threading.Thread(target=_start_dashboard_server, args=(ui_port,), daemon=True)
        server_thread.start()
        typer.echo(f"Dashboard running at http://127.0.0.1:{ui_port}")
        webbrowser.open(f"http://127.0.0.1:{ui_port}")

    try:
        asyncio.run(runner.resume(checkpoint))
    except FileNotFoundError:
        typer.echo(f"Error: Checkpoint file not found: {checkpoint}", err=True)
        raise typer.Exit(code=1)
    except KeyboardInterrupt:
        typer.echo("\nInterrupted by user. Checkpoint saved.", err=True)
        raise typer.Exit(code=130)
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def evaluate(
    params: Optional[str] = typer.Option(None, "--params", help="JSON string of sampling params (e.g., '{\"temperature\": 0.7}'). If omitted with --ui, params can be set from the dashboard."),
    data_dir: str = typer.Option("data/ai2_arc/ARC-Challenge", "--data-dir", help="Data directory"),
    concurrency: int = typer.Option(5, "--concurrency", help="Concurrent API requests"),
    enable_thinking: Optional[bool] = typer.Option(None, "--enable-thinking/--disable-thinking", help="Explicitly enable or disable model thinking/reasoning mode if supported by the API"),
    ui: bool = typer.Option(True, "--ui/--no-ui", help="Launch web dashboard for real-time monitoring"),
    ui_port: int = typer.Option(8501, "--ui-port", help="Dashboard server port"),
) -> None:
    """Evaluate the model on the test split with specific sampling parameters.

    If --params is provided, runs evaluation immediately.
    If --params is omitted and --ui is set, opens the dashboard where you can
    input parameters and start evaluation from the browser.
    """
    try:
        settings = Settings()
    except ValidationError as e:
        typer.echo("Error: Failed to load settings. Check your environment variables or .env file.", err=True)
        typer.echo(f"\nDetails:\n{e}", err=True)
        raise typer.Exit(code=1)

    runner = TuningRunner(
        settings=settings,
        rounds=1,
        data_dir=data_dir,
        concurrency=concurrency,
        cache_enabled=False,
        enable_thinking=enable_thinking,
    )

    if ui:
        server_thread = threading.Thread(target=_start_dashboard_server, args=(ui_port,), daemon=True)
        server_thread.start()
        eval_url = f"http://127.0.0.1:{ui_port}/evaluate"
        typer.echo(f"Dashboard running at {eval_url}")
        webbrowser.open(eval_url)

    if params is not None:
        try:
            params_dict = json.loads(params)
        except json.JSONDecodeError as e:
            typer.echo(f"Error: Invalid JSON in --params: {e}", err=True)
            typer.echo("Example: --params '{\"temperature\": 0.7, \"top_p\": 0.9}'", err=True)
            raise typer.Exit(code=1)

        try:
            asyncio.run(runner.evaluate_with_params(params_dict))
        except KeyboardInterrupt:
            typer.echo("\nInterrupted by user.", err=True)
            raise typer.Exit(code=130)
        except Exception as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(code=1)
    elif ui:
        try:
            asyncio.run(runner.evaluate_interactive())
        except KeyboardInterrupt:
            typer.echo("\nInterrupted by user.", err=True)
            raise typer.Exit(code=130)
        except Exception as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(code=1)
    else:
        typer.echo("Error: Either --params or --ui must be provided.", err=True)
        typer.echo("  Use --params to evaluate directly, or --ui to set params from the dashboard.", err=True)
        raise typer.Exit(code=1)


@app.command()
def visualize(
    checkpoint: Optional[str] = typer.Option(None, "--checkpoint", help="Path to checkpoint JSON file. If omitted, uses the latest checkpoint."),
    save_dir: Optional[str] = typer.Option(None, "--save-dir", help="Directory to save chart images. Defaults to checkpoints/viz_{run_id}/"),
    no_show: bool = typer.Option(False, "--no-show", help="Save charts only, do not open interactive window"),
) -> None:
    """Visualize tuning results from a checkpoint file.

    Generates accuracy trend, parameter evolution, parameter importance,
    and other charts matching the web UI style. Charts are saved as PNG files.
    """
    from vllm_hpt.orchestrator.checkpoint import CheckpointManager
    from vllm_hpt.visualization import visualize_checkpoint

    if checkpoint is None:
        mgr = CheckpointManager()
        checkpoint = mgr.find_latest()
        if checkpoint is None:
            typer.echo("Error: No checkpoint files found in checkpoints/", err=True)
            typer.echo("Specify a checkpoint with --checkpoint <path>", err=True)
            raise typer.Exit(code=1)
        typer.echo(f"Using latest checkpoint: {checkpoint}")

    try:
        saved = visualize_checkpoint(
            checkpoint_path=checkpoint,
            save_dir=save_dir,
            show=not no_show,
        )
    except FileNotFoundError:
        typer.echo(f"Error: Checkpoint file not found: {checkpoint}", err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
