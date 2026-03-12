"""Offline visualization of tuning results from checkpoint files.

Recreates the same charts shown in the web UI dashboard, plus additional
Optuna analysis charts (parameter importance, parameter evolution, etc.).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable, get_cmap

from vllm_hpt.orchestrator.checkpoint import CheckpointManager
from vllm_hpt.utils.logger import get_logger

logger = get_logger(__name__)

# Colors from web UI index.html CSS variables — must stay in sync
_BLUE = "#1d9bf0"         # --accent-blue
_GREEN = "#00ba7c"        # --accent-green
_RED = "#f4212e"          # --accent-red
_YELLOW = "#ffd400"       # --accent-yellow
_BG_DARK = "#0f1419"      # --bg-dark
_BG_CARD = "#1a1f26"      # --bg-card approx
_TEXT_PRIMARY = "#e7e9ea"  # --text-primary
_TEXT_SECONDARY = "#71767b"  # --text-secondary
_BORDER_COLOR = "#2a2f36"

_GRID_RGBA = (1.0, 1.0, 1.0, 0.05)

_PARAM_COLORS = {
    "temperature": "#1d9bf0",
    "top_p": "#00ba7c",
    "top_k": "#ffd400",
    "repetition_penalty": "#f4212e",
    "max_tokens": "#c77dff",
}


def _apply_dark_theme(ax: Any) -> None:
    ax.set_facecolor(_BG_CARD)
    ax.tick_params(colors=_TEXT_SECONDARY, which="both")
    ax.xaxis.label.set_color(_TEXT_SECONDARY)
    ax.yaxis.label.set_color(_TEXT_SECONDARY)
    ax.title.set_color(_TEXT_PRIMARY)
    for spine in ax.spines.values():
        spine.set_color(_BORDER_COLOR)
    ax.grid(True, color=_GRID_RGBA, linewidth=0.5)


def _style_figure(fig: Any) -> None:
    fig.patch.set_facecolor(_BG_DARK)


def _make_axes(figsize: tuple[int, int] = (12, 5), ax: Any = None) -> tuple[Any, Any]:
    if ax is not None:
        return ax.get_figure(), ax
    fig, new_ax = plt.subplots(figsize=figsize)
    return fig, new_ax


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Chart 1: Accuracy Trend — matches web UI index.html Chart.js config exactly
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_accuracy_trend(history: list[dict[str, Any]], ax: Any = None) -> Any:
    fig, ax = _make_axes((12, 5), ax)
    _apply_dark_theme(ax)

    rounds = [h["round_num"] for h in history]
    train: list[float] = [h["train_accuracy"] for h in history]
    val_raw: list[float | None] = [h.get("validation_accuracy") for h in history]

    ax.plot(rounds, train, color=_BLUE, linewidth=1.8, marker="o",
            markersize=3, label="Train Accuracy", zorder=3)
    ax.fill_between(rounds, train, alpha=0.10, color=_BLUE)

    val_rounds = [r for r, v in zip(rounds, val_raw) if v is not None]
    val_values: list[float] = [v for v in val_raw if v is not None]
    if val_values:
        ax.plot(val_rounds, val_values, color=_GREEN, linewidth=1.8, marker="s",
                markersize=3, label="Validation Accuracy", zorder=3)
        ax.fill_between(val_rounds, val_values, alpha=0.10, color=_GREEN)

    # Y-axis dynamic scaling — mirrors index.html lines 416-438 exactly:
    # MIN_VISIBLE_RANGE=0.1, otherwise 10% padding, clamped to [0,1]
    all_values = train + val_values
    if all_values:
        min_val = min(all_values)
        max_val = max(all_values)
        data_range = max_val - min_val
        MIN_VISIBLE_RANGE = 0.10
        if data_range < MIN_VISIBLE_RANGE:
            center = (min_val + max_val) / 2
            half_range = MIN_VISIBLE_RANGE / 2
            y_min = max(0.0, center - half_range)
            y_max = min(1.0, center + half_range)
        else:
            padding = data_range * 0.10
            y_min = max(0.0, min_val - padding)
            y_max = min(1.0, max_val + padding)
        ax.set_ylim(y_min, y_max)

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v * 100:.0f}%"))
    ax.set_xlabel("Round")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy Trend", fontsize=14, fontweight="bold", pad=12)
    ax.legend(facecolor=_BG_CARD, edgecolor=_BORDER_COLOR,
              labelcolor=_TEXT_SECONDARY, fontsize=9)

    return fig


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Chart 2: Best Accuracy So-Far (cumulative max)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_best_so_far(history: list[dict[str, Any]], ax: Any = None) -> Any:
    fig, ax = _make_axes((12, 5), ax)
    _apply_dark_theme(ax)

    rounds = [h["round_num"] for h in history]
    train: list[float] = [h["train_accuracy"] for h in history]

    best_train: list[float] = []
    cur_best = 0.0
    for t in train:
        cur_best = max(cur_best, t)
        best_train.append(cur_best)

    ax.plot(rounds, best_train, color=_BLUE, linewidth=2, label="Best Train (so far)", zorder=3)
    ax.fill_between(rounds, best_train, alpha=0.10, color=_BLUE)

    val_raw: list[float | None] = [h.get("validation_accuracy") for h in history]
    val_rounds = [r for r, v in zip(rounds, val_raw) if v is not None]
    val_values: list[float] = [v for v in val_raw if v is not None]
    if val_values:
        best_val: list[float] = []
        cur_best_v = 0.0
        for v in val_values:
            cur_best_v = max(cur_best_v, v)
            best_val.append(cur_best_v)
        ax.plot(val_rounds, best_val, color=_GREEN, linewidth=2, label="Best Validation (so far)", zorder=3)
        ax.fill_between(val_rounds, best_val, alpha=0.10, color=_GREEN)

    all_values = best_train + val_values
    if all_values:
        min_val = min(all_values)
        max_val = max(all_values)
        data_range = max_val - min_val
        padding = max(data_range * 0.10, 0.05)
        ax.set_ylim(max(0.0, min_val - padding), min(1.0, max_val + padding))

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v * 100:.0f}%"))
    ax.set_xlabel("Round")
    ax.set_ylabel("Accuracy")
    ax.set_title("Best Accuracy So Far", fontsize=14, fontweight="bold", pad=12)
    ax.legend(facecolor=_BG_CARD, edgecolor=_BORDER_COLOR,
              labelcolor=_TEXT_SECONDARY, fontsize=9)

    return fig


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Chart 3: Parameter Evolution (one subplot per parameter)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_param_evolution(history: list[dict[str, Any]], ax: Any = None) -> Any:
    param_names = ["temperature", "top_p", "top_k", "repetition_penalty", "max_tokens"]

    if ax is not None:
        fig = ax.get_figure()
        axes = [ax]
    else:
        fig, axes = plt.subplots(len(param_names), 1, figsize=(12, 3 * len(param_names)),
                                  sharex=True)
        if not hasattr(axes, "__len__"):
            axes = [axes]

    rounds = [h["round_num"] for h in history]
    train: list[float] = [h["train_accuracy"] for h in history]

    for i, pname in enumerate(param_names):
        ax_p = axes[i] if len(axes) > 1 else axes[0]
        _apply_dark_theme(ax_p)

        values = [h["params"][pname] for h in history]
        color = _PARAM_COLORS.get(pname, _BLUE)

        ax_p.plot(rounds, values, color=color, linewidth=1.5, marker="o",
                  markersize=3, zorder=3)

        ax_acc = ax_p.twinx()
        ax_acc.scatter(rounds, train, c=train, cmap="RdYlGn", s=12,
                       alpha=0.5, zorder=2, edgecolors="none")
        ax_acc.set_ylabel("train acc", fontsize=8, color=_TEXT_SECONDARY)
        ax_acc.tick_params(colors=_TEXT_SECONDARY, labelsize=8)
        ax_acc.set_ylim(0, 1)
        for spine in ax_acc.spines.values():
            spine.set_color(_BORDER_COLOR)

        ax_p.set_ylabel(pname, fontsize=10, color=color, fontweight="bold")
        ax_p.set_title(f"Parameter: {pname}", fontsize=11, fontweight="bold",
                       pad=6, color=_TEXT_PRIMARY)

    axes[-1].set_xlabel("Round")
    fig.suptitle("Parameter Evolution", fontsize=14, fontweight="bold",
                 color=_TEXT_PRIMARY, y=1.01)

    return fig


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Chart 4: Optuna Parameter Importance (requires study DB)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_param_importance(study_db_path: str, run_id: str, ax: Any = None) -> Any | None:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    db_path = Path(study_db_path)
    if not db_path.exists():
        logger.warning("optuna_db_not_found", path=str(db_path))
        return None

    try:
        study = optuna.load_study(
            study_name=run_id,
            storage=f"sqlite:///{db_path}",
        )
    except Exception as e:
        logger.warning("optuna_study_load_failed", error=str(e))
        return None

    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    if len(completed) < 2:
        logger.warning("too_few_trials_for_importance", n=len(completed))
        return None

    try:
        importances = optuna.importance.get_param_importances(study)
    except Exception as e:
        logger.warning("importance_calculation_failed", error=str(e))
        return None

    fig, ax = _make_axes((10, 5), ax)
    _apply_dark_theme(ax)

    names = list(importances.keys())
    values = list(importances.values())

    sorted_pairs = sorted(zip(names, values), key=lambda x: x[1], reverse=True)
    names = [p[0] for p in sorted_pairs]
    values = [p[1] for p in sorted_pairs]

    colors = [_PARAM_COLORS.get(n, _BLUE) for n in names]
    bars = ax.barh(names, values, color=colors, edgecolor=_BORDER_COLOR, height=0.6)

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=10, color=_TEXT_PRIMARY)

    ax.set_xlabel("Importance")
    ax.set_title("Parameter Importance (Optuna)", fontsize=14, fontweight="bold", pad=12)
    ax.invert_yaxis()
    ax.set_xlim(0, max(values) * 1.2 if values else 1)

    return fig


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Chart 5: Optuna Optimization History (trial value scatter + best-so-far)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_optuna_history(study_db_path: str, run_id: str, ax: Any = None) -> Any | None:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    db_path = Path(study_db_path)
    if not db_path.exists():
        return None

    try:
        study = optuna.load_study(study_name=run_id, storage=f"sqlite:///{db_path}")
    except Exception:
        return None

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        return None

    fig, ax = _make_axes((12, 5), ax)
    _apply_dark_theme(ax)

    trial_nums = [t.number for t in completed]
    values: list[float] = [t.value for t in completed if t.value is not None]

    if not values:
        return None

    ax.scatter(trial_nums[:len(values)], values, color=_BLUE, s=20, alpha=0.7, zorder=3, label="Trial Value")

    best_so_far: list[float] = []
    cur_best = 0.0
    for v in values:
        cur_best = max(cur_best, v)
        best_so_far.append(cur_best)
    ax.plot(trial_nums[:len(values)], best_so_far, color=_GREEN, linewidth=2,
            label="Best Value", zorder=4)

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v * 100:.0f}%"))
    ax.set_xlabel("Trial Number")
    ax.set_ylabel("Objective Value (Accuracy)")
    ax.set_title("Optuna Optimization History", fontsize=14, fontweight="bold", pad=12)
    ax.legend(facecolor=_BG_CARD, edgecolor=_BORDER_COLOR,
              labelcolor=_TEXT_SECONDARY, fontsize=9)

    return fig


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Chart 6: Parallel Coordinates (params normalized to [0,1], color = accuracy)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_parallel_coordinates(history: list[dict[str, Any]], ax: Any = None) -> Any:
    param_names = ["temperature", "top_p", "top_k", "repetition_penalty", "max_tokens"]
    from vllm_hpt.tuning.params import PARAM_RANGES

    fig, ax = _make_axes((12, 6), ax)
    _apply_dark_theme(ax)

    accuracies: list[float] = [h["train_accuracy"] for h in history]
    norm = Normalize(vmin=min(accuracies), vmax=max(accuracies))
    cmap = get_cmap("RdYlGn")

    for h in history:
        normalized: list[float] = []
        for pname in param_names:
            low, high = PARAM_RANGES[pname]
            val = h["params"][pname]
            if high == low:
                normalized.append(0.5)
            else:
                normalized.append((val - low) / (high - low))

        color = cmap(norm(h["train_accuracy"]))
        ax.plot(range(len(param_names)), normalized, color=color, alpha=0.5, linewidth=1)

    ax.set_xticks(range(len(param_names)))
    ax.set_xticklabels(param_names, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Normalized Value (0-1)")
    ax.set_title("Parallel Coordinates (color = accuracy)", fontsize=14,
                 fontweight="bold", pad=12)

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Train Accuracy", color=_TEXT_SECONDARY)
    cbar.ax.tick_params(colors=_TEXT_SECONDARY)

    return fig


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main entry point
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def visualize_checkpoint(
    checkpoint_path: str,
    save_dir: str | None = None,
    show: bool = True,
) -> list[str]:
    """Load a checkpoint and generate all visualization charts.

    Args:
        checkpoint_path: Path to checkpoint JSON file.
        save_dir: Directory to save PNG images. If None, creates ``viz_{run_id}/`` next to checkpoint.
        show: Whether to attempt opening an interactive matplotlib window.

    Returns:
        List of saved file paths.
    """
    mgr = CheckpointManager()
    checkpoint = mgr.load(checkpoint_path)

    run_id = checkpoint.run_id
    history_raw = checkpoint.history.to_dict()

    logger.info(
        "visualize_started",
        checkpoint=checkpoint_path,
        run_id=run_id,
        rounds=len(history_raw),
    )

    cp_path = Path(checkpoint_path)
    out_dir = Path(save_dir) if save_dir else cp_path.parent / f"viz_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    study_db = cp_path.parent / f"{run_id}_study.db"

    saved_files: list[str] = []

    def _save(fig: Any, name: str) -> None:
        path = out_dir / f"{name}.png"
        fig.savefig(str(path), dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor(), edgecolor="none")
        saved_files.append(str(path))
        logger.info("chart_saved", path=str(path))

    fig1, ax1 = plt.subplots(figsize=(12, 5))
    _style_figure(fig1)
    plot_accuracy_trend(history_raw, ax=ax1)
    fig1.tight_layout()
    _save(fig1, "01_accuracy_trend")

    fig2, ax2 = plt.subplots(figsize=(12, 5))
    _style_figure(fig2)
    plot_best_so_far(history_raw, ax=ax2)
    fig2.tight_layout()
    _save(fig2, "02_best_so_far")

    fig3 = plot_param_evolution(history_raw)
    _style_figure(fig3)
    fig3.tight_layout()
    _save(fig3, "03_param_evolution")

    fig4 = plot_param_importance(str(study_db), run_id)
    if fig4 is not None:
        _style_figure(fig4)
        fig4.tight_layout()
        _save(fig4, "04_param_importance")

    fig5 = plot_optuna_history(str(study_db), run_id)
    if fig5 is not None:
        _style_figure(fig5)
        fig5.tight_layout()
        _save(fig5, "05_optuna_history")

    fig6, ax6 = plt.subplots(figsize=(12, 6))
    _style_figure(fig6)
    plot_parallel_coordinates(history_raw, ax=ax6)
    fig6.tight_layout()
    _save(fig6, "06_parallel_coordinates")

    summary_path = out_dir / "summary.txt"
    _write_summary(checkpoint, history_raw, summary_path)
    saved_files.append(str(summary_path))

    print(f"\n{'=' * 60}")
    print(f"  Visualization complete for run: {run_id}")
    print(f"  Charts saved to: {out_dir}")
    print(f"  Files generated: {len(saved_files)}")
    for f in saved_files:
        print(f"    - {Path(f).name}")
    print(f"{'=' * 60}\n")

    if show:
        _show_charts()

    plt.close("all")
    return saved_files


def _write_summary(checkpoint: Any, history_raw: list[dict[str, Any]], path: Path) -> None:
    lines = [
        f"Run ID: {checkpoint.run_id}",
        f"Strategy: {checkpoint.strategy_name}",
        f"Rounds: {checkpoint.current_round} / {checkpoint.total_rounds}",
        f"Best Validation Accuracy: {checkpoint.best_validation_accuracy:.4f}",
        f"Best Params: {checkpoint.best_params.model_dump()}",
        "",
        "Round-by-Round History:",
        "-" * 80,
        f"{'Round':>5}  {'Train':>8}  {'Validation':>10}  {'temperature':>11}  {'top_p':>6}  {'top_k':>5}  {'rep_pen':>7}  {'max_tok':>7}",
        "-" * 80,
    ]

    for h in history_raw:
        val_str = f"{h['validation_accuracy']:.4f}" if h.get("validation_accuracy") is not None else "   -"
        p = h["params"]
        lines.append(
            f"{h['round_num']:>5}  {h['train_accuracy']:>8.4f}  {val_str:>10}"
            f"  {p['temperature']:>11.4f}  {p['top_p']:>6.4f}  {p['top_k']:>5}  {p['repetition_penalty']:>7.4f}  {p['max_tokens']:>7}"
        )

    lines.append("-" * 80)

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _show_charts() -> None:
    try:
        matplotlib.use("TkAgg")
        plt.show()
    except Exception:
        try:
            matplotlib.use("Qt5Agg")
            plt.show()
        except Exception:
            print("Note: No interactive display available. Charts saved as PNG files.")
