"""
Visualize arena results and compute Elo ratings.

Usage:
    uv run python -m memory_machines.arena.viz_results [results_dir]
"""

import argparse
import numpy as np

from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import cast
from numpy.typing import ArrayLike
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd
from rich.console import Console
from rich.table import Table

from memory_machines.arena.elo import (
    CALIBRATED_JUDGE,
    PairwiseOutcome,
    ScoringPolicy,
    WinLossTie,
    score_highlight,
    update_elo,
)

console = Console()


def compute_win_loss_tie(df: pd.DataFrame) -> dict[str, WinLossTie]:
    """Compute win/loss/tie records for each model.

    Args:
        df: DataFrame with ['highlight_id', 'model', 'model_score']

    Returns:
        Dict mapping model name to WinLossTie record
    """
    wins: dict[str, int] = defaultdict(int)
    losses: dict[str, int] = defaultdict(int)
    ties: dict[str, int] = defaultdict(int)

    outcomes = generate_pairwise_outcomes(df)
    for outcome in outcomes:
        model_a = outcome["model_a"]
        model_b = outcome["model_b"]
        result = outcome["outcome"]

        if result == 1.0:
            wins[model_a] += 1
            losses[model_b] += 1
        elif result == 0.0:
            wins[model_b] += 1
            losses[model_a] += 1
        else:
            ties[model_a] += 1
            ties[model_b] += 1

    models = set(wins.keys()) | set(losses.keys()) | set(ties.keys())
    return {model: WinLossTie(wins=wins[model], losses=losses[model], ties=ties[model]) for model in models}


def generate_pairwise_outcomes(
    df: pd.DataFrame,
    score_column: str = "model_score",
) -> list[PairwiseOutcome]:
    """Generate pairwise comparison outcomes from per-highlight model scores.

    For each highlight, generates all unordered pairs of models and compares
    their scores. No tie threshold is used - exact equality results in a tie.

    Args:
        df: DataFrame with columns ['highlight_id', 'model', score_column]
        score_column: Name of column containing model scores

    Returns:
        List of PairwiseOutcome dicts. Each highlight with N models generates
        N*(N-1)/2 pairwise comparisons.
    """
    assert score_column in df.columns, f"Score column {score_column} not found in DataFrame"
    assert "model" in df.columns, "Model column not found in DataFrame"
    outcomes: list[PairwiseOutcome] = []

    for highlight_id, group in df.groupby("highlight_id"):
        # Get all models and their scores for this highlight
        models_scores = list(zip(group["model"], group[score_column], strict=True))

        # Generate all unordered pairs
        for (model_a, score_a), (model_b, score_b) in combinations(models_scores, 2):
            # Determine outcome
            if score_a > score_b:
                outcome = 1.0  # a wins
            elif score_b > score_a:
                outcome = 0.0  # b wins
            else:
                outcome = 0.5  # tie

            outcomes.append(
                PairwiseOutcome(
                    highlight_id=int(highlight_id),  # pyright: ignore[reportArgumentType]
                    model_a=model_a,
                    model_b=model_b,
                    outcome=outcome,
                )
            )

    return outcomes


def compute_elo_ratings(
    df: pd.DataFrame,
    initial_rating: float = 1500.0,
    k: float = 32.0,
    n_runs: int = 10,
    random_state: int | None = 42,
) -> dict[str, float]:
    """Compute Elo ratings from pairwise comparisons.

    For stability, runs multiple times with different orderings and averages.
    Each run shuffles the highlight order and processes matches sequentially.

    Args:
        df: DataFrame with ['highlight_id', 'model', 'model_score']
        initial_rating: Starting rating for all models (default 1500)
        k: K-factor controlling rating volatility
        n_runs: Number of runs with different orderings (default 10)
        random_state: Random seed for reproducibility

    Returns:
        Dict mapping model name to averaged Elo rating
    """
    rng = np.random.default_rng(random_state)
    models = df["model"].unique()
    all_ratings: dict[str, list[float]] = defaultdict(list)

    for _ in range(n_runs):
        # Initialize ratings
        ratings: dict[str, float] = {model: initial_rating for model in models}

        # Shuffle highlight order
        highlight_ids = cast(ArrayLike, df["highlight_id"].unique())
        shuffled_ids = rng.permutation(highlight_ids)

        for highlight_id in shuffled_ids:
            group = df[df["highlight_id"] == highlight_id]
            if len(group) < 2:
                continue

            # Get all models and their scores for this highlight
            models_scores = list(zip(group["model"], group["model_score"], strict=True))

            # Update ratings for all pairs
            for (model_a, score_a), (model_b, score_b) in combinations(models_scores, 2):
                if score_a > score_b:
                    s_a = 1.0
                elif score_b > score_a:
                    s_a = 0.0
                else:
                    s_a = 0.5

                ratings[model_a], ratings[model_b] = update_elo(ratings[model_a], ratings[model_b], s_a, k)

        # Collect ratings from this run
        for model, rating in ratings.items():
            all_ratings[model].append(rating)

    # Average across runs
    return {model: float(np.mean(ratings_list)) for model, ratings_list in all_ratings.items()}


def _print_calibration_info(policy: ScoringPolicy) -> None:
    """Print calibration information for the scoring policy."""
    console.print()
    console.print(f"[bold]Scoring Policy: {policy.name}[/bold]")
    console.print("=" * 60)

    console.print("\n[cyan]Posterior Expected Utility per Judge Tier:[/cyan]")
    for tier in policy.tiers:
        console.print(f"  Judge says {tier} -> Expected utility: {policy.posterior_utilities[tier]:+.2f}")

    console.print("\n[cyan]Junk Probability per Judge Tier (P(human T0/T1 | judge)):[/cyan]")
    for tier in policy.tiers:
        console.print(f"  Judge says {tier} -> P(junk): {policy.junk_probabilities[tier]:.2%}")


def _print_tier_distribution(df: pd.DataFrame) -> None:
    """Print tier distribution table by model."""
    # Explode model_prompts for tier-level analysis
    df_expanded = df.explode("model_prompts", ignore_index=True)
    df_final = pd.concat(
        [df_expanded.drop("model_prompts", axis=1), df_expanded["model_prompts"].apply(pd.Series)],
        axis=1,
    )

    table = Table(title="Judged Tier Distribution by Model")
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Release", justify="right", style="white")
    table.add_column("T0", justify="right", style="magenta")
    table.add_column("T1", justify="right", style="magenta")
    table.add_column("T2", justify="right", style="green")
    table.add_column("T3", justify="right", style="green")
    table.add_column("Total", justify="right", style="white")

    # Sort models by release date
    models = list(df_final["model"].unique())
    models_sorted = sorted(models, key=lambda m: MODEL_RELEASE_DATES.get(str(m), "9999-12-31"))

    for model in models_sorted:
        df_model = cast(pd.DataFrame, df_final[df_final["model"] == model])
        tier_counts = df_model["judged_tier"].value_counts()
        total = len(df_model)

        t0_count = int(tier_counts.get("T0", 0))  # pyright: ignore[reportArgumentType]
        t1_count = int(tier_counts.get("T1", 0))  # pyright: ignore[reportArgumentType]
        t2_count = int(tier_counts.get("T2", 0))  # pyright: ignore[reportArgumentType]
        t3_count = int(tier_counts.get("T3", 0))  # pyright: ignore[reportArgumentType]

        model_key = _strip_model_suffix(str(model))
        release_date = MODEL_RELEASE_DATES.get(model_key, "unknown")

        table.add_row(
            str(model),
            release_date,
            f"{t0_count} ({t0_count / total:.1%})",
            f"{t1_count} ({t1_count / total:.1%})",
            f"{t2_count} ({t2_count / total:.1%})",
            f"{t3_count} ({t3_count / total:.1%})",
            str(total),
        )

    console.print()
    console.print(table)


def _compute_tier_stats(df_model: pd.DataFrame) -> dict[str, int | float]:
    """Compute tier distribution stats for a model's prompts.

    Args:
        df_model: DataFrame filtered to a single model with 'model_prompts' column

    Returns:
        Dict with tier counts, percentages, and derived stats
    """
    # Explode model_prompts to get individual prompts with their tiers
    df_expanded = df_model.explode("model_prompts", ignore_index=True)
    prompts_df = cast(pd.DataFrame, df_expanded["model_prompts"].apply(pd.Series))

    tier_counts = prompts_df["judged_tier"].value_counts()
    total_prompts = len(prompts_df)

    t0_count = int(tier_counts.get("T0", 0))  # pyright: ignore[reportArgumentType]
    t1_count = int(tier_counts.get("T1", 0))  # pyright: ignore[reportArgumentType]
    t2_count = int(tier_counts.get("T2", 0))  # pyright: ignore[reportArgumentType]
    t3_count = int(tier_counts.get("T3", 0))  # pyright: ignore[reportArgumentType]

    return {
        "total_prompts": total_prompts,
        "t0_count": t0_count,
        "t1_count": t1_count,
        "t2_count": t2_count,
        "t3_count": t3_count,
        "t0_pct": t0_count / total_prompts if total_prompts > 0 else 0,
        "t1_pct": t1_count / total_prompts if total_prompts > 0 else 0,
        "t2_pct": t2_count / total_prompts if total_prompts > 0 else 0,
        "t3_pct": t3_count / total_prompts if total_prompts > 0 else 0,
        "unusable_pct": (t0_count + t1_count) / total_prompts if total_prompts > 0 else 0,
        "usable_pct": (t2_count + t3_count) / total_prompts if total_prompts > 0 else 0,
    }


def _print_score_summary(df: pd.DataFrame) -> None:
    """Print score summary by model."""
    console.print()
    console.print("[bold]Cost-Sensitive Scoring Summary[/bold]")
    console.print("=" * 60)

    for model in sorted(df["model"].unique()):
        df_model = cast(pd.DataFrame, df[df["model"] == model])
        n_highlights = len(df_model)

        mean_score = df_model["model_score"].mean()
        std_score = df_model["model_score"].std()
        total_score = df_model["model_score"].sum()

        mean_benefit = df_model["metrics"].apply(lambda m: m["benefit"]).mean()
        mean_cost = df_model["metrics"].apply(lambda m: m["cost"]).mean()
        mean_num_prompts = df_model["metrics"].apply(lambda m: m["num_prompts"]).mean()

        # Compute tier distribution
        tier_stats = _compute_tier_stats(df_model)

        console.print(f"\n[cyan]Model: {model}[/cyan]")
        console.print(f"  Highlights: {n_highlights}")
        console.print(f"  Total prompts generated: {tier_stats['total_prompts']}")
        console.print(f"  Mean prompts per highlight: {mean_num_prompts:.1f}")
        console.print("  ---")
        console.print("[bold]  Tier Distribution:[/bold]")
        console.print(
            f"    T0: {tier_stats['t0_count']} ({tier_stats['t0_pct']:.1%}) | "
            f"T1: {tier_stats['t1_count']} ({tier_stats['t1_pct']:.1%}) | "
            f"T2: {tier_stats['t2_count']} ({tier_stats['t2_pct']:.1%}) | "
            f"T3: {tier_stats['t3_count']} ({tier_stats['t3_pct']:.1%})"
        )
        console.print(
            f"    [red]Unusable (T0+T1): {tier_stats['unusable_pct']:.1%}[/red] | "
            f"[green]Usable (T2+T3): {tier_stats['usable_pct']:.1%}[/green]"
        )
        console.print("  ---")
        console.print(f"  Mean Score (U = B - λC): {mean_score:+.3f} ± {std_score:.3f}")
        console.print(f"  Total Score: {total_score:+.1f}")
        console.print("  ---")
        console.print(f"  Mean Benefit (top-1 utility): {mean_benefit:+.3f}")
        console.print(f"  Mean Cost (expected junk): {mean_cost:.3f}")


def _print_elo_ratings(df: pd.DataFrame, n_runs: int = 10, k: float = 32.0) -> None:
    """Print Elo ratings with win/loss/tie records."""
    console.print()
    console.print("[bold]Elo Ratings[/bold]")
    console.print("=" * 60)

    # Compute ratings
    ratings = compute_elo_ratings(df, n_runs=n_runs, k=k, random_state=42)
    win_loss_tie = compute_win_loss_tie(df)

    # Count highlights per model
    highlights_per_model = df.groupby("model")["highlight_id"].nunique().to_dict()

    # Compute tier stats per model
    tier_stats_per_model: dict[str, dict[str, int | float]] = {}
    for model in df["model"].unique():
        df_model = cast(pd.DataFrame, df[df["model"] == model])
        tier_stats_per_model[model] = _compute_tier_stats(df_model)

    # Create table sorted by rating
    table = Table(title=f"Elo Ratings (averaged over {n_runs} runs, k={k})")
    table.add_column("Rank", justify="right", style="white")
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Release", justify="right", style="white")
    table.add_column("Elo", justify="right", style="yellow")
    table.add_column("Usable %", justify="right", style="green")
    table.add_column("Highlights", justify="right", style="white")
    table.add_column("W-L-T", justify="right", style="white")
    table.add_column("Win Rate", justify="right", style="green")

    sorted_models = sorted(ratings.keys(), key=lambda m: ratings[m], reverse=True)

    for rank, model in enumerate(sorted_models, 1):
        rating = ratings[model]
        wlt = win_loss_tie.get(model, {"wins": 0, "losses": 0, "ties": 0})
        total_games = wlt["wins"] + wlt["losses"] + wlt["ties"]
        win_rate = wlt["wins"] / total_games if total_games > 0 else 0
        n_highlights = highlights_per_model.get(model, 0)
        usable_pct = tier_stats_per_model.get(model, {}).get("usable_pct", 0)
        model_key = _strip_model_suffix(model)
        release_date = MODEL_RELEASE_DATES.get(model_key, "unknown")

        table.add_row(
            str(rank),
            model,
            release_date,
            f"{rating:.1f}",
            f"{usable_pct:.1%}",
            str(n_highlights),
            f"{wlt['wins']}-{wlt['losses']}-{wlt['ties']}",
            f"{win_rate:.1%}",
        )

    console.print()
    console.print(table)


# Model release dates for chronological sorting
MODEL_RELEASE_DATES = {
    "claude-3-haiku-20240307": "2024-03-07",
    "gpt-4o": "2024-05-13",
    "claude-3-7-sonnet-20250219": "2025-02-19",
    "gpt-4.1": "2025-04-14",
    "o3": "2025-04-16",
    "gemini-2.5-pro": "2025-06-17",
    "claude-sonnet-4-5": "2025-09-29",
    "claude-opus-4-5-20251101": "2025-11-01",
    "gemini-3-pro-preview": "2025-11-18",
    "gpt-5.2": "2025-12-11",
    "gemini-3-flash-preview": "2025-12-17",
    "claude-opus-4-6": "2026-02-05",
    "gemini-3.1-pro-preview": "2026-02-19",
    "gpt-5.4": "2026-03-05",
    "claude-opus-4-7": "2026-04-16",
    "gpt-5.5": "2026-04-23",
    "claude-opus-4-8": "2026-05-28",
    "claude-fable-5": "2026-06-09",
}


def _strip_model_suffix(model: str) -> str:
    """Strip common suffixes like '(simple)' from model names."""
    model = model.strip()
    if model.endswith("(simple)"):
        model = model[: -len("(simple)")].strip()
    return model


# Pretty display names for models
MODEL_PRETTY_NAMES = {
    "claude-3-haiku-20240307": "Claude 3 Haiku",
    "gpt-4o": "GPT-4o",
    "claude-3-7-sonnet-20250219": "Claude 3.7 Sonnet",
    "gpt-4.1": "GPT-4.1",
    "o3": "o3",
    "gemini-2.5-pro": "Gemini 2.5 Pro",
    "claude-sonnet-4-5-20250929": "Claude Sonnet 4.5",
    "claude-opus-4-5-20251101": "Claude Opus 4.5",
    "gemini-3-pro-preview": "Gemini 3 Pro",
    "gpt-5.2": "GPT-5.2",
    "gemini-3-flash-preview": "Gemini 3 Flash",
    "claude-opus-4-6": "Claude Opus 4.6",
    "gemini-3.1-pro-preview": "Gemini 3.1 Pro",
    "gpt-5.4": "GPT-5.4",
    "claude-opus-4-7": "Claude Opus 4.7",
    "claude-opus-4-8": "Claude Opus 4.8",
    "claude-fable-5": "Claude Fable 5",
}


def _plot_unusable_distribution(
    df: pd.DataFrame,
    output_path: Path | None = None,
    show: bool = False,
) -> None:
    """Plot stacked bar chart of unusable prompts (T0 + T1) by model.

    Args:
        df: DataFrame with model results
        output_path: Path to save the chart (if None, uses default)
        show: Whether to display the plot interactively
    """
    # Color scheme for unusable tiers (red gradient)
    TIER_COLORS = {
        "T0": "#E63946",  # Bright red (off-target/unusable)
        "T1": "#F4A261",  # Coral/salmon (needs major refactor)
    }

    # Compute tier stats per model (excluding control)
    model_stats: list[tuple[str, str, float, float]] = []  # (model, release_date, t0_pct, t1_pct)
    for model in df["model"].unique():
        model_str = str(model)
        # Skip the trivia control
        if "(trivia)" in model_str.lower():
            continue
        df_model = cast(pd.DataFrame, df[df["model"] == model])
        tier_stats = _compute_tier_stats(df_model)
        # Use release date if known, otherwise put at the end with a far-future date
        model_key = _strip_model_suffix(model_str)
        release_date = MODEL_RELEASE_DATES.get(model_key, "9999-12-31")
        model_stats.append(
            (
                model_str,
                release_date,
                float(tier_stats["t0_pct"]) * 100,
                float(tier_stats["t1_pct"]) * 100,
            )
        )

    # Sort by release date (ascending - oldest first)
    model_stats = sorted(model_stats, key=lambda x: x[1])

    models = [m[0] for m in model_stats]
    t0_values = [m[2] for m in model_stats]
    t1_values = [m[3] for m in model_stats]

    # Set up the plot
    x = np.arange(len(models))
    width = 0.6

    fig, ax = plt.subplots(figsize=(12, 7))

    # Create stacked bars with rounded corners
    for i, (model, t0, t1) in enumerate(zip(models, t0_values, t1_values)):
        cumulative = 0.0

        # T0 bar (bottom)
        if t0 > 0:
            rounded_rect = mpatches.FancyBboxPatch(
                (x[i] - width / 2, cumulative),
                width,
                t0,
                boxstyle=mpatches.BoxStyle("Round", pad=0.02),
                edgecolor="white",
                facecolor=TIER_COLORS["T0"],
                linewidth=1.5,
                alpha=0.9,
            )
            ax.add_patch(rounded_rect)
        cumulative += t0

        # T1 bar (top)
        if t1 > 0:
            rounded_rect = mpatches.FancyBboxPatch(
                (x[i] - width / 2, cumulative),
                width,
                t1,
                boxstyle=mpatches.BoxStyle("Round", pad=0.02),
                edgecolor="white",
                facecolor=TIER_COLORS["T1"],
                linewidth=1.5,
                alpha=0.9,
            )
            ax.add_patch(rounded_rect)

        # Add total unusable label above bar
        total_unusable = t0 + t1
        if total_unusable > 0:
            ax.text(
                x[i],
                total_unusable + 1,
                f"{total_unusable:.1f}%",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
                color="#2C3E50",
            )

    # Customize the plot
    ax.set_xlabel("Model (sorted by release date)", fontsize=13, fontweight="bold", color="#2C3E50")
    ax.set_ylabel("Unusable Prompts (%)", fontsize=13, fontweight="bold", color="#2C3E50")
    ax.set_title(
        "Unusable Prompt Distribution by Model (T0 + T1)",
        fontsize=15,
        fontweight="bold",
        pad=20,
        color="#2C3E50",
    )
    ax.set_xticks(x)
    pretty_labels = [MODEL_PRETTY_NAMES.get(_strip_model_suffix(m), _strip_model_suffix(m)) for m in models]
    ax.set_xticklabels(pretty_labels, fontsize=10, rotation=45, ha="right")

    # Create legend
    legend_patches = [
        mpatches.Patch(color=TIER_COLORS["T0"], label="T0 (Off-target/Unusable)", alpha=0.9),
        mpatches.Patch(color=TIER_COLORS["T1"], label="T1 (Needs Major Refactor)", alpha=0.9),
    ]
    ax.legend(
        handles=legend_patches,
        fontsize=11,
        loc="upper left",
        framealpha=0.95,
    )

    # Set axis limits and styling
    ax.set_xlim(-0.5, len(models) - 0.5)
    max_unusable = max(t0 + t1 for t0, t1 in zip(t0_values, t1_values))
    ax.set_ylim(0, max_unusable * 1.15)  # Add 15% headroom for labels
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0f}%"))
    ax.grid(axis="y", alpha=0.2, linestyle="--", color="#B0B0B0")
    ax.set_facecolor("#FAFAFA")

    plt.tight_layout()

    # Save the figure
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
        console.print(f"\n[green]Chart saved to: {output_path}[/green]")

    # Show if requested
    if show:
        plt.show()
    else:
        plt.close()


def _load_arena_results(
    results_dir: str | Path,
    policy: ScoringPolicy,
    lambda_junk: float = 1.0,
) -> pd.DataFrame:
    results_path = Path(results_dir)
    dfs = []

    for jsonl_file in results_path.glob("generation_results_*.jsonl"):
        df = pd.read_json(jsonl_file, lines=True)
        dfs.append(df)

    if not dfs:
        raise ValueError(f"No generation_results_*.jsonl files found in {results_dir}")

    df = pd.concat(dfs, ignore_index=True)

    # Compute scores using the policy
    df["metrics"] = df["model_prompts"].apply(
        lambda prompts: score_highlight(generated_prompts=prompts, policy=policy, lambda_junk=lambda_junk)
    )
    df["model_score"] = df["metrics"].apply(lambda m: m["score"])

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize arena results and compute Elo ratings")
    parser.add_argument(
        "results_dir",
        nargs="?",
        default="memory_machines/arena/results/",
        help="Directory containing generation_results_*.jsonl files",
    )
    parser.add_argument(
        "--lambda-junk",
        type=float,
        default=1.0,
        help="Penalty weight for junk prompts (default: 1.0)",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=100,
        help="Number of Elo calculation runs for stability (default: 100)",
    )
    parser.add_argument(
        "--k",
        type=float,
        default=32.0,
        help="Elo K-factor (default: 32.0)",
    )
    parser.add_argument(
        "--chart-output",
        type=str,
        default=None,
        help="Output path for unusable distribution chart (default: results_dir/unusable_distribution.png)",
    )
    parser.add_argument(
        "--show-chart",
        action="store_true",
        help="Display chart interactively (default: save only)",
    )
    args = parser.parse_args()

    policy = CALIBRATED_JUDGE

    console.print("[bold magenta]Arena Results Visualization[/bold magenta]")
    console.print(f"Results directory: {args.results_dir}")
    console.print(f"Lambda (junk penalty): {args.lambda_junk}")

    # Print calibration info
    _print_calibration_info(policy)

    # Load results
    console.print()
    console.print("[bold]Loading results...[/bold]")
    df = _load_arena_results(args.results_dir, policy, args.lambda_junk)
    console.print(f"Loaded {len(df)} results from {df['model'].nunique()} models")
    console.print(f"Unique highlights: {df['highlight_id'].nunique()}")

    # Print tier distribution
    _print_tier_distribution(df)

    # Print score summary
    _print_score_summary(df)

    # Print Elo ratings
    _print_elo_ratings(df, n_runs=args.n_runs, k=args.k)

    # Plot unusable distribution chart
    chart_output = (
        Path(args.chart_output) if args.chart_output else Path(args.results_dir) / "unusable_distribution.png"
    )
    _plot_unusable_distribution(df, output_path=chart_output, show=args.show_chart)

    console.print()


if __name__ == "__main__":
    main()
