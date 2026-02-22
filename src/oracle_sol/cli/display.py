"""
Terminal display engine for ORACLE-Sol.

Design principles (adapted from frontend design skill):
- Neutral base (dim white, grey) with single emerald accent
- No emojis anywhere — clean Unicode symbols only
- Generous spacing, not cramped
- Proper loading / error / empty states
- Hierarchy through color and weight, never clutter
"""

from __future__ import annotations

from typing import Optional, Sequence

from rich.align import Align
from rich.columns import Columns
from rich.console import Console, Group
from rich.markup import escape
from rich.padding import Padding
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from oracle_sol.core.predictor import SolubilityResult

# ---------------------------------------------------------------------------
# Palette — Zinc base + Emerald accent (no purple, no neon)
# ---------------------------------------------------------------------------

class Palette:
    ACCENT = "green"
    ACCENT_DIM = "dark_green"
    ACCENT_BRIGHT = "bright_green"
    TEXT = "white"
    TEXT_DIM = "bright_black"
    TEXT_MUTED = "dim"
    SURFACE = "grey23"
    ERROR = "red"
    WARNING = "yellow"
    BORDER = "grey37"
    SOLUBLE = "green"
    INSOLUBLE = "red"
    CONFIDENCE_HIGH = "green"
    CONFIDENCE_MED = "yellow"
    CONFIDENCE_LOW = "red"


# ---------------------------------------------------------------------------
# Symbols — no emojis, clean Unicode only
# ---------------------------------------------------------------------------

class Sym:
    ARROW_RIGHT = "\u2192"
    ARROW_DOWN = "\u2193"
    BULLET = "\u2022"
    CHECK = "\u2713"
    CROSS = "\u2717"
    DIAMOND = "\u25c6"
    BAR_FULL = "\u2588"
    BAR_MED = "\u2593"
    BAR_LOW = "\u2591"
    DOT = "\u00b7"
    PIPE = "\u2502"
    DASH = "\u2500"
    CORNER_TL = "\u256d"
    CORNER_TR = "\u256e"
    CORNER_BL = "\u2570"
    CORNER_BR = "\u256f"


console = Console()


# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------

def print_banner() -> None:
    banner_text = Text()
    banner_text.append("\n")
    banner_text.append("  ORACLE", style=f"bold {Palette.ACCENT}")
    banner_text.append("-Sol", style=f"bold {Palette.TEXT}")
    banner_text.append(f"  {Sym.DOT}  ", style=Palette.TEXT_DIM)
    banner_text.append("v0.6.0", style=Palette.TEXT_DIM)
    banner_text.append("\n")
    banner_text.append(
        "  Protein solubility prediction", style=Palette.TEXT_DIM
    )
    banner_text.append("\n")
    console.print(banner_text)
    console.print(
        Rule(style=Palette.BORDER, characters=Sym.DASH),
    )
    console.print()


# ---------------------------------------------------------------------------
# Single result display
# ---------------------------------------------------------------------------

def print_result(result: SolubilityResult, show_residues: bool = False) -> None:
    """Display a single prediction result with premium formatting."""

    # Header: name or truncated sequence
    name_display = result.name or result.sequence[:40] + ("..." if len(result.sequence) > 40 else "")

    header = Text()
    header.append(f"  {Sym.DIAMOND} ", style=Palette.ACCENT)
    header.append(name_display, style=f"bold {Palette.TEXT}")
    if result.truncated:
        header.append(f"  [truncated to {1022} aa]", style=Palette.WARNING)
    console.print(header)
    console.print()

    # Score bar
    _print_score_bar(result.score, result.label, result.confidence)
    console.print()

    # Metrics row
    metrics = Text()
    metrics.append("    Length  ", style=Palette.TEXT_DIM)
    metrics.append(f"{result.length:,}", style=Palette.TEXT)
    metrics.append(f"  {Sym.PIPE}  ", style=Palette.BORDER)
    metrics.append("Score  ", style=Palette.TEXT_DIM)
    metrics.append(result.score_pct, style=f"bold {_label_color(result.label)}")
    metrics.append(f"  {Sym.PIPE}  ", style=Palette.BORDER)
    metrics.append("Label  ", style=Palette.TEXT_DIM)
    metrics.append(result.label.upper(), style=f"bold {_label_color(result.label)}")
    metrics.append(f"  {Sym.PIPE}  ", style=Palette.BORDER)
    metrics.append("Confidence  ", style=Palette.TEXT_DIM)
    metrics.append(
        result.confidence,
        style=f"bold {_confidence_color(result.confidence)}",
    )
    console.print(metrics)

    # Per-residue heatmap
    if show_residues and result.residue_scores is not None:
        console.print()
        _print_residue_heatmap(result.residue_scores, result.length)

    console.print()
    console.print(Rule(style=Palette.BORDER, characters=Sym.DASH))
    console.print()


# ---------------------------------------------------------------------------
# Score bar — the signature visual element
# ---------------------------------------------------------------------------

def _print_score_bar(score: float, label: str, confidence: str) -> None:
    """Render a horizontal score bar with the prediction point marked."""
    bar_width = 50
    filled = int(score * bar_width)
    color = _label_color(label)

    bar = Text("    ")

    # Left label
    bar.append("INSOL ", style=f"dim {Palette.INSOLUBLE}")

    # The bar itself
    for i in range(bar_width):
        if i < filled:
            bar.append(Sym.BAR_FULL, style=color)
        elif i == filled:
            bar.append(Sym.DIAMOND, style=f"bold {Palette.TEXT}")
        else:
            bar.append(Sym.BAR_LOW, style=Palette.TEXT_DIM)

    # Right label
    bar.append(" SOL", style=f"dim {Palette.SOLUBLE}")

    console.print(bar)


# ---------------------------------------------------------------------------
# Per-residue heatmap
# ---------------------------------------------------------------------------

def _print_residue_heatmap(
    scores: list[float], total_length: int, width: int = 60
) -> None:
    """
    Render a per-residue contribution heatmap.
    Green = contributes to solubility, Red = contributes to insolubility.
    """
    console.print(
        Text("    Residue contribution map:", style=Palette.TEXT_DIM)
    )
    console.print()

    if not scores:
        return

    # Normalize scores to 0-1
    min_s, max_s = min(scores), max(scores)
    range_s = max_s - min_s if max_s > min_s else 1.0
    normalized = [(s - min_s) / range_s for s in scores]

    # Downsample to fit terminal width
    display_width = min(width, len(normalized))
    chunk_size = max(1, len(normalized) // display_width)

    row = Text("    ")
    for i in range(0, len(normalized), chunk_size):
        chunk = normalized[i : i + chunk_size]
        avg = sum(chunk) / len(chunk)
        row.append(Sym.BAR_FULL, style=_heatmap_color(avg))

    console.print(row)

    # Position labels
    labels = Text("    ")
    labels.append("1", style=Palette.TEXT_DIM)
    gap = display_width - len(str(total_length)) - 1
    if gap > 0:
        labels.append(" " * gap, style="")
    labels.append(str(total_length), style=Palette.TEXT_DIM)
    console.print(labels)

    # Legend
    legend = Text("    ")
    legend.append(Sym.BAR_FULL, style="red")
    legend.append(" low ", style=Palette.TEXT_DIM)
    legend.append(Sym.BAR_FULL, style="yellow")
    legend.append(" mid ", style=Palette.TEXT_DIM)
    legend.append(Sym.BAR_FULL, style="green")
    legend.append(" high", style=Palette.TEXT_DIM)
    console.print(legend)


def _heatmap_color(value: float) -> str:
    """Map a 0-1 value to a red-yellow-green gradient."""
    if value < 0.33:
        return "red"
    elif value < 0.66:
        return "yellow"
    return "green"


# ---------------------------------------------------------------------------
# Batch results table
# ---------------------------------------------------------------------------

def print_batch_results(
    results: list[SolubilityResult],
    sort_by: str = "score",
) -> None:
    """Display batch results in a ranked table."""

    if sort_by == "score":
        results = sorted(results, key=lambda r: r.score, reverse=True)
    elif sort_by == "length":
        results = sorted(results, key=lambda r: r.length)

    table = Table(
        show_header=True,
        header_style=f"bold {Palette.TEXT_DIM}",
        border_style=Palette.BORDER,
        box=None,
        padding=(0, 2),
        collapse_padding=True,
    )

    table.add_column("#", style=Palette.TEXT_DIM, width=5, justify="right")
    table.add_column("Name", style=Palette.TEXT, max_width=30)
    table.add_column("Length", style=Palette.TEXT_DIM, justify="right", width=8)
    table.add_column("Score", justify="right", width=8)
    table.add_column("Label", width=12)
    table.add_column("Confidence", width=12)
    table.add_column("", width=20)  # Mini bar

    for rank, r in enumerate(results, 1):
        # Mini inline bar
        bar_len = 15
        filled = int(r.score * bar_len)
        mini_bar = Text()
        for j in range(bar_len):
            if j < filled:
                mini_bar.append(Sym.BAR_FULL, style=_label_color(r.label))
            else:
                mini_bar.append(Sym.BAR_LOW, style=Palette.TEXT_DIM)

        name = r.name or r.sequence[:20] + "..."

        table.add_row(
            str(rank),
            name,
            f"{r.length:,}",
            Text(r.score_pct, style=f"bold {_label_color(r.label)}"),
            Text(r.label, style=_label_color(r.label)),
            Text(r.confidence, style=_confidence_color(r.confidence)),
            mini_bar,
        )

    console.print()
    console.print(Padding(table, (0, 2)))
    console.print()


# ---------------------------------------------------------------------------
# Batch summary stats
# ---------------------------------------------------------------------------

def print_batch_summary(results: list[SolubilityResult]) -> None:
    """Print aggregate statistics for a batch."""
    n = len(results)
    n_sol = sum(1 for r in results if r.label == "soluble")
    n_insol = n - n_sol
    avg_score = sum(r.score for r in results) / n if n > 0 else 0
    high_conf = sum(1 for r in results if r.confidence == "high")

    console.print()
    summary = Text("  Summary\n\n", style=f"bold {Palette.TEXT}")
    summary.append(f"    Sequences analyzed   {n}\n", style=Palette.TEXT_DIM)
    summary.append(f"    Predicted soluble    ", style=Palette.TEXT_DIM)
    summary.append(f"{n_sol}", style=Palette.SOLUBLE)
    summary.append(f"  ({n_sol/n*100:.0f}%)\n" if n > 0 else "\n", style=Palette.TEXT_DIM)
    summary.append(f"    Predicted insoluble  ", style=Palette.TEXT_DIM)
    summary.append(f"{n_insol}", style=Palette.INSOLUBLE)
    summary.append(f"  ({n_insol/n*100:.0f}%)\n" if n > 0 else "\n", style=Palette.TEXT_DIM)
    summary.append(f"    Average score        {avg_score:.3f}\n", style=Palette.TEXT_DIM)
    summary.append(f"    High confidence      {high_conf}/{n}\n", style=Palette.TEXT_DIM)
    console.print(summary)


# ---------------------------------------------------------------------------
# Comparison display
# ---------------------------------------------------------------------------

def print_comparison(
    results: list[SolubilityResult],
    references: list[SolubilityResult],
) -> None:
    """Display user results alongside reference proteins."""

    console.print()
    console.print(
        Text("  Reference comparison\n", style=f"bold {Palette.TEXT}")
    )

    # Combined list with source tag
    all_items = []
    for r in references:
        all_items.append(("ref", r))
    for r in results:
        all_items.append(("usr", r))

    all_items.sort(key=lambda x: x[1].score, reverse=True)

    table = Table(
        show_header=True,
        header_style=f"bold {Palette.TEXT_DIM}",
        border_style=Palette.BORDER,
        box=None,
        padding=(0, 2),
    )

    table.add_column("", width=5)
    table.add_column("Protein", max_width=25)
    table.add_column("Length", justify="right", width=8)
    table.add_column("Score", justify="right", width=8)
    table.add_column("Label", width=12)
    table.add_column("", width=20)

    for source, r in all_items:
        tag = Text("REF", style=Palette.TEXT_DIM) if source == "ref" else Text(
            f" {Sym.ARROW_RIGHT} ", style=f"bold {Palette.ACCENT}"
        )
        name_style = Palette.TEXT_DIM if source == "ref" else f"bold {Palette.TEXT}"

        bar_len = 15
        filled = int(r.score * bar_len)
        mini_bar = Text()
        for j in range(bar_len):
            if j < filled:
                mini_bar.append(Sym.BAR_FULL, style=_label_color(r.label))
            else:
                mini_bar.append(Sym.BAR_LOW, style=Palette.TEXT_DIM)

        table.add_row(
            tag,
            Text(r.name or "query", style=name_style),
            f"{r.length:,}",
            Text(r.score_pct, style=f"bold {_label_color(r.label)}"),
            Text(r.label, style=_label_color(r.label)),
            mini_bar,
        )

    console.print(Padding(table, (0, 2)))
    console.print()


# ---------------------------------------------------------------------------
# Progress / Status
# ---------------------------------------------------------------------------

def create_progress() -> Progress:
    """Create a styled progress bar for batch processing."""
    return Progress(
        SpinnerColumn(style=Palette.ACCENT),
        TextColumn("[bold]{task.description}"),
        BarColumn(
            bar_width=30,
            style=Palette.TEXT_DIM,
            complete_style=Palette.ACCENT,
            finished_style=Palette.ACCENT_DIM,
        ),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
    )


def print_status(message: str, style: str = Palette.TEXT_DIM) -> None:
    console.print(Text(f"  {Sym.BULLET} {message}", style=style))


def print_success(message: str) -> None:
    console.print(Text(f"  {Sym.CHECK} {message}", style=Palette.ACCENT))


def print_error(message: str) -> None:
    console.print(Text(f"  {Sym.CROSS} {message}", style=Palette.ERROR))


def print_warning(message: str) -> None:
    console.print(Text(f"  {Sym.BULLET} {message}", style=Palette.WARNING))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _label_color(label: str) -> str:
    return Palette.SOLUBLE if label == "soluble" else Palette.INSOLUBLE


def _confidence_color(confidence: str) -> str:
    return {
        "high": Palette.CONFIDENCE_HIGH,
        "medium": Palette.CONFIDENCE_MED,
        "low": Palette.CONFIDENCE_LOW,
    }.get(confidence, Palette.TEXT_DIM)
