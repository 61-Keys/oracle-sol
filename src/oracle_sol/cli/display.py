"""
Terminal display engine for ORACLE-Sol.

Design: Cyberpunk biology aesthetic.
Teal accent, big ASCII headers, clean data presentation.
No emojis — Unicode symbols only.
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
# Palette — Teal/Cyan biology-hacker aesthetic
# ---------------------------------------------------------------------------

class Palette:
    ACCENT = "#00d4aa"
    ACCENT_DIM = "#007a63"
    ACCENT_BRIGHT = "#5fffdf"
    HEADER = "#00d4aa"
    HEADER_DIM = "#006e58"
    TEXT = "#e0e0e0"
    TEXT_DIM = "#666666"
    TEXT_MUTED = "dim"
    SOLUBLE = "#00d4aa"
    INSOLUBLE = "#ff5555"
    WARNING = "#ffaa00"
    ERROR = "#ff5555"
    SURFACE = "grey23"
    BORDER = "#333333"
    CONFIDENCE_HIGH = "#00d4aa"
    CONFIDENCE_MED = "#ffaa00"
    CONFIDENCE_LOW = "#ff5555"


# ---------------------------------------------------------------------------
# Symbols — no emojis, clean Unicode only
# ---------------------------------------------------------------------------

class Sym:
    ARROW_RIGHT = "\u2192"
    ARROW_DOWN = "\u2193"
    ARROW_UP = "\u2191"
    BULLET = "\u2022"
    CHECK = "\u2713"
    CROSS = "\u2717"
    DIAMOND = "\u25c6"
    BAR_FULL = "\u2588"
    BAR_MED = "\u2593"
    BAR_LOW = "\u2591"
    BAR_THIN = "\u2581"
    DOT = "\u00b7"
    PIPE = "\u2502"
    DASH = "\u2500"
    THICK_DASH = "\u2501"
    CORNER_TL = "\u256d"
    CORNER_TR = "\u256e"
    CORNER_BL = "\u2570"
    CORNER_BR = "\u256f"


console = Console()


# ---------------------------------------------------------------------------
# ASCII Art Banner
# ---------------------------------------------------------------------------

BANNER_ART = """
[#007a63]   ________  ___   ______   ____        _____      __[/]
[#00d4aa]  / __ / _ \\/ _ | / ___/  / __/ ___   / ___/___  / /[/]
[#00d4aa] / /_// , _/ __ |/ /__   / _/  /___/ _\\__ \\/ _ \\/ / [/]
[#5fffdf] \\____/_/|_/_/ |_|\\___/  /___/       /____/\\___/_/  [/]
"""


def print_banner() -> None:
    """Print the big ASCII art banner."""
    console.print(BANNER_ART, highlight=False)

    sub = Text("  ")
    sub.append("Protein Solubility Prediction", style=Palette.TEXT_DIM)
    sub.append("  |  ", style=Palette.BORDER)
    sub.append("ESM2-650M + MLP", style=Palette.ACCENT_DIM)
    sub.append("  |  ", style=Palette.BORDER)
    sub.append("v0.6.0", style=Palette.TEXT_DIM)
    console.print(sub)
    console.print()
    _print_separator()
    console.print()


def print_mini_banner() -> None:
    """Compact banner for secondary commands."""
    t = Text()
    t.append("\n  ORACLE", style=f"bold {Palette.ACCENT}")
    t.append("-Sol", style=Palette.TEXT)
    t.append(f"  {Sym.DOT}  ", style=Palette.TEXT_DIM)
    t.append("v0.6.0\n", style=Palette.TEXT_DIM)
    console.print(t)


# ---------------------------------------------------------------------------
# System status block
# ---------------------------------------------------------------------------

def print_system_status(device: str, weights_source: str = "bundled") -> None:
    """Print a status block during model loading."""
    console.print()
    s = Text()
    s.append("  SYSTEM\n", style=f"bold {Palette.ACCENT}")
    s.append(f"  {Sym.THICK_DASH * 44}\n", style=Palette.BORDER)

    dev_color = Palette.ACCENT if "cuda" in device or "mps" in device else Palette.TEXT
    rows = [
        ("Device", device, dev_color),
        ("Backbone", "ESM2-650M (frozen)", Palette.TEXT),
        ("Head", "MLP 1280 > 512 > 128 > 2", Palette.TEXT),
        ("Weights", weights_source, Palette.ACCENT_DIM),
        ("Max length", "1022 residues", Palette.TEXT),
    ]
    for label, val, color in rows:
        s.append(f"  {label:<17}", style=Palette.TEXT_DIM)
        s.append(f"{val}\n", style=color)

    s.append(f"  {Sym.THICK_DASH * 44}", style=Palette.BORDER)
    console.print(s)
    console.print()


# ---------------------------------------------------------------------------
# Single result display
# ---------------------------------------------------------------------------

def print_result(result: SolubilityResult, show_residues: bool = False) -> None:
    """Display a single prediction result."""
    name_display = result.name or result.sequence[:40] + ("..." if len(result.sequence) > 40 else "")

    console.print()
    header = Text()
    header.append(f"  {Sym.DIAMOND} ", style=Palette.ACCENT)
    header.append(name_display, style=f"bold {Palette.TEXT}")
    if result.truncated:
        header.append(f"  [truncated to 1022 aa]", style=Palette.WARNING)
    console.print(header)
    console.print()

    # Big score
    pct = f"{result.score * 100:.1f}"
    color = _label_color(result.label)
    score_line = Text("    ")
    score_line.append(pct, style=f"bold {color}")
    score_line.append(" %", style=color)
    score_line.append(f"    {result.label.upper()}", style=f"bold {color}")
    console.print(score_line)
    console.print()

    # Score bar
    _print_score_bar(result.score, result.label)
    console.print()

    # Metrics
    m = Text()
    m.append("    Length  ", style=Palette.TEXT_DIM)
    m.append(f"{result.length:,}", style=Palette.TEXT)
    m.append(f"  {Sym.PIPE}  ", style=Palette.BORDER)
    m.append("Score  ", style=Palette.TEXT_DIM)
    m.append(result.score_pct, style=f"bold {_label_color(result.label)}")
    m.append(f"  {Sym.PIPE}  ", style=Palette.BORDER)
    m.append("Verdict  ", style=Palette.TEXT_DIM)
    m.append(result.label.upper(), style=f"bold {_label_color(result.label)}")
    m.append(f"  {Sym.PIPE}  ", style=Palette.BORDER)
    m.append("Confidence  ", style=Palette.TEXT_DIM)
    m.append(result.confidence.upper(), style=f"bold {_confidence_color(result.confidence)}")
    console.print(m)

    if show_residues and result.residue_scores is not None:
        console.print()
        _print_residue_heatmap(result.residue_scores, result.length)

    console.print()
    _print_separator()
    console.print()


# ---------------------------------------------------------------------------
# Score bar
# ---------------------------------------------------------------------------

def _print_score_bar(score: float, label: str) -> None:
    bar_width = 50
    filled = int(score * bar_width)

    bar = Text("    ")
    bar.append("INSOL ", style=f"dim {Palette.INSOLUBLE}")

    for i in range(bar_width):
        pos = i / bar_width
        if i < filled:
            if pos < 0.3:
                bar.append(Sym.BAR_FULL, style=Palette.INSOLUBLE)
            elif pos < 0.5:
                bar.append(Sym.BAR_FULL, style=Palette.WARNING)
            else:
                bar.append(Sym.BAR_FULL, style=Palette.SOLUBLE)
        elif i == filled:
            bar.append(Sym.DIAMOND, style="bold white")
        else:
            bar.append(Sym.BAR_LOW, style=Palette.BORDER)

    bar.append(" SOL", style=f"dim {Palette.SOLUBLE}")
    console.print(bar)


# ---------------------------------------------------------------------------
# Per-residue heatmap
# ---------------------------------------------------------------------------

def _print_residue_heatmap(
    scores: list[float], total_length: int, width: int = 60
) -> None:
    console.print(Text("    RESIDUE CONTRIBUTION MAP", style=f"bold {Palette.ACCENT}"))
    console.print()

    if not scores:
        return

    min_s, max_s = min(scores), max(scores)
    range_s = max_s - min_s if max_s > min_s else 1.0
    normalized = [(s - min_s) / range_s for s in scores]

    display_width = min(width, len(normalized))
    chunk_size = max(1, len(normalized) // display_width)

    row = Text("    ")
    for i in range(0, len(normalized), chunk_size):
        chunk = normalized[i : i + chunk_size]
        avg = sum(chunk) / len(chunk)
        row.append(Sym.BAR_FULL, style=_heatmap_color(avg))
    console.print(row)

    row2 = Text("    ")
    for i in range(0, len(normalized), chunk_size):
        chunk = normalized[i : i + chunk_size]
        avg = sum(chunk) / len(chunk)
        row2.append(Sym.BAR_THIN, style=_heatmap_color(avg))
    console.print(row2)

    labels = Text("    ")
    labels.append("1", style=Palette.TEXT_DIM)
    gap = display_width - len(str(total_length)) - 1
    if gap > 0:
        labels.append(" " * gap)
    labels.append(str(total_length), style=Palette.TEXT_DIM)
    console.print(labels)

    legend = Text("    ")
    legend.append(Sym.BAR_FULL, style=Palette.INSOLUBLE)
    legend.append(" destabilizing  ", style=Palette.TEXT_DIM)
    legend.append(Sym.BAR_FULL, style=Palette.WARNING)
    legend.append(" neutral  ", style=Palette.TEXT_DIM)
    legend.append(Sym.BAR_FULL, style=Palette.SOLUBLE)
    legend.append(" stabilizing", style=Palette.TEXT_DIM)
    console.print(legend)


def _heatmap_color(value: float) -> str:
    if value < 0.33:
        return Palette.INSOLUBLE
    elif value < 0.66:
        return Palette.WARNING
    return Palette.SOLUBLE


# ---------------------------------------------------------------------------
# Batch results
# ---------------------------------------------------------------------------

def print_batch_results(
    results: list[SolubilityResult], sort_by: str = "score"
) -> None:
    if sort_by == "score":
        results = sorted(results, key=lambda r: r.score, reverse=True)

    console.print()
    console.print(Text("  RANKED PREDICTIONS\n", style=f"bold {Palette.ACCENT}"))

    table = Table(
        show_header=True,
        header_style=f"bold {Palette.TEXT_DIM}",
        border_style=Palette.BORDER,
        box=None, padding=(0, 2), collapse_padding=True,
    )

    table.add_column("#", style=Palette.TEXT_DIM, width=5, justify="right")
    table.add_column("Name", style=Palette.TEXT, max_width=30)
    table.add_column("Length", style=Palette.TEXT_DIM, justify="right", width=8)
    table.add_column("Score", justify="right", width=8)
    table.add_column("Label", width=12)
    table.add_column("Conf", width=8)
    table.add_column("", width=22)

    for rank, r in enumerate(results, 1):
        bar_len = 16
        filled = int(r.score * bar_len)
        mini_bar = Text()
        for j in range(bar_len):
            if j < filled:
                mini_bar.append(Sym.BAR_FULL, style=_label_color(r.label))
            else:
                mini_bar.append(Sym.BAR_LOW, style=Palette.BORDER)

        name = r.name or r.sequence[:20] + "..."

        table.add_row(
            str(rank).zfill(2),
            name,
            f"{r.length:,}",
            Text(r.score_pct, style=f"bold {_label_color(r.label)}"),
            Text(r.label.upper(), style=_label_color(r.label)),
            Text(r.confidence.upper()[:3], style=_confidence_color(r.confidence)),
            mini_bar,
        )

    console.print(Padding(table, (0, 2)))
    console.print()


def print_batch_summary(results: list[SolubilityResult]) -> None:
    n = len(results)
    n_sol = sum(1 for r in results if r.label == "soluble")
    n_insol = n - n_sol
    avg_score = sum(r.score for r in results) / n if n > 0 else 0
    high_conf = sum(1 for r in results if r.confidence == "high")

    console.print()
    _print_separator()
    console.print()
    s = Text()
    s.append("  BATCH SUMMARY\n\n", style=f"bold {Palette.ACCENT}")
    s.append(f"    Total sequences    ", style=Palette.TEXT_DIM)
    s.append(f"{n}\n", style=f"bold {Palette.TEXT}")
    s.append(f"    Soluble            ", style=Palette.TEXT_DIM)
    s.append(f"{n_sol}", style=f"bold {Palette.SOLUBLE}")
    s.append(f"  ({n_sol/n*100:.0f}%)\n" if n else "\n", style=Palette.TEXT_DIM)
    s.append(f"    Insoluble          ", style=Palette.TEXT_DIM)
    s.append(f"{n_insol}", style=f"bold {Palette.INSOLUBLE}")
    s.append(f"  ({n_insol/n*100:.0f}%)\n" if n else "\n", style=Palette.TEXT_DIM)
    s.append(f"    Average score      ", style=Palette.TEXT_DIM)
    s.append(f"{avg_score:.3f}\n", style=Palette.TEXT)
    s.append(f"    High confidence    ", style=Palette.TEXT_DIM)
    s.append(f"{high_conf}/{n}\n", style=Palette.TEXT)
    console.print(s)


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def print_comparison(
    results: list[SolubilityResult],
    references: list[SolubilityResult],
) -> None:
    console.print()
    console.print(Text("  REFERENCE COMPARISON\n", style=f"bold {Palette.ACCENT}"))
    console.print(Text("  Your protein vs known E. coli expression benchmarks.\n", style=Palette.TEXT_DIM))

    all_items = [("ref", r) for r in references] + [("usr", r) for r in results]
    all_items.sort(key=lambda x: x[1].score, reverse=True)

    table = Table(
        show_header=True,
        header_style=f"bold {Palette.TEXT_DIM}",
        border_style=Palette.BORDER,
        box=None, padding=(0, 2),
    )
    table.add_column("", width=5)
    table.add_column("Protein", max_width=25)
    table.add_column("Length", justify="right", width=8)
    table.add_column("Score", justify="right", width=8)
    table.add_column("Label", width=12)
    table.add_column("", width=22)

    for source, r in all_items:
        is_user = source == "usr"
        tag = Text(f" {Sym.ARROW_RIGHT} ", style=f"bold {Palette.ACCENT}") if is_user else Text(" REF", style=Palette.TEXT_DIM)
        name_style = f"bold {Palette.ACCENT}" if is_user else Palette.TEXT_DIM

        bar_len = 16
        filled = int(r.score * bar_len)
        mini_bar = Text()
        for j in range(bar_len):
            if j < filled:
                mini_bar.append(Sym.BAR_FULL, style=Palette.ACCENT if is_user else _label_color(r.label))
            else:
                mini_bar.append(Sym.BAR_LOW, style=Palette.BORDER)

        table.add_row(
            tag,
            Text(r.name or "query", style=name_style),
            f"{r.length:,}",
            Text(r.score_pct, style=f"bold {_label_color(r.label)}"),
            Text(r.label.upper(), style=_label_color(r.label)),
            mini_bar,
        )

    console.print(Padding(table, (0, 2)))
    console.print()


# ---------------------------------------------------------------------------
# Progress / Status
# ---------------------------------------------------------------------------

def create_progress() -> Progress:
    return Progress(
        SpinnerColumn(style=Palette.ACCENT),
        TextColumn("[bold]{task.description}"),
        BarColumn(bar_width=30, style=Palette.BORDER, complete_style=Palette.ACCENT, finished_style=Palette.ACCENT_DIM),
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

def _print_separator() -> None:
    sep = Text("  ")
    for i in range(56):
        sep.append(Sym.THICK_DASH, style=Palette.ACCENT_DIM if i % 10 < 5 else Palette.BORDER)
    console.print(sep)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _label_color(label: str) -> str:
    return Palette.SOLUBLE if label == "soluble" else Palette.INSOLUBLE

def _confidence_color(confidence: str) -> str:
    return {"high": Palette.CONFIDENCE_HIGH, "medium": Palette.CONFIDENCE_MED, "low": Palette.CONFIDENCE_LOW}.get(confidence, Palette.TEXT_DIM)
