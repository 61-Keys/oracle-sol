"""
ORACLE-Sol CLI — Claude Code-inspired terminal interface.

Usage:
    oracle predict MVKVYAPASS...           # Single sequence
    oracle predict proteins.fasta          # Batch from FASTA
    oracle predict structure.pdb           # From PDB file
    oracle predict proteins.fasta --rank   # Ranked batch output
    oracle predict SEQUENCE --compare      # Compare against references
    oracle predict SEQUENCE --residues     # Per-residue heatmap
    oracle info                            # Show model info + references
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from oracle_sol.cli.display import (
    Palette,
    Sym,
    console,
    create_progress,
    print_banner,
    print_batch_results,
    print_batch_summary,
    print_comparison,
    print_error,
    print_result,
    print_status,
    print_success,
    print_system_status,
    print_warning,
)
from oracle_sol.core.predictor import OraclePredictor, SolubilityResult
from oracle_sol.data.parsers import (
    detect_input_format,
    parse_fasta,
    parse_pdb_sequence,
    validate_sequence,
)
from oracle_sol.data.references import REFERENCE_PANEL

app = typer.Typer(
    name="oracle",
    help="Protein solubility prediction for the modern design pipeline.",
    add_completion=False,
    no_args_is_help=True,
    rich_markup_mode="rich",
)


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

_predictor: Optional[OraclePredictor] = None


def _get_predictor(
    weights: Optional[str] = None,
    device: Optional[str] = None,
) -> OraclePredictor:
    global _predictor
    if _predictor is None:
        _predictor = OraclePredictor(
            weights_path=weights,
            device=device,
        )
    return _predictor


# ---------------------------------------------------------------------------
# oracle predict
# ---------------------------------------------------------------------------

@app.command()
def predict(
    input_val: str = typer.Argument(
        ...,
        help="Amino acid sequence, FASTA file, or PDB file.",
    ),
    compare: bool = typer.Option(
        False, "--compare", "-c",
        help="Compare against reference proteins (GFP, insulin, etc.)",
    ),
    residues: bool = typer.Option(
        False, "--residues", "-r",
        help="Show per-residue solubility contribution heatmap.",
    ),
    rank: bool = typer.Option(
        False, "--rank",
        help="Display batch results as a ranked table.",
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o",
        help="Save results to CSV file.",
    ),
    weights: Optional[str] = typer.Option(
        None, "--weights", "-w",
        help="Path to trained MLP weights (.pt file).",
    ),
    device: Optional[str] = typer.Option(
        None, "--device", "-d",
        help="Compute device: cuda, mps, or cpu.",
    ),
    batch_size: int = typer.Option(
        4, "--batch-size", "-b",
        help="Inference batch size.",
    ),
    quiet: bool = typer.Option(
        False, "--quiet", "-q",
        help="Minimal output (scores only).",
    ),
) -> None:
    """Predict protein solubility from sequence, FASTA, or PDB."""

    if not quiet:
        print_banner()

    # --- Resolve input type ------------------------------------------------
    input_path = Path(input_val)
    sequences: list[str] = []
    names: list[str] = []

    if input_path.exists() and input_path.is_file():
        fmt = detect_input_format(input_path)
        if fmt == "fasta":
            if not quiet:
                print_status(f"Reading FASTA: {input_path.name}")
            entries = parse_fasta(input_path)
            for name, seq in entries:
                clean, warnings = validate_sequence(seq)
                if warnings and not quiet:
                    for w in warnings:
                        print_warning(f"{name}: {w}")
                if clean:
                    names.append(name)
                    sequences.append(clean)

        elif fmt == "pdb":
            if not quiet:
                print_status(f"Reading PDB: {input_path.name}")
            chains = parse_pdb_sequence(input_path)
            for chain_id, seq in chains:
                clean, warnings = validate_sequence(seq)
                if warnings and not quiet:
                    for w in warnings:
                        print_warning(f"Chain {chain_id}: {w}")
                if clean:
                    names.append(f"{input_path.stem}_chain{chain_id}")
                    sequences.append(clean)

        elif fmt == "csv":
            if not quiet:
                print_status(f"Reading CSV: {input_path.name}")
            import pandas as pd
            df = pd.read_csv(input_path)
            # Auto-detect columns
            seq_col = None
            name_col = None
            for col in df.columns:
                cl = col.lower()
                if cl in ("sequence", "seq", "protein_sequence", "aa_sequence"):
                    seq_col = col
                elif cl in ("name", "id", "protein", "protein_name", "identifier"):
                    name_col = col
            if seq_col is None:
                print_error("CSV must have a column named 'sequence' or 'seq'.")
                raise typer.Exit(1)
            for idx, row in df.iterrows():
                clean, warnings = validate_sequence(str(row[seq_col]))
                if clean:
                    n = str(row[name_col]) if name_col else f"seq_{idx}"
                    names.append(n)
                    sequences.append(clean)

        else:
            print_error(f"Unsupported file format: {input_path.suffix}")
            print_status("Supported: .fasta, .fa, .pdb, .csv")
            raise typer.Exit(1)
    else:
        # Treat as raw sequence
        clean, warnings = validate_sequence(input_val)
        if warnings and not quiet:
            for w in warnings:
                print_warning(w)
        if not clean:
            print_error("Invalid input: not a file path or valid amino acid sequence.")
            raise typer.Exit(1)
        sequences.append(clean)
        names.append("query")

    if not sequences:
        print_error("No valid sequences found in input.")
        raise typer.Exit(1)

    if not quiet:
        print_status(f"Loaded {len(sequences)} sequence{'s' if len(sequences) > 1 else ''}")
        console.print()

    # --- Load model --------------------------------------------------------
    predictor = _get_predictor(weights=weights, device=device)

    if not quiet:
        print_status("Loading ESM2-650M...")
    predictor._ensure_loaded()

    if not quiet:
        weights_source = getattr(predictor, '_weights_source', 'unknown')
        print_system_status(str(predictor._device), weights_source)

    # --- Run predictions ---------------------------------------------------
    all_sequences = list(sequences)
    all_names = list(names)

    # If comparing, also predict reference proteins
    ref_results: list[SolubilityResult] = []
    if compare:
        ref_seqs = [ref.sequence for ref in REFERENCE_PANEL]
        ref_names = [ref.name for ref in REFERENCE_PANEL]
        all_sequences.extend(ref_seqs)
        all_names.extend(ref_names)

    if not quiet and len(all_sequences) > 1:
        progress = create_progress()
        with progress:
            task = progress.add_task(
                "Predicting...",
                total=len(all_sequences),
            )
            results: list[SolubilityResult] = []
            for i in range(0, len(all_sequences), batch_size):
                batch_s = all_sequences[i : i + batch_size]
                batch_n = all_names[i : i + batch_size]
                batch_r = predictor.predict(
                    batch_s,
                    names=batch_n,
                    batch_size=batch_size,
                    residue_level=residues,
                )
                results.extend(batch_r)
                progress.advance(task, len(batch_s))
        console.print()
    else:
        results = predictor.predict(
            all_sequences,
            names=all_names,
            batch_size=batch_size,
            residue_level=residues,
        )

    # Split user results from reference results
    user_results = results[: len(sequences)]
    if compare:
        ref_results = results[len(sequences) :]

    # --- Display results ---------------------------------------------------
    if quiet:
        # Minimal output: name \t score \t label
        for r in user_results:
            print(f"{r.name or 'query'}\t{r.score:.4f}\t{r.label}")
    elif len(user_results) == 1 and not rank:
        # Single sequence: detailed view
        print_result(user_results[0], show_residues=residues)
    elif rank or len(user_results) > 1:
        # Batch: ranked table
        print_batch_results(user_results)
        print_batch_summary(user_results)

    # --- Comparison view ---------------------------------------------------
    if compare and not quiet:
        print_comparison(user_results, ref_results)

    # --- CSV export --------------------------------------------------------
    if output:
        _export_csv(user_results, output)
        if not quiet:
            print_success(f"Results saved to {output}")
            console.print()


# ---------------------------------------------------------------------------
# oracle info
# ---------------------------------------------------------------------------

@app.command()
def info() -> None:
    """Display model information and reference proteins."""

    print_banner()

    from rich.text import Text
    from rich.table import Table
    from rich.padding import Padding

    # Model card
    card = Text()
    card.append("  Model\n\n", style=f"bold {Palette.TEXT}")
    card.append("    Backbone       ", style=Palette.TEXT_DIM)
    card.append("ESM2-650M (frozen)\n", style=Palette.TEXT)
    card.append("    Head           ", style=Palette.TEXT_DIM)
    card.append("MLP (1280 > 512 > 128 > 2)\n", style=Palette.TEXT)
    card.append("    Pooling        ", style=Palette.TEXT_DIM)
    card.append("CLS token\n", style=Palette.TEXT)
    card.append("    Training data  ", style=Palette.TEXT_DIM)
    card.append("UESolDS (70K E. coli proteins)\n", style=Palette.TEXT)
    card.append("    Test MCC       ", style=Palette.TEXT_DIM)
    card.append("0.455 +/- 0.004", style=f"bold {Palette.ACCENT}")
    card.append("  (5-seed average)\n", style=Palette.TEXT_DIM)
    card.append("    Test Accuracy  ", style=Palette.TEXT_DIM)
    card.append("73.4%\n", style=Palette.TEXT)
    card.append("    SOTA (PLM_Sol) ", style=Palette.TEXT_DIM)
    card.append("73.0% Acc, 0.469 MCC\n", style=Palette.TEXT)
    card.append("\n")
    card.append("    ", style="")
    card.append("Note: ", style=f"bold {Palette.WARNING}")
    card.append(
        "Field ceiling is ~77% accuracy / ~0.50 MCC due to\n"
        "    label noise in experimental data. All methods plateau here.\n",
        style=Palette.TEXT_DIM,
    )
    console.print(card)

    console.print(
        Text(
            f"  {Sym.DASH * 56}\n",
            style=Palette.BORDER,
        )
    )

    # Reference panel
    console.print(Text("  Reference proteins\n", style=f"bold {Palette.TEXT}"))

    table = Table(
        show_header=True,
        header_style=f"bold {Palette.TEXT_DIM}",
        border_style=Palette.BORDER,
        box=None,
        padding=(0, 2),
    )

    table.add_column("Protein", max_width=30)
    table.add_column("Length", justify="right", width=8)
    table.add_column("Known", width=12)
    table.add_column("Notes", max_width=50)

    for ref in REFERENCE_PANEL:
        known_style = Palette.SOLUBLE if ref.known_soluble else Palette.INSOLUBLE
        known_label = "soluble" if ref.known_soluble else "insoluble"
        table.add_row(
            Text(ref.name, style=Palette.TEXT),
            str(len(ref.sequence)),
            Text(known_label, style=known_style),
            Text(ref.notes, style=Palette.TEXT_DIM),
        )

    console.print(Padding(table, (0, 2)))
    console.print()


# ---------------------------------------------------------------------------
# oracle version
# ---------------------------------------------------------------------------

@app.command()
def version() -> None:
    """Show version."""
    from oracle_sol import __version__
    console.print(f"oracle-sol {__version__}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _export_csv(results: list[SolubilityResult], path: str) -> None:
    """Export results to CSV."""
    fieldnames = [
        "rank", "name", "length", "score", "label",
        "confidence", "truncated",
    ]
    sorted_results = sorted(results, key=lambda r: r.score, reverse=True)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, r in enumerate(sorted_results, 1):
            writer.writerow({
                "rank": rank,
                "name": r.name or "",
                "length": r.length,
                "score": round(r.score, 4),
                "label": r.label,
                "confidence": r.confidence,
                "truncated": r.truncated,
            })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
