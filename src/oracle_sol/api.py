"""
Public API for programmatic usage.

Usage:
    from oracle_sol import predict_solubility

    results = predict_solubility([
        "MVKVYAPASS...",
        "MKTLLLTLVV...",
    ])

    for r in results:
        print(f"{r.label}: {r.score_pct} confidence={r.confidence}")
"""

from __future__ import annotations

from typing import Optional, Sequence

from oracle_sol.core.predictor import OraclePredictor, SolubilityResult

# Module-level singleton for convenience
_default_predictor: Optional[OraclePredictor] = None


def predict_solubility(
    sequences: str | Sequence[str],
    names: Optional[Sequence[str]] = None,
    weights_path: Optional[str] = None,
    device: Optional[str] = None,
    batch_size: int = 4,
    residue_level: bool = False,
) -> list[SolubilityResult]:
    """
    Predict protein solubility from amino acid sequences.

    Args:
        sequences: Single sequence string or list of sequences.
        names: Optional identifiers for each sequence.
        weights_path: Path to trained MLP weights (.pt).
        device: Compute device (cuda, mps, cpu). Auto-detected if None.
        batch_size: Inference batch size.
        residue_level: If True, compute per-residue contribution scores.

    Returns:
        List of SolubilityResult objects with .score, .label, .confidence, etc.

    Example:
        >>> results = predict_solubility("MVKVYAPASSANMSVGF...")
        >>> results[0].score_pct
        '78.3%'
        >>> results[0].label
        'soluble'
    """
    global _default_predictor

    if isinstance(sequences, str):
        sequences = [sequences]

    if _default_predictor is None or weights_path is not None:
        _default_predictor = OraclePredictor(
            weights_path=weights_path,
            device=device,
        )

    return _default_predictor.predict(
        sequences=sequences,
        names=names,
        batch_size=batch_size,
        residue_level=residue_level,
    )
