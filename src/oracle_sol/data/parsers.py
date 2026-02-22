"""
Input parsing utilities.

Handles: FASTA files, raw sequences, PDB files, AlphaFold outputs.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional


def parse_fasta(path: str | Path) -> list[tuple[str, str]]:
    """
    Parse a FASTA file into (name, sequence) pairs.

    Returns:
        List of (header, sequence) tuples.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"FASTA file not found: {path}")

    entries: list[tuple[str, str]] = []
    current_name = ""
    current_seq: list[str] = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_name or current_seq:
                    entries.append((current_name, "".join(current_seq)))
                current_name = line[1:].split()[0]  # Take first word as ID
                current_seq = []
            else:
                current_seq.append(line.upper())

    if current_name or current_seq:
        entries.append((current_name, "".join(current_seq)))

    return entries


def parse_pdb_sequence(path: str | Path) -> list[tuple[str, str]]:
    """
    Extract amino acid sequence from a PDB file (ATOM records).

    Supports standard PDB format and mmCIF. For AlphaFold outputs,
    extracts the sequence from ATOM records of chain A.

    Returns:
        List of (chain_id, sequence) tuples.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PDB file not found: {path}")

    THREE_TO_ONE = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
        "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
        "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
        "SEC": "U", "PYL": "O",
    }

    chains: dict[str, dict[int, str]] = {}

    with open(path) as f:
        for line in f:
            if line.startswith(("ATOM  ", "HETATM")):
                atom_name = line[12:16].strip()
                if atom_name != "CA":  # Only CA atoms to avoid duplicates
                    continue
                res_name = line[17:20].strip()
                chain_id = line[21].strip() or "A"
                res_seq = int(line[22:26].strip())

                if res_name in THREE_TO_ONE:
                    if chain_id not in chains:
                        chains[chain_id] = {}
                    chains[chain_id][res_seq] = THREE_TO_ONE[res_name]

    result = []
    for chain_id in sorted(chains.keys()):
        residues = chains[chain_id]
        sorted_positions = sorted(residues.keys())
        sequence = "".join(residues[pos] for pos in sorted_positions)
        result.append((chain_id, sequence))

    return result


def validate_sequence(sequence: str) -> tuple[str, list[str]]:
    """
    Validate and clean a protein sequence.

    Returns:
        (cleaned_sequence, list_of_warnings)
    """
    VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")
    EXTENDED_AA = set("ACDEFGHIKLMNPQRSTVWYUOX")

    warnings: list[str] = []
    cleaned = sequence.upper().strip()

    # Remove common artifacts
    cleaned = re.sub(r"\s+", "", cleaned)  # Whitespace
    cleaned = re.sub(r"[0-9]", "", cleaned)  # Numbers
    cleaned = cleaned.replace("*", "")  # Stop codons

    # Check for non-standard residues
    non_standard = set(cleaned) - VALID_AA
    if non_standard:
        extended_non_standard = non_standard - EXTENDED_AA
        if extended_non_standard:
            warnings.append(
                f"Unknown characters removed: {', '.join(sorted(extended_non_standard))}"
            )
            cleaned = "".join(c for c in cleaned if c in EXTENDED_AA)
        else:
            warnings.append(
                f"Non-standard residues present: {', '.join(sorted(non_standard))}"
            )

    if len(cleaned) < 10:
        warnings.append(f"Very short sequence ({len(cleaned)} aa) -- predictions may be unreliable")

    if len(cleaned) > 2500:
        warnings.append(f"Long sequence ({len(cleaned)} aa) -- will be truncated to 1022 aa")

    return cleaned, warnings


def detect_input_format(path: str | Path) -> str:
    """Detect file format from extension."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".fasta", ".fa", ".faa", ".fas"):
        return "fasta"
    elif suffix in (".pdb", ".ent"):
        return "pdb"
    elif suffix in (".cif", ".mmcif"):
        return "mmcif"
    elif suffix in (".csv", ".tsv"):
        return "csv"
    else:
        return "unknown"
