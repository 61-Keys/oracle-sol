# ORACLE-Sol

Protein solubility prediction for the modern design pipeline.

```
oracle predict MVKVYAPASSANMSVGFDVLGAAVTPVDGALLGDVVTVEAAETFSLNNLGR...

  ◆ query

    INSOL ████████████████████████████████████◆░░░░░░░░░░░░░░ SOL

    Length  238  │  Score  72.3%  │  Label  SOLUBLE  │  Confidence  medium
```

---

## What it does

ORACLE-Sol predicts whether a protein will be soluble when expressed in *E. coli*, the most common bacterial expression system. It uses frozen [ESM2-650M](https://huggingface.co/facebook/esm2_t33_650M_UR50D) embeddings with a trained classification head, matching published SOTA performance (PLM_Sol, 2024) with a fraction of the complexity.

It fits between computational design tools and the wet lab:

```
RFdiffusion / ProteinMPNN        design sequences
        ↓
AlphaFold / ESMFold              predict structure
        ↓
ORACLE-Sol                       score solubility ← you are here
        ↓
wet lab                          express top candidates
```

## Performance

| Method | Accuracy | MCC | Source |
|--------|----------|-----|--------|
| Composition baseline | 59.8% | 0.258 | Logistic regression on AA frequencies |
| SoluProt (2021) | 67.0% | 0.40 | ProtBERT + gradient boosting |
| DSResSol (2022) | 71.0% | 0.43 | ESM-1b + ResNet |
| **ORACLE-Sol** | **73.4%** | **0.455** | **ESM2-650M + MLP (frozen)** |
| PLM_Sol (2024) | 73.0% | 0.469 | ProtT5 + biLSTM_TextCNN |
| Field ceiling | ~77% | ~0.50 | Limited by label noise in training data |

Trained on 70K proteins from the UESolDS dataset (TargetTrack-derived, real experimental outcomes).

## Install

```bash
pip install oracle-sol
```

For GPU acceleration:

```bash
pip install oracle-sol[gpu]
```

## CLI usage

```bash
# Single sequence
oracle predict MVKVYAPASSANMSVGF...

# From FASTA file (batch)
oracle predict designed_proteins.fasta --rank

# From PDB / AlphaFold output
oracle predict fold_output.pdb

# Compare against reference proteins
oracle predict SEQUENCE --compare

# Per-residue contribution heatmap
oracle predict SEQUENCE --residues

# Export to CSV
oracle predict proteins.fasta --rank --output results.csv

# Quiet mode (TSV output, pipeable)
oracle predict proteins.fasta --quiet > scores.tsv

# Model info + reference panel
oracle info
```

### Pipeline example

Score all ProteinMPNN designs and pick the top 10:

```bash
oracle predict mpnn_designs.fasta --rank --output ranked.csv
head -11 ranked.csv  # header + top 10
```

## Python API

```python
from oracle_sol import predict_solubility

# Single sequence
results = predict_solubility("MVKVYAPASS...")
print(results[0].score_pct)   # "72.3%"
print(results[0].label)       # "soluble"
print(results[0].confidence)  # "medium"

# Batch
results = predict_solubility(
    ["MVKVY...", "MKTLL...", "DAEFR..."],
    names=["design_1", "design_2", "design_3"],
)

for r in results:
    print(f"{r.name}: {r.label} ({r.score_pct})")

# Per-residue analysis
results = predict_solubility("MVKVY...", residue_level=True)
print(results[0].residue_scores)  # [0.82, 0.91, ...]
```

### SolubilityResult fields

| Field | Type | Description |
|-------|------|-------------|
| `score` | `float` | P(soluble), range 0-1 |
| `score_pct` | `str` | Score as percentage string |
| `label` | `str` | `"soluble"` or `"insoluble"` |
| `confidence` | `str` | `"high"`, `"medium"`, or `"low"` |
| `length` | `int` | Sequence length |
| `truncated` | `bool` | True if sequence exceeded 1022 aa |
| `residue_scores` | `list[float]` | Per-residue contributions (if requested) |
| `name` | `str` | Sequence identifier |

## Input formats

| Format | Extensions | Notes |
|--------|-----------|-------|
| Raw sequence | (paste directly) | Standard 20 amino acids |
| FASTA | `.fasta`, `.fa`, `.faa` | Multi-sequence supported |
| PDB | `.pdb` | Extracts sequence from ATOM records |
| CSV | `.csv` | Must have `sequence` column |

## Confidence calibration

Confidence reflects distance from the decision boundary, not model certainty:

- **High**: score > 0.90 or < 0.10 (strong signal)
- **Medium**: score 0.675-0.90 or 0.10-0.325
- **Low**: score 0.325-0.675 (near decision boundary, unreliable)

The field-wide accuracy ceiling is approximately 77% due to label noise in experimental data (TargetTrack database). This affects all solubility prediction methods equally.

## Reference proteins

ORACLE-Sol ships with a curated panel of well-characterized proteins for comparison:

| Protein | Length | Known outcome |
|---------|--------|---------------|
| GFP (A. victoria) | 238 | Soluble |
| Lysozyme (hen egg-white) | 129 | Soluble |
| Thioredoxin (E. coli) | 109 | Soluble |
| MBP (E. coli) | 370 | Soluble |
| SUMO1 (human) | 101 | Soluble |
| Insulin (human) | 110 | Insoluble |
| p53 full-length (human) | 393 | Insoluble |
| Amyloid-beta 42 (human) | 42 | Insoluble |

Use `oracle predict SEQUENCE --compare` to see where your protein ranks.

## Architecture

```
Input sequence (amino acids)
        ↓
ESM2-650M (frozen, 33 layers, 650M params)
        ↓
CLS token embedding (1280-dim)
        ↓
MLP head (1280 → 512 → 128 → 2)
        ↓
Softmax → P(soluble)
```

No fine-tuning of ESM2. No MSA computation. No structure prediction required at inference. The frozen backbone makes predictions fast and reproducible.

## How it works

ESM2 is a protein language model trained on 65M protein sequences. Its CLS token captures a global summary of sequence properties. We train a small MLP to map this representation to binary solubility labels from the UESolDS dataset (proteins expressed in E. coli with known experimental outcomes).

**Why not fine-tune ESM2?** We tried. QLoRA fine-tuning produced MCC 0.444, which is *worse* than the frozen baseline (0.455). The bottleneck is data quality, not model capacity.

**Why not combine multiple PLMs?** We tried that too. ESM2 + ProtT5 ensembles scored 0.460, barely above ESM2 alone (0.463 with 5-seed ensembling). The models are too correlated (r=0.84) to provide complementary signal.

## License

MIT

## Citation

```bibtex
@software{oracle_sol_2026,
  author = {Rath, Asutosh},
  title = {ORACLE-Sol: Protein Solubility Prediction for the Modern Design Pipeline},
  year = {2026},
  url = {https://github.com/61-Keys/oracle-sol}
}
```
