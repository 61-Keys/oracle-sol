# Deploying ORACLE-Sol

Everything you need, in order.

---

## Phase 1: Export trained weights from Colab

Open your ORACLE v5 Colab notebook and run:

```python
import torch
import torch.nn as nn
from pathlib import Path

# 1. Define the exact architecture (must match the package)
class SolubilityMLP(nn.Module):
    def __init__(self, input_dim=1280, hidden1=512, hidden2=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden2, 2),
        )
    def forward(self, x):
        return self.net(x)

# 2. Load your best trained model
#    EDIT this path to wherever your best checkpoint lives
checkpoint = torch.load(
    "/content/drive/MyDrive/oracle_v5_results/YOUR_BEST_MODEL.pt",
    map_location="cpu"
)

# If it's wrapped in a dict:
# state_dict = checkpoint["model_state_dict"]
# If it's a raw state dict:
# state_dict = checkpoint

# 3. Verify shapes
model = SolubilityMLP()
model.load_state_dict(state_dict)  # This will error if shapes mismatch
print("Shapes verified.")

# 4. Save
Path("weights").mkdir(exist_ok=True)
torch.save(state_dict, "weights/mlp_head.pt")
print(f"Saved: {Path('weights/mlp_head.pt').stat().st_size / 1024:.0f} KB")

# 5. Download to your Mac
from google.colab import files
files.download("weights/mlp_head.pt")
```

**If you don't have a saved checkpoint**, the full retrain script is in `scripts/export_weights.py`. It retrains the best config (ESM2 CLS + MLP, seed 42) in ~2 minutes on Colab GPU using your pre-extracted embeddings.

---

## Phase 2: Set up the repo on your Mac

```bash
# 1. Create the directory (or download from Claude)
cd ~/Desktop  # or wherever you work
# The oracle-sol/ folder from Claude should be here

# 2. Put the weights file in place
mkdir -p oracle-sol/weights
mv ~/Downloads/mlp_head.pt oracle-sol/weights/

# 3. Verify the structure looks right
cd oracle-sol
find . -type f | head -25
```

Expected structure:
```
oracle-sol/
  .gitignore
  LICENSE
  README.md
  pyproject.toml
  requirements.txt       # For HuggingFace Spaces
  app.py                 # Gradio web app
  weights/
    mlp_head.pt           # Your trained weights (~2.6 MB)
  src/oracle_sol/
    __init__.py
    api.py
    cli/
      __init__.py
      app.py
      display.py
    core/
      __init__.py
      predictor.py
    data/
      __init__.py
      parsers.py
      references.py
  scripts/
    export_weights.py
  tests/
    test_core.py
```

---

## Phase 3: Test locally

```bash
cd oracle-sol

# Install in dev mode
pip install -e ".[dev]"

# Run tests (no GPU required — tests don't load ESM2)
pytest tests/ -v

# Test the CLI (requires GPU or will be slow on CPU)
oracle info
oracle predict MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLK --weights weights/mlp_head.pt

# Test the Python API
python -c "
from oracle_sol import predict_solubility
r = predict_solubility('MSKGEELFTGVVPILVELDGD', weights_path='weights/mlp_head.pt')
print(f'{r[0].label}: {r[0].score_pct}')
"
```

---

## Phase 4: Push to GitHub

```bash
cd oracle-sol

# Initialize git
git init
git add .
git commit -m "ORACLE-Sol v0.6.0: protein solubility prediction

- Frozen ESM2-650M + MLP head (73.4% acc, MCC 0.455)
- Matches PLM_Sol SOTA with fraction of complexity
- CLI: oracle predict <sequence|fasta|pdb>
- Python API: predict_solubility([sequences])
- Gradio web app for HuggingFace Spaces
- Reference protein comparison panel
- Per-residue contribution analysis"

# Create the repo on GitHub
# Option A: GitHub CLI (if you have `gh` installed)
gh repo create oracle-sol --public --source=. --push

# Option B: Manual
# 1. Go to github.com/new
# 2. Create "oracle-sol" (public, no README — we have our own)
# 3. Then:
git remote add origin https://github.com/61-Keys/oracle-sol.git
git branch -M main
git push -u origin main
```

**Note on weights file:** The MLP weights (`mlp_head.pt`) are only ~2.6 MB, so they can go directly in the git repo. If they were larger (>50 MB), you'd use Git LFS.

---

## Phase 5 (optional): Deploy web app to HuggingFace Spaces

This gives you a free web-based predictor with GPU access.

```bash
# 1. Install HuggingFace CLI
pip install huggingface-hub

# 2. Log in
huggingface-cli login
# Enter your HF token from https://huggingface.co/settings/tokens

# 3. Create the Space
huggingface-cli repo create oracle-sol --type space --space-sdk gradio

# 4. Clone and push
git clone https://huggingface.co/spaces/YOUR_USERNAME/oracle-sol hf-oracle-sol
cd hf-oracle-sol

# Copy the files HF needs
cp ../oracle-sol/app.py .
cp ../oracle-sol/requirements.txt .
cp -r ../oracle-sol/weights .

# Push
git add .
git commit -m "Initial deploy"
git push
```

The Space will build and deploy automatically. Your app will be live at:
`https://huggingface.co/spaces/YOUR_USERNAME/oracle-sol`

**For GPU acceleration**, edit the Space settings on huggingface.co and select a GPU runtime (T4 is free for some plans).

---

## Phase 6 (optional): Publish to PyPI

```bash
cd oracle-sol

# Build
pip install build
python -m build

# Upload to PyPI
pip install twine
twine upload dist/*
# Enter your PyPI credentials

# Now anyone can:
# pip install oracle-sol
# oracle predict SEQUENCE --weights path/to/weights.pt
```

---

## Quick reference

| What | Command |
|------|---------|
| Predict one protein | `oracle predict MVKVYA... --weights weights/mlp_head.pt` |
| Batch from FASTA | `oracle predict designs.fasta --rank --output results.csv` |
| Compare vs references | `oracle predict MVKVYA... --compare` |
| Per-residue heatmap | `oracle predict MVKVYA... --residues` |
| Model info | `oracle info` |
| Python API | `from oracle_sol import predict_solubility` |
| Run tests | `pytest tests/ -v` |
| Run web app locally | `python app.py` |

---

## What goes where

| Platform | Purpose | Content |
|----------|---------|---------|
| **GitHub** | Source code, pip install | Everything |
| **PyPI** | `pip install oracle-sol` | Python package only |
| **HuggingFace Spaces** | Web demo with GPU | `app.py` + `requirements.txt` + `weights/` |
