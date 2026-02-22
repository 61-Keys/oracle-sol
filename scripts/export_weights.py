"""
ORACLE-Sol: Export trained MLP weights for deployment.

Run this in your Colab notebook AFTER training is complete.
It saves the MLP head weights in the exact format that
`oracle predict --weights mlp_weights.pt` expects.

Usage in Colab:
    %run scripts/export_weights.py

Or paste the relevant section into a cell.
"""

import torch
import torch.nn as nn
from pathlib import Path

# ─── 1. Define the EXACT same architecture as the package ────────────────────
# This MUST match src/oracle_sol/core/predictor.py :: SolubilityMLP

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


# ─── 2. Load your trained model from v5 training ────────────────────────────
# EDIT THIS PATH to match your best model checkpoint

CHECKPOINT_DIR = "/content/drive/MyDrive/oracle_v5_results"

# Option A: If you saved the full model state dict during training
# checkpoint = torch.load(f"{CHECKPOINT_DIR}/best_model.pt", map_location="cpu")

# Option B: If you need to reconstruct from your training code,
# load the embeddings and retrain quickly:

def export_from_training_run():
    """
    Reconstruct the best model from your v5 ablation results.
    
    Your training code saved models during the ablation. This function
    loads the best one (ESM2 CLS, seed with best validation MCC).
    """
    import glob
    
    # Find all saved models
    model_paths = glob.glob(f"{CHECKPOINT_DIR}/**/*.pt", recursive=True)
    print(f"Found {len(model_paths)} model files:")
    for p in sorted(model_paths):
        print(f"  {p}")
    
    # If you have a specific best model, load it directly:
    # best_path = f"{CHECKPOINT_DIR}/ablation/esm2_cls_seed42/best_model.pt"
    
    return model_paths


def retrain_best_model():
    """
    Quick retrain of the best configuration if no saved weights exist.
    
    Config: ESM2-650M CLS pooling, MLP (1280->512->128->2), seed=42
    This takes ~2 minutes on Colab GPU.
    """
    import numpy as np
    from torch.utils.data import DataLoader, TensorDataset
    
    # Load pre-extracted embeddings
    print("Loading ESM2 embeddings...")
    data = torch.load(
        f"{CHECKPOINT_DIR}/embeddings/esm2_650m_embeddings.pt",
        map_location="cpu",
    )
    
    embeddings = data["embeddings"]  # dict: protein_id -> embedding tensor
    
    # Load labels
    import pandas as pd
    train_df = pd.read_csv("/content/oracle_v5/data/processed/train.csv")
    val_df = pd.read_csv("/content/oracle_v5/data/processed/val.csv")
    test_df = pd.read_csv("/content/oracle_v5/data/processed/test.csv")
    
    def make_dataset(df):
        X, y = [], []
        for _, row in df.iterrows():
            pid = str(row.get("protein_id", row.get("id", row.name)))
            if pid in embeddings:
                emb = embeddings[pid]
                # CLS pooling: first token
                if emb.dim() == 2:
                    emb = emb[0]  # CLS token
                X.append(emb)
                y.append(int(row["label"]))  # 1=soluble, 0=insoluble
        return torch.stack(X), torch.tensor(y, dtype=torch.long)
    
    X_train, y_train = make_dataset(train_df)
    X_val, y_val = make_dataset(val_df)
    X_test, y_test = make_dataset(test_df)
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Train
    torch.manual_seed(42)
    model = SolubilityMLP(input_dim=X_train.shape[1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    criterion = nn.CrossEntropyLoss()
    
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=256, shuffle=True,
    )
    
    best_val_mcc = -1
    best_state = None
    
    for epoch in range(30):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = criterion(model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        
        # Validate
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val.to(device))
            val_preds = val_logits.argmax(dim=1).cpu()
            
            # MCC calculation
            tp = ((val_preds == 1) & (y_val == 1)).sum().item()
            tn = ((val_preds == 0) & (y_val == 0)).sum().item()
            fp = ((val_preds == 1) & (y_val == 0)).sum().item()
            fn = ((val_preds == 0) & (y_val == 1)).sum().item()
            
            denom = ((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)) ** 0.5
            mcc = (tp*tn - fp*fn) / denom if denom > 0 else 0
            acc = (tp + tn) / (tp + tn + fp + fn)
        
        if mcc > best_val_mcc:
            best_val_mcc = mcc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  Epoch {epoch+1:2d}: MCC={mcc:.4f}  Acc={acc:.4f}  [new best]")
        elif epoch % 5 == 0:
            print(f"  Epoch {epoch+1:2d}: MCC={mcc:.4f}  Acc={acc:.4f}")
    
    # Final test evaluation
    model.load_state_dict(best_state)
    model.eval()
    model.to(device)
    with torch.no_grad():
        test_logits = model(X_test.to(device))
        test_preds = test_logits.argmax(dim=1).cpu()
        tp = ((test_preds == 1) & (y_test == 1)).sum().item()
        tn = ((test_preds == 0) & (y_test == 0)).sum().item()
        fp = ((test_preds == 1) & (y_test == 0)).sum().item()
        fn = ((test_preds == 0) & (y_test == 1)).sum().item()
        denom = ((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)) ** 0.5
        test_mcc = (tp*tn - fp*fn) / denom if denom > 0 else 0
        test_acc = (tp + tn) / (tp + tn + fp + fn)
    
    print(f"\n  Final test: MCC={test_mcc:.4f}  Acc={test_acc:.4f}")
    
    return best_state


# ─── 3. Export ───────────────────────────────────────────────────────────────

def export_weights(state_dict, output_path="weights/mlp_head.pt"):
    """Save MLP state dict in the format the package expects."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    
    # Verify shape matches expected architecture
    expected_shapes = {
        "net.0.weight": (512, 1280),
        "net.0.bias": (512,),
        "net.3.weight": (128, 512),
        "net.3.bias": (128,),
        "net.6.weight": (2, 128),
        "net.6.bias": (2,),
    }
    
    for key, expected_shape in expected_shapes.items():
        if key not in state_dict:
            raise ValueError(f"Missing key: {key}")
        actual_shape = tuple(state_dict[key].shape)
        if actual_shape != expected_shape:
            raise ValueError(
                f"Shape mismatch for {key}: "
                f"expected {expected_shape}, got {actual_shape}"
            )
    
    torch.save(state_dict, output)
    size_mb = output.stat().st_size / (1024 * 1024)
    print(f"\n  Weights saved to: {output}")
    print(f"  File size: {size_mb:.2f} MB")
    print(f"  Keys: {list(state_dict.keys())}")
    
    # Also save to Google Drive for persistence
    drive_path = f"/content/drive/MyDrive/oracle_v5_results/deploy/mlp_head.pt"
    Path(drive_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, drive_path)
    print(f"  Backup saved to: {drive_path}")
    
    return output


# ─── Run ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  ORACLE-Sol Weight Export")
    print("=" * 60)
    print()
    
    # Try to find existing saved weights first
    paths = export_from_training_run()
    
    if not paths:
        print("\n  No saved models found. Retraining best config...")
        state = retrain_best_model()
    else:
        print(f"\n  Loading best model from: {paths[0]}")
        # EDIT: pick the right path from the list above
        checkpoint = torch.load(paths[0], map_location="cpu")
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state = checkpoint["state_dict"]
        else:
            state = checkpoint  # Assume it's a raw state dict
    
    output = export_weights(state)
    
    print("\n  Done. Use with:")
    print(f"    oracle predict SEQUENCE --weights {output}")
    print()
