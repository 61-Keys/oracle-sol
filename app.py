"""
ORACLE-Sol — HuggingFace Spaces App

Deploy: huggingface-cli repo create oracle-sol --type space --space-sdk gradio
Then push this file as app.py to the Space.

Requires: torch, transformers, gradio, pandas
"""

import os
import csv
import io
import tempfile
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ─── Model ───────────────────────────────────────────────────────────────────

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


class Predictor:
    MAX_LEN = 1022

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.esm_model = None
        self.esm_tokenizer = None
        self.mlp = None
        self._loaded = False

    def load(self):
        if self._loaded:
            return
        from transformers import AutoModel, AutoTokenizer

        print(f"Loading ESM2-650M on {self.device}...")
        model_name = "facebook/esm2_t33_650M_UR50D"
        self.esm_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.esm_model = AutoModel.from_pretrained(model_name).eval().to(self.device)

        self.mlp = SolubilityMLP(input_dim=1280)
        weights_path = Path("weights/mlp_head.pt")
        if weights_path.exists():
            state = torch.load(weights_path, map_location=self.device, weights_only=True)
            self.mlp.load_state_dict(state)
            print("MLP weights loaded.")
        else:
            print("WARNING: No weights found. Using untrained MLP.")
        self.mlp.eval().to(self.device)
        self._loaded = True
        print("Model ready.")

    def predict_single(self, sequence: str) -> dict:
        self.load()
        seq = sequence.upper().strip().replace(" ", "").replace("\n", "")
        if len(seq) < 5:
            return {"error": "Sequence too short (min 5 residues)."}

        truncated = len(seq) > self.MAX_LEN
        if truncated:
            seq = seq[:self.MAX_LEN]

        encoded = self.esm_tokenizer(
            [seq], return_tensors="pt", padding=True,
            truncation=True, max_length=self.MAX_LEN + 2,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.esm_model(input_ids=input_ids, attention_mask=attention_mask)
            cls = outputs.last_hidden_state[:, 0, :].float()
            logits = self.mlp(cls)
            probs = torch.softmax(logits, dim=-1)
            score = probs[0, 1].item()

        conf_dist = abs(score - 0.5) * 2
        confidence = "high" if conf_dist >= 0.6 else ("medium" if conf_dist >= 0.3 else "low")

        return {
            "score": score,
            "label": "soluble" if score >= 0.5 else "insoluble",
            "confidence": confidence,
            "length": len(seq),
            "truncated": truncated,
        }

    def predict_batch(self, sequences: list[tuple[str, str]]) -> list[dict]:
        results = []
        for name, seq in sequences:
            r = self.predict_single(seq)
            r["name"] = name
            results.append(r)
        return sorted(results, key=lambda x: x.get("score", 0), reverse=True)


predictor = Predictor()

# ─── Reference proteins ─────────────────────────────────────────────────────

REFERENCES = {
    "GFP (A. victoria)": "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK",
    "Lysozyme (G. gallus)": "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL",
    "Thioredoxin (E. coli)": "MSDKIIHLTDDSFDTDVLKADGAILVDFWAEWCGPCKMIAPILDEIADEYQGKLTVAKLNIDQNPGTAPKYGIRGIPTLLLFKNGEVAATKVGALSKGQLKEFLDANLA",
    "Insulin (H. sapiens)": "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN",
    "Amyloid-beta 42": "DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA",
}

# ─── Parsing helpers ─────────────────────────────────────────────────────────

def parse_fasta_text(text):
    entries = []
    name, seqs = "", []
    for line in text.strip().split("\n"):
        if line.startswith(">"):
            if name:
                entries.append((name, "".join(seqs)))
            name = line[1:].split()[0]
            seqs = []
        else:
            seqs.append(line.strip().upper())
    if name:
        entries.append((name, "".join(seqs)))
    return entries


# ─── Gradio handlers ────────────────────────────────────────────────────────

def predict_handler(text_input, file_input):
    """Main prediction handler."""
    sequences = []

    # Parse file upload
    if file_input is not None:
        content = Path(file_input).read_text()
        if content.strip().startswith(">"):
            sequences = parse_fasta_text(content)
        else:
            sequences = [("query", content.strip())]

    # Parse text input
    elif text_input and text_input.strip():
        text = text_input.strip()
        if text.startswith(">"):
            sequences = parse_fasta_text(text)
        else:
            sequences = [("query", text)]
    else:
        return "Paste a sequence or upload a FASTA file.", None, None

    if not sequences:
        return "No valid sequences found.", None, None

    # Run predictions
    results = predictor.predict_batch(sequences)

    # Format single result
    if len(results) == 1:
        r = results[0]
        if "error" in r:
            return r["error"], None, None

        summary = (
            f"## Prediction: **{r['label'].upper()}**\n\n"
            f"| Metric | Value |\n|--------|-------|\n"
            f"| Score | **{r['score']*100:.1f}%** |\n"
            f"| Confidence | {r['confidence']} |\n"
            f"| Length | {r['length']} aa |\n"
            f"| Truncated | {'Yes' if r['truncated'] else 'No'} |\n"
        )
        return summary, None, None

    # Format batch results
    rows = []
    for i, r in enumerate(results, 1):
        if "error" in r:
            continue
        rows.append({
            "Rank": i,
            "Name": r.get("name", f"seq_{i}"),
            "Length": r["length"],
            "Score": f"{r['score']*100:.1f}%",
            "Label": r["label"],
            "Confidence": r["confidence"],
        })

    df = pd.DataFrame(rows)

    n_sol = sum(1 for r in results if r.get("label") == "soluble")
    summary = (
        f"## Batch Results: {len(results)} sequences\n\n"
        f"**Soluble:** {n_sol} / {len(results)} "
        f"({n_sol/len(results)*100:.0f}%)\n\n"
    )

    # CSV download
    csv_buffer = io.StringIO()
    pd.DataFrame([{
        "rank": i+1, "name": r.get("name", ""), "length": r["length"],
        "score": round(r["score"], 4), "label": r["label"], "confidence": r["confidence"],
    } for i, r in enumerate(results) if "error" not in r]).to_csv(csv_buffer, index=False)

    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w")
    tmp.write(csv_buffer.getvalue())
    tmp.close()

    return summary, df, tmp.name


def compare_handler(text_input):
    """Compare user sequence against reference panel."""
    if not text_input or not text_input.strip():
        return "Paste a sequence to compare."

    seq = text_input.strip()
    if seq.startswith(">"):
        entries = parse_fasta_text(seq)
        if entries:
            seq = entries[0][1]

    user_result = predictor.predict_single(seq)
    if "error" in user_result:
        return user_result["error"]

    # Predict references
    rows = []
    for name, ref_seq in REFERENCES.items():
        r = predictor.predict_single(ref_seq)
        rows.append((name, r["score"], r["label"], False))

    rows.append(("YOUR PROTEIN", user_result["score"], user_result["label"], True))
    rows.sort(key=lambda x: x[1], reverse=True)

    lines = ["## Reference Comparison\n"]
    lines.append("| | Protein | Score | Label |")
    lines.append("|---|---------|-------|-------|")
    for name, score, label, is_user in rows:
        marker = ">>>" if is_user else ""
        bold = "**" if is_user else ""
        lines.append(
            f"| {marker} | {bold}{name}{bold} | "
            f"{bold}{score*100:.1f}%{bold} | {label} |"
        )

    return "\n".join(lines)


# ─── Gradio interface ────────────────────────────────────────────────────────

DESCRIPTION = """
# ORACLE-Sol

Predict whether your protein will be soluble when expressed in *E. coli*.

Uses frozen **ESM2-650M** embeddings + trained MLP head. Matches published SOTA 
(PLM_Sol 2024: 73% Acc, 0.469 MCC) with minimal complexity.

**Pipeline position:** RFdiffusion / ProteinMPNN  -->  AlphaFold  -->  **ORACLE-Sol**  -->  wet lab
"""

with gr.Blocks(
    title="ORACLE-Sol",
    theme=gr.themes.Base(
        primary_hue=gr.themes.colors.emerald,
        neutral_hue=gr.themes.colors.zinc,
        font=gr.themes.GoogleFont("Manrope"),
        font_mono=gr.themes.GoogleFont("Fira Code"),
    ),
) as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Tabs():
        with gr.TabItem("Predict"):
            with gr.Row():
                with gr.Column(scale=1):
                    text_in = gr.Textbox(
                        label="Sequence or FASTA",
                        placeholder="Paste amino acid sequence or multi-entry FASTA...",
                        lines=8,
                    )
                    file_in = gr.File(label="Or upload FASTA / CSV", file_types=[".fasta", ".fa", ".faa", ".csv"])
                    predict_btn = gr.Button("Predict solubility", variant="primary")

                with gr.Column(scale=1):
                    output_md = gr.Markdown(label="Result")
                    output_table = gr.Dataframe(label="Ranked results", visible=True)
                    output_csv = gr.File(label="Download CSV")

            predict_btn.click(
                predict_handler,
                inputs=[text_in, file_in],
                outputs=[output_md, output_table, output_csv],
            )

            gr.Examples(
                examples=[
                    ["MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"],
                    ["DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA"],
                ],
                inputs=[text_in],
                label="Try these",
            )

        with gr.TabItem("Compare"):
            compare_in = gr.Textbox(
                label="Your sequence",
                placeholder="Paste sequence to compare against reference proteins...",
                lines=5,
            )
            compare_btn = gr.Button("Compare against references", variant="primary")
            compare_out = gr.Markdown()

            compare_btn.click(compare_handler, inputs=[compare_in], outputs=[compare_out])

    gr.Markdown(
        "---\n"
        "*Accuracy ceiling ~77% across all methods (label noise in TargetTrack data). "
        "Model: ESM2-650M frozen + MLP. Training data: 70K proteins from UESolDS.*\n\n"
        "[GitHub](https://github.com/61-Keys/oracle-sol) | "
        "[Paper context](https://github.com/61-Keys/oracle-sol#performance)"
    )


if __name__ == "__main__":
    demo.launch()
