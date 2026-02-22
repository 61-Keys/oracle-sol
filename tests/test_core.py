"""Tests for ORACLE-Sol core functionality."""

import pytest


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

class TestValidateSequence:
    def test_valid_sequence(self):
        from oracle_sol.data.parsers import validate_sequence
        seq, warnings = validate_sequence("MVKVYAPASSANMSVGF")
        assert seq == "MVKVYAPASSANMSVGF"
        assert len(warnings) == 0

    def test_lowercase_normalized(self):
        from oracle_sol.data.parsers import validate_sequence
        seq, warnings = validate_sequence("mvkvyapass")
        assert seq == "MVKVYAPASS"

    def test_whitespace_stripped(self):
        from oracle_sol.data.parsers import validate_sequence
        seq, _ = validate_sequence("MVK VYA\nPASS")
        assert seq == "MVKVYAPASS"

    def test_numbers_removed(self):
        from oracle_sol.data.parsers import validate_sequence
        seq, _ = validate_sequence("1MVKVYA2PASS3")
        assert seq == "MVKVYAPASS"

    def test_stop_codon_removed(self):
        from oracle_sol.data.parsers import validate_sequence
        seq, _ = validate_sequence("MVKVYA*")
        assert seq == "MVKVYA"

    def test_short_sequence_warning(self):
        from oracle_sol.data.parsers import validate_sequence
        _, warnings = validate_sequence("MVKV")
        assert any("Very short" in w for w in warnings)

    def test_long_sequence_warning(self):
        from oracle_sol.data.parsers import validate_sequence
        _, warnings = validate_sequence("A" * 3000)
        assert any("Long sequence" in w for w in warnings)

    def test_non_standard_residues(self):
        from oracle_sol.data.parsers import validate_sequence
        seq, warnings = validate_sequence("MVKVUAPASS")  # U = selenocysteine
        assert "U" in seq
        assert any("Non-standard" in w for w in warnings)


class TestFastaParser:
    def test_parse_single_entry(self, tmp_path):
        from oracle_sol.data.parsers import parse_fasta
        fasta = tmp_path / "test.fasta"
        fasta.write_text(">protein1\nMVKVYAPASS\nANMSVGF\n")
        entries = parse_fasta(fasta)
        assert len(entries) == 1
        assert entries[0][0] == "protein1"
        assert entries[0][1] == "MVKVYAPASSANMSVGF"

    def test_parse_multiple_entries(self, tmp_path):
        from oracle_sol.data.parsers import parse_fasta
        fasta = tmp_path / "test.fasta"
        fasta.write_text(">prot1\nMVKVY\n>prot2\nDAEFR\n")
        entries = parse_fasta(fasta)
        assert len(entries) == 2

    def test_missing_file(self):
        from oracle_sol.data.parsers import parse_fasta
        with pytest.raises(FileNotFoundError):
            parse_fasta("/nonexistent/file.fasta")


class TestInputDetection:
    def test_fasta_detection(self):
        from oracle_sol.data.parsers import detect_input_format
        assert detect_input_format("proteins.fasta") == "fasta"
        assert detect_input_format("proteins.fa") == "fasta"
        assert detect_input_format("proteins.faa") == "fasta"

    def test_pdb_detection(self):
        from oracle_sol.data.parsers import detect_input_format
        assert detect_input_format("structure.pdb") == "pdb"

    def test_csv_detection(self):
        from oracle_sol.data.parsers import detect_input_format
        assert detect_input_format("data.csv") == "csv"

    def test_unknown_format(self):
        from oracle_sol.data.parsers import detect_input_format
        assert detect_input_format("file.xyz") == "unknown"


# ---------------------------------------------------------------------------
# Predictor (unit tests, no model loading)
# ---------------------------------------------------------------------------

class TestSolubilityResult:
    def test_score_pct(self):
        from oracle_sol.core.predictor import SolubilityResult
        r = SolubilityResult(
            sequence="MVKVY",
            score=0.723,
            label="soluble",
            confidence="medium",
            length=5,
        )
        assert r.score_pct == "72.3%"

    def test_to_dict(self):
        from oracle_sol.core.predictor import SolubilityResult
        r = SolubilityResult(
            sequence="MVKVY",
            score=0.723,
            label="soluble",
            confidence="medium",
            length=5,
            name="test_protein",
        )
        d = r.to_dict()
        assert d["name"] == "test_protein"
        assert d["score"] == 0.723
        assert d["label"] == "soluble"


class TestConfidenceClassification:
    def test_high_confidence_soluble(self):
        from oracle_sol.core.predictor import OraclePredictor
        assert OraclePredictor._classify_confidence(0.95) == "high"

    def test_high_confidence_insoluble(self):
        from oracle_sol.core.predictor import OraclePredictor
        assert OraclePredictor._classify_confidence(0.05) == "high"

    def test_low_confidence_boundary(self):
        from oracle_sol.core.predictor import OraclePredictor
        assert OraclePredictor._classify_confidence(0.50) == "low"

    def test_medium_confidence(self):
        from oracle_sol.core.predictor import OraclePredictor
        assert OraclePredictor._classify_confidence(0.75) == "medium"


class TestMLP:
    def test_forward_shape(self):
        import torch
        from oracle_sol.core.predictor import SolubilityMLP

        mlp = SolubilityMLP(input_dim=1280)
        x = torch.randn(2, 1280)
        out = mlp(x)
        assert out.shape == (2, 2)


# ---------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------

class TestReferences:
    def test_reference_panel_not_empty(self):
        from oracle_sol.data.references import REFERENCE_PANEL
        assert len(REFERENCE_PANEL) >= 5

    def test_has_soluble_and_insoluble(self):
        from oracle_sol.data.references import REFERENCE_PANEL
        soluble = [r for r in REFERENCE_PANEL if r.known_soluble]
        insoluble = [r for r in REFERENCE_PANEL if not r.known_soluble]
        assert len(soluble) >= 2
        assert len(insoluble) >= 2

    def test_sequences_are_valid(self):
        from oracle_sol.data.references import REFERENCE_PANEL
        valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
        for ref in REFERENCE_PANEL:
            assert len(ref.sequence) > 0
            non_standard = set(ref.sequence) - valid_aa
            assert len(non_standard) == 0, f"{ref.name} has non-standard AAs: {non_standard}"
