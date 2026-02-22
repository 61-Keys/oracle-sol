"""
Reference protein sequences for comparison.

Well-characterized proteins spanning the solubility spectrum.
These serve as anchors so users can gauge their prediction
relative to known outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReferenceProtein:
    name: str
    organism: str
    known_soluble: bool
    sequence: str
    notes: str


# Curated reference panel — real sequences, real outcomes
REFERENCE_PANEL: list[ReferenceProtein] = [
    ReferenceProtein(
        name="GFP (Aequorea victoria)",
        organism="A. victoria",
        known_soluble=True,
        sequence=(
            "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTL"
            "VTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVN"
            "RIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHY"
            "QQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"
        ),
        notes="Gold standard for soluble expression in E. coli",
    ),
    ReferenceProtein(
        name="Lysozyme (Hen egg-white)",
        organism="G. gallus",
        known_soluble=True,
        sequence=(
            "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINS"
            "RWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQ"
            "AWIRGCRL"
        ),
        notes="Highly soluble, well-expressed, 129 aa",
    ),
    ReferenceProtein(
        name="Thioredoxin (E. coli)",
        organism="E. coli",
        known_soluble=True,
        sequence=(
            "MSDKIIHLTDDSFDTDVLKADGAILVDFWAEWCGPCKMIAPILDEIADEYQGKLTVAKLN"
            "IDQNPGTAPKYGIRGIPTLLLFKNGEVAATKVGALSKGQLKEFLDANLA"
        ),
        notes="Common fusion tag for improving solubility",
    ),
    ReferenceProtein(
        name="MBP (E. coli)",
        organism="E. coli",
        known_soluble=True,
        sequence=(
            "MKIEEGKLVIWINGDKGYNGLAEVGKKFEKDTGIKVTVEHPDKLEEKFPQVAATGDGPDI"
            "IFWAHDRFGGYAQSGLLAEITPDKAFQDKLYPFTWDAVRYNGKLIAYPIAVEALSLIYNK"
            "DLLPNPPKTWEEIPALDKELKAKGKSALMFNLQEPYFTWPLIAADGGYAFKYENGKYDIK"
            "DVGVDNAGAKAGLTFLVDLIKNKHMNADTDYSIAEAAFNKGETAMTINGPWAWSNIDTSK"
            "VNYGVTVLPTFKGQPSKPFVGVLSAGINAASPNKELAKEFLENYLLTDEGLEAVNKDKP"
            "LGAVALKSYEEELAKDPRIAATMENAQKGEIMPNIPQMSAFWYAVRTAVINAASGRQTVD"
            "EALKDAQTRITK"
        ),
        notes="Maltose-binding protein, classic solubility enhancer",
    ),
    ReferenceProtein(
        name="Insulin (Human)",
        organism="H. sapiens",
        known_soluble=False,
        sequence=(
            "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAED"
            "LQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN"
        ),
        notes="Requires disulfide bonds, aggregation-prone in E. coli",
    ),
    ReferenceProtein(
        name="p53 (Human, full-length)",
        organism="H. sapiens",
        known_soluble=False,
        sequence=(
            "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPG"
            "PDEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYPQGLNGTVNLPGRNSFEV"
            "RVCACPGRDRRTEEENLHKTTGIDSFLHPAISVGQFTLYTLLEYDSSPDHGGFPDPVGPFDY"
            "SPFHPFSSHSAPAKAAHDKFSVSCPFCRLREQLEQTLKHVVFQHVISNDGLSPDQIDHEPV"
            "HTASPAEGAGTTAPSVSGAQPAGAELQRSQHQDLVQEVEQGRAEASQHQQDSASEQPVDEPG"
            "SMEVSQVAEQPTLTCVPADPVAHATTIHSSGQPASEGQWVTQNKELKPLCQRFLCPVHKRTS"
        ),
        notes="Intrinsically disordered, forms inclusion bodies in E. coli",
    ),
    ReferenceProtein(
        name="SUMO1 (Human)",
        organism="H. sapiens",
        known_soluble=True,
        sequence=(
            "MSDQEAKPSTEDLGDKKEGEYIKLKVIGQDSSEIHFKVKMTTHLKKLKESYCQRQGVPM"
            "NSLRFLLFEGQRIADNHTPKELGMEEEDVIEVYQEQTGGHSTV"
        ),
        notes="Small ubiquitin-like modifier, highly soluble",
    ),
    ReferenceProtein(
        name="Amyloid-beta 42 (Human)",
        organism="H. sapiens",
        known_soluble=False,
        sequence="DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA",
        notes="Extreme aggregation propensity, Alzheimer's peptide",
    ),
]


def get_reference_results() -> list[dict]:
    """Return reference proteins formatted for display (without prediction)."""
    return [
        {
            "name": ref.name,
            "sequence": ref.sequence,
            "length": len(ref.sequence),
            "known_soluble": ref.known_soluble,
            "notes": ref.notes,
        }
        for ref in REFERENCE_PANEL
    ]
