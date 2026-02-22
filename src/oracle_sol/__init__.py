"""ORACLE-Sol: Protein solubility prediction for the modern design pipeline."""

__version__ = "0.6.0"
__author__ = "Asutosh Rath"

from oracle_sol.api import predict_solubility
from oracle_sol.core.predictor import OraclePredictor, SolubilityResult

__all__ = ["predict_solubility", "OraclePredictor", "SolubilityResult"]
