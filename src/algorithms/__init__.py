"""Phase-retrieval algorithm implementations.

Includes classic and state-of-the-art algorithms:
  • Error Reduction (ER) — Fienup 1982
  • Gerchberg–Saxton (GS) — Gerchberg & Saxton 1972
  • Hybrid Input-Output (HIO) — Fienup 1982
  • Relaxed Averaged Alternating Reflections (RAAR) — Luke 2005
  • Phase Diversity (PD) — Gonsalves 1982, Paxman et al. 1992
  • Wirtinger Flow (WF) — Candès, Li & Soltanolkotabi 2015
  • Douglas-Rachford (DR) — Bauschke, Combettes & Luke 2002
  • ADMM — Boyd et al. 2011, Chang & Marchesini 2018
"""

from src.algorithms.base import PhaseRetriever
from src.algorithms.admm import ADMM
from src.algorithms.douglas_rachford import DouglasRachford
from src.algorithms.error_reduction import ErrorReduction
from src.algorithms.gerchberg_saxton import GerchbergSaxton
from src.algorithms.hybrid_input_output import HybridInputOutput
from src.algorithms.multi_start import multi_start_run
from src.algorithms.phase_diversity import PhaseDiversity
from src.algorithms.raar import RAAR
from src.algorithms.registry import AlgorithmRegistry
from src.algorithms.wirtinger_flow import WirtingerFlow

__all__ = [
    "ADMM",
    "AlgorithmRegistry",
    "DouglasRachford",
    "ErrorReduction",
    "GerchbergSaxton",
    "HybridInputOutput",
    "PhaseDiversity",
    "PhaseRetriever",
    "RAAR",
    "WirtingerFlow",
    "multi_start_run",
]
