"""Algorithm registry / factory — single entry-point to instantiate any algorithm."""

from __future__ import annotations

from typing import Type

from src.algorithms.base import PhaseRetriever
from src.algorithms.admm import ADMM
from src.algorithms.douglas_rachford import DouglasRachford
from src.algorithms.error_reduction import ErrorReduction
from src.algorithms.gerchberg_saxton import GerchbergSaxton
from src.algorithms.hybrid_input_output import HybridInputOutput
from src.algorithms.phase_diversity import PhaseDiversity
from src.algorithms.pinn import PINNPhaseRetriever
from src.algorithms.raar import RAAR
from src.algorithms.wirtinger_flow import WirtingerFlow
from src.models.config import AlgorithmConfig, AlgorithmName
from src.models.optics import PupilModel


class AlgorithmRegistry:
    """Factory that maps algorithm names to concrete implementations."""

    _registry: dict[AlgorithmName, Type[PhaseRetriever]] = {
        AlgorithmName.ERROR_REDUCTION: ErrorReduction,
        AlgorithmName.GERCHBERG_SAXTON: GerchbergSaxton,
        AlgorithmName.HYBRID_INPUT_OUTPUT: HybridInputOutput,
        AlgorithmName.RAAR: RAAR,
        AlgorithmName.PHASE_DIVERSITY: PhaseDiversity,
        AlgorithmName.WIRTINGER_FLOW: WirtingerFlow,
        AlgorithmName.DOUGLAS_RACHFORD: DouglasRachford,
        AlgorithmName.ADMM: ADMM,
        AlgorithmName.PINN: PINNPhaseRetriever,
    }

    @classmethod
    def create(cls, config: AlgorithmConfig, pupil: PupilModel) -> PhaseRetriever:
        """Instantiate the algorithm specified by *config.name*.

        Parameters
        ----------
        config : AlgorithmConfig
            Algorithm hyper-parameters.
        pupil : PupilModel
            Telescope pupil model.

        Returns
        -------
        PhaseRetriever
            Ready-to-run algorithm instance.
        """
        klass = cls._registry.get(config.name)
        if klass is None:
            raise ValueError(
                f"Unknown algorithm '{config.name}'. "
                f"Available: {list(cls._registry.keys())}"
            )
        return klass(config, pupil)

    @classmethod
    def available(cls) -> list[str]:
        """List registered algorithm keys."""
        return [k.value for k in cls._registry]
