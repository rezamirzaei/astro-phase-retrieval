"""Multi-start optimization runner for phase retrieval.

Runs the configured algorithm from multiple random initializations and
returns the best result (lowest final cost).  This dramatically reduces
the chance of getting stuck in a poor local minimum, especially for
non-convex algorithms like HIO, RAAR, and Wirtinger Flow.
"""

from __future__ import annotations

import logging

from src.algorithms.registry import AlgorithmRegistry
from src.models.config import AlgorithmConfig
from src.models.optics import PSFData, PhaseRetrievalResult, PupilModel

logger = logging.getLogger(__name__)


def multi_start_run(
    config: AlgorithmConfig,
    pupil: PupilModel,
    psf_data: PSFData,
    n_starts: int | None = None,
) -> PhaseRetrievalResult:
    """Run phase retrieval from multiple random seeds, return the best result.

    Parameters
    ----------
    config : AlgorithmConfig
        Algorithm configuration (seed will be varied per start).
    pupil : PupilModel
        Telescope pupil model.
    psf_data : PSFData
        Observed PSF data.
    n_starts : int | None
        Number of random restarts.  If None, uses ``config.n_starts``.

    Returns
    -------
    PhaseRetrievalResult
        The result with the lowest final cost across all starts.
    """
    if n_starts is None:
        n_starts = config.n_starts

    if n_starts <= 1:
        retriever = AlgorithmRegistry.create(config, pupil)
        return retriever.run(psf_data)

    base_seed = config.random_seed if config.random_seed is not None else 0
    best_result: PhaseRetrievalResult | None = None
    best_cost = float("inf")

    for i in range(n_starts):
        seed = base_seed + i * 1000
        cfg_i = config.model_copy(update={"random_seed": seed})
        retriever = AlgorithmRegistry.create(cfg_i, pupil)
        result = retriever.run(psf_data)

        final_cost = result.cost_history[-1] if result.cost_history else float("inf")
        logger.info(
            "Multi-start %d/%d (seed=%d): cost=%.6f, Strehl=%.4f",
            i + 1, n_starts, seed, final_cost, result.strehl_ratio,
        )

        if final_cost < best_cost:
            best_cost = final_cost
            best_result = result

    assert best_result is not None
    logger.info("Best start: cost=%.6f, Strehl=%.4f", best_cost, best_result.strehl_ratio)
    return best_result
