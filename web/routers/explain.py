"""Educational / explanatory endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from web.schemas import AlgorithmExplain, MetricExplain

router = APIRouter(prefix="/api/explain", tags=["explain"])

_ALGORITHMS: list[dict[str, str]] = [
    {
        "key": "er",
        "name": "Error Reduction",
        "category": "Classic",
        "description": (
            "Fienup's basic projection algorithm (1982). Alternates between "
            "pupil-support and Fourier-magnitude constraints. Guaranteed to "
            "decrease cost monotonically but converges slowly and is prone to "
            "stagnation."
        ),
        "reference": "Fienup J.R. (1982) Applied Optics 21(15):2758-2769",
    },
    {
        "key": "gs",
        "name": "Gerchberg–Saxton",
        "category": "Classic",
        "description": (
            "Classic two-plane amplitude constraint algorithm (1972). "
            "Enforces known amplitudes in both pupil and focal planes. "
            "Equivalent to ER when the pupil amplitude is binary."
        ),
        "reference": "Gerchberg R.W., Saxton W.O. (1972) Optik 35:237-246",
    },
    {
        "key": "hio",
        "name": "Hybrid Input-Output",
        "category": "Classic",
        "description": (
            "Fienup's workhorse algorithm with a feedback parameter β. "
            "Outside the support, the update mixes the current and projected "
            "estimates, preventing stagnation. The most widely used classical "
            "phase retrieval algorithm."
        ),
        "reference": "Fienup J.R. (1982) Applied Optics 21(15):2758-2769",
    },
    {
        "key": "raar",
        "name": "Relaxed Averaged Alternating Reflections",
        "category": "Modern",
        "description": (
            "Luke's convex-relaxation method (2005). Uses a combination of "
            "reflectors and projectors with parameter β to achieve more robust "
            "convergence than HIO. Includes an ER polish stage in the final "
            "10% of iterations."
        ),
        "reference": "Luke D.R. (2005) Inverse Problems 21(1):37-50",
    },
    {
        "key": "wf",
        "name": "Wirtinger Flow",
        "category": "State-of-the-Art",
        "description": (
            "Gradient descent with spectral initialization (Candès et al. 2015). "
            "Uses Wirtinger (complex) calculus to compute the gradient of the "
            "intensity loss. With spectral init, converges linearly to the "
            "global optimum — provably optimal among first-order methods."
        ),
        "reference": (
            "Candès E.J., Li X., Soltanolkotabi M. (2015) IEEE Trans. IT 61(4):1985-2007"
        ),
    },
    {
        "key": "dr",
        "name": "Douglas-Rachford",
        "category": "State-of-the-Art",
        "description": (
            "Proximal splitting with provable fixed-point convergence. "
            "Particularly effective for non-convex constraint sets. Uses "
            "reflectors to navigate between the two constraint sets."
        ),
        "reference": ("Bauschke H.H., Combettes P.L., Luke D.R. (2002) JOSA A 19(7):1334-1345"),
    },
    {
        "key": "admm",
        "name": "ADMM",
        "category": "State-of-the-Art",
        "description": (
            "Alternating Direction Method of Multipliers — splits the problem "
            "into Fourier-magnitude and pupil-support sub-problems linked by "
            "a dual variable. Naturally handles regularization and converges "
            "robustly on ill-conditioned problems."
        ),
        "reference": "Chang H., Marchesini S. (2018) arXiv:1804.05306",
    },
    {
        "key": "pinn",
        "name": "Physics-Informed Neural Field",
        "category": "Neural",
        "description": (
            "A coordinate MLP with random Fourier feature encoding, optimised "
            "through a differentiable Fourier-optics forward model. Uses "
            "two-phase Adam → L-BFGS optimisation and warm-starts from RAAR "
            "for dramatic convergence improvements. Requires PyTorch."
        ),
        "reference": "Tancik M. et al. (2020) NeurIPS — Fourier Features",
    },
    {
        "key": "fista",
        "name": "FISTA",
        "category": "State-of-the-Art",
        "description": (
            "Fast Iterative Shrinkage-Thresholding Algorithm with Nesterov "
            "acceleration (Beck & Teboulle 2009). A proximal-gradient method "
            "that supports pluggable regularisers (TV, L1-wavelet) for "
            "noise-robust phase recovery. Converges as O(1/k²)."
        ),
        "reference": "Beck A., Teboulle M. (2009) SIAM J. Imaging Sciences 2(1):183-202",
    },
    {
        "key": "sparse_pr",
        "name": "Sparse Phase Retrieval (ThWF)",
        "category": "State-of-the-Art",
        "description": (
            "Thresholded Wirtinger Flow for sparsity-promoting phase recovery "
            "(Cai, Li & Ma 2016). Applies hard or soft thresholding with "
            "adaptive decay to exploit signal sparsity — particularly effective "
            "for crystallographic data with sparse electron density maps."
        ),
        "reference": "Cai T., Li X., Ma Z. (2016) Annals of Statistics 44(5):2221-2251",
    },
]

_METRICS: list[dict[str, str]] = [
    {
        "name": "Strehl Ratio",
        "description": (
            "Peak intensity of the aberrated PSF divided by the peak of the "
            "diffraction-limited PSF. A Strehl of 1.0 means a perfect optical "
            "system. The Maréchal criterion (Strehl ≥ 0.8) defines 'diffraction-limited'."
        ),
        "unit": "dimensionless (0–1)",
    },
    {
        "name": "RMS Wavefront Error",
        "description": (
            "Root-mean-square of the recovered phase over the pupil support, "
            "after removing piston. Smaller values indicate a flatter "
            "(less aberrated) wavefront."
        ),
        "unit": "radians",
    },
    {
        "name": "Zernike Decomposition",
        "description": (
            "Projects the recovered phase onto Noll-ordered Zernike polynomials "
            "(tip, tilt, defocus, astigmatism, coma, spherical, etc.). "
            "Tells you exactly which classical aberrations are present."
        ),
        "unit": "radians per mode",
    },
    {
        "name": "MTF (Modulation Transfer Function)",
        "description": (
            "Radially averaged magnitude of the optical transfer function. "
            "Shows how well the system transmits spatial-frequency detail — "
            "a perfect system has MTF = 1 at all frequencies up to the cutoff."
        ),
        "unit": "normalised (0–1) vs. cycles/pixel",
    },
    {
        "name": "SSIM (Structural Similarity)",
        "description": (
            "Structural Similarity Index between the observed and reconstructed "
            "PSFs. Measures luminance, contrast, and structure similarity. "
            "SSIM = 1 means a perfect reconstruction."
        ),
        "unit": "dimensionless (−1 to 1)",
    },
    {
        "name": "Phase Structure Function D_φ(r)",
        "description": (
            "Mean-squared phase difference as a function of separation: "
            "D_φ(r) = ⟨|φ(x) − φ(x+r)|²⟩. Standard diagnostic for wavefront "
            "quality and atmospheric turbulence characterisation."
        ),
        "unit": "rad² vs. pixels",
    },
]


@router.get("/algorithms", response_model=list[AlgorithmExplain])
def explain_algorithms() -> list[dict[str, str]]:
    """Detailed educational descriptions of all available algorithms."""
    return _ALGORITHMS


@router.get("/metrics", response_model=list[MetricExplain])
def explain_metrics() -> list[dict[str, str]]:
    """Descriptions of all quality metrics used in the pipeline."""
    return _METRICS


@router.get("/science")
def explain_science() -> dict[str, str]:
    """A short primer on phase retrieval and wavefront sensing."""
    return {
        "title": "Phase Retrieval for Astronomical Wavefront Sensing",
        "overview": (
            "Every optical telescope introduces wavefront aberrations "
            "(defocus, astigmatism, coma, spherical aberration, etc.). "
            "When a telescope images a point source (star), the resulting "
            "point-spread function (PSF) encodes these aberrations — but "
            "the detector only records INTENSITY (the squared modulus), "
            "losing all phase information."
        ),
        "problem": (
            "Phase retrieval recovers the lost wavefront phase from "
            "intensity-only PSF measurements, given knowledge of the "
            "telescope pupil geometry. This is exactly how NASA diagnosed "
            "HST's famous spherical aberration in 1990 and how JWST's "
            "mirrors are aligned today."
        ),
        "method": (
            "The algorithms alternate between two planes connected by "
            "the Fourier transform: the pupil plane (where we know the "
            "support/amplitude) and the focal plane (where we know the "
            "measured intensity). By enforcing known constraints in each "
            "plane, the phase estimate converges to the true wavefront."
        ),
        "applications": (
            "• HST/JWST mirror alignment and monitoring\n"
            "• Adaptive optics calibration\n"
            "• Optical metrology and manufacturing\n"
            "• Electron microscopy\n"
            "• X-ray crystallography"
        ),
        "crystallography": (
            "X-ray crystallography is one of the original applications of "
            "phase retrieval. When X-rays scatter off a crystal lattice, the "
            "detector records diffraction intensities |F(hkl)|² but loses the "
            "phase φ(hkl). Recovering these phases from the measured intensities "
            "is the 'crystallographic phase problem' — solved since the 1950s "
            "using direct methods (Hauptman & Karle, Nobel Prize 1985) and more "
            "recently with iterative projection algorithms identical to those "
            "used in astronomical wavefront sensing."
        ),
    }
