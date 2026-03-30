"""Zernike polynomial basis for wavefront decomposition.

Uses the Noll (1976) single-index ordering convention standard in astronomy.
"""

from __future__ import annotations

import math

import numpy as np
from functools import lru_cache


def _noll_to_nm(j: int) -> tuple[int, int]:
    """Convert Noll index *j* (1-based) to radial order *n* and azimuthal frequency *m*."""
    if j < 1:
        raise ValueError(f"Noll index must be >= 1, got {j}")
    n = 0
    j_remaining = j
    while j_remaining > (n + 1):
        j_remaining -= (n + 1)
        n += 1
    m = -n + 2 * (j_remaining - 1)
    if j % 2 == 0:
        m = abs(m)
    else:
        m = -abs(m) if m != 0 else 0
    # Correct conversion using the standard formula
    return _noll_to_nm_standard(j)


def _noll_to_nm_standard(j: int) -> tuple[int, int]:
    """Robust Noll-to-(n, m) following Noll 1976, Table I."""
    n = int(np.ceil((-3 + np.sqrt(9 + 8 * (j - 1))) / 2))
    # ensure n is correct
    while (n + 1) * (n + 2) // 2 < j:
        n += 1
    while n > 0 and (n) * (n + 1) // 2 >= j:
        n -= 1
    k = j - n * (n + 1) // 2 - 1  # 0-based index within order n
    m_abs = n - 2 * ((n * (n + 1) // 2 + n - j + 1) // 2)
    # Simpler: use the triangular relation
    m_candidates = list(range(-n, n + 1, 2))
    # Sort by |m| then sign
    m_candidates_sorted = sorted(m_candidates, key=lambda x: (abs(x), -x))
    idx = j - n * (n + 1) // 2 - 1
    m = m_candidates_sorted[idx] if idx < len(m_candidates_sorted) else 0
    # Fall back to well-tested formula
    return _noll_lookup(j)


_NOLL_TABLE: dict[int, tuple[int, int]] = {
    1: (0, 0), 2: (1, 1), 3: (1, -1), 4: (2, 0),
    5: (2, -2), 6: (2, 2), 7: (3, -1), 8: (3, 1),
    9: (3, -3), 10: (3, 3), 11: (4, 0), 12: (4, 2),
    13: (4, -2), 14: (4, 4), 15: (4, -4), 16: (5, 1),
    17: (5, -1), 18: (5, 3), 19: (5, -3), 20: (5, 5),
    21: (5, -5), 22: (6, 0), 23: (6, -2), 24: (6, 2),
    25: (6, -4), 26: (6, 4), 27: (6, -6), 28: (6, 6),
    29: (7, -1), 30: (7, 1), 31: (7, -3), 32: (7, 3),
    33: (7, -5), 34: (7, 5), 35: (7, -7), 36: (7, 7),
    37: (8, 0),
}


def _noll_lookup(j: int) -> tuple[int, int]:
    """Lookup or compute (n, m) from Noll index."""
    if j in _NOLL_TABLE:
        return _NOLL_TABLE[j]
    # General formula
    n = int((-1 + np.sqrt(1 + 8 * j)) / 2)
    if (n + 1) * (n + 2) // 2 < j:
        n += 1
    remainder = j - n * (n + 1) // 2
    m = n - 2 * (remainder - 1) if remainder > 0 else n
    if j % 2 == 0 and m < 0:
        m = -m
    elif j % 2 != 0 and m > 0:
        m = -m
    return (n, abs(m) if j % 2 == 0 else -abs(m)) if m != 0 else (n, 0)


def radial_polynomial(n: int, m: int, rho: np.ndarray) -> np.ndarray:
    """Compute the radial polynomial R_n^|m|(rho) via direct summation."""
    m_abs = abs(m)
    if (n - m_abs) % 2 != 0:
        return np.zeros_like(rho)
    result = np.zeros_like(rho, dtype=np.float64)
    for s in range((n - m_abs) // 2 + 1):
        sign = (-1) ** s
        num = math.factorial(n - s)
        den = (
            math.factorial(s)
            * math.factorial((n + m_abs) // 2 - s)
            * math.factorial((n - m_abs) // 2 - s)
        )
        result += sign * (num / den) * rho ** (n - 2 * s)
    return result


def zernike(j: int, rho: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Evaluate Noll-ordered Zernike polynomial *j* on polar grid (*rho*, *theta*).

    Parameters
    ----------
    j : int
        Noll index (1-based).
    rho : ndarray
        Normalised radial coordinate (0 … 1 inside pupil).
    theta : ndarray
        Azimuthal angle (radians).

    Returns
    -------
    ndarray
        Zernike polynomial values; zero outside unit circle.
    """
    n, m = _noll_lookup(j)
    R = radial_polynomial(n, m, rho)
    if m == 0:
        norm = np.sqrt(n + 1)
        Z = norm * R
    elif m > 0:
        norm = np.sqrt(2 * (n + 1))
        Z = norm * R * np.cos(m * theta)
    else:
        norm = np.sqrt(2 * (n + 1))
        Z = norm * R * np.sin(abs(m) * theta)
    Z[rho > 1.0] = 0.0
    return Z


def zernike_basis(
    n_terms: int,
    grid_size: int,
    *,
    start_j: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a stack of Zernike polynomials on a square grid.

    Parameters
    ----------
    n_terms : int
        Number of Zernike terms to generate.
    grid_size : int
        Side length of the square grid.
    start_j : int
        Starting Noll index (default 2 = skip piston).

    Returns
    -------
    basis : ndarray, shape (n_terms, grid_size, grid_size)
    rho : ndarray, shape (grid_size, grid_size)
    theta : ndarray, shape (grid_size, grid_size)
    """
    y, x = np.mgrid[-1:1:complex(0, grid_size), -1:1:complex(0, grid_size)]
    rho = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(y, x)
    basis = np.zeros((n_terms, grid_size, grid_size), dtype=np.float64)
    for idx, j in enumerate(range(start_j, start_j + n_terms)):
        basis[idx] = zernike(j, rho, theta)
    return basis, rho, theta


# Convenience names for the first few aberrations
ZERNIKE_NAMES: dict[int, str] = {
    1: "Piston",
    2: "Tip (x-tilt)",
    3: "Tilt (y-tilt)",
    4: "Defocus",
    5: "Astigmatism (oblique)",
    6: "Astigmatism (vertical)",
    7: "Coma (vertical)",
    8: "Coma (horizontal)",
    9: "Trefoil (oblique)",
    10: "Trefoil (vertical)",
    11: "Primary spherical",
    12: "Secondary astigmatism",
    13: "Secondary astigmatism",
    14: "Quadrafoil",
    15: "Quadrafoil",
    22: "Secondary spherical",
    37: "Tertiary spherical",
}



