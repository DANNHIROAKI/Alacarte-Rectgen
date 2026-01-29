"""alacarte_rectgen.core

A practical, reproducible synthetic axis-aligned hyper-rectangle (box) generator
in the spirit of *Benchmarking Spatial Joins À La Carte*.

Main goal
---------
Generate two box sets R and S with specified sizes |R|, |S| and a *target* output
density:

    alpha_out = |J(R,S)| / (|R| + |S|)

where J(R,S) = {(r,s): r∈R, s∈S, r intersects s}.

High-level idea
---------------
We control the *coverage* C (expected total box volume as a fraction of the
universe volume):

    C := (sum_i vol(box_i)) / vol(universe)

For a set of size n, mean box volume is v̄ = C·vol(U)/n.

We solve for the coverage C that makes alpha_out match the requested target
*in expectation* (via Monte Carlo over rectangle side-lengths), then generate
R and S at that coverage with uniform placement under the assumption that each
box fits in the universe.

Coordinate convention
---------------------
Each box is stored as a Cartesian product of half-open intervals:

    box_i = Π_k [lower[i,k], upper[i,k])

This matches common spatial-join benchmark conventions and avoids boundary
double-counting.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Tuple, Literal

import numpy as np

VolumeDist = Literal["fixed", "exponential", "normal", "lognormal"]


# ---------------------------------------------------------------------------
# Public data container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BoxSet:
    """A set of n axis-aligned half-open boxes in d dimensions.

    For i in [0, n):
        box_i = Π_k [lower[i,k], upper[i,k])

    Attributes
    ----------
    lower, upper:
        Arrays of shape (n, d).
    universe:
        Array of shape (d, 2) giving [min, max] per dimension.
    """

    lower: np.ndarray  # (n, d)
    upper: np.ndarray  # (n, d)
    universe: np.ndarray  # (d, 2)

    @property
    def n(self) -> int:
        return int(self.lower.shape[0])

    @property
    def d(self) -> int:
        return int(self.lower.shape[1])

    def __len__(self) -> int:
        return self.n

    def as_array(self) -> np.ndarray:
        """Return a (n, 2d) array: [L0..Ld-1, U0..Ud-1]."""
        return np.hstack([self.lower, self.upper])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ensure_universe(universe: Optional[np.ndarray], d: int) -> np.ndarray:
    """Validate / normalize the universe.

    Parameters
    ----------
    universe:
        Array-like of shape (d, 2) giving [min, max] per dimension.
        If None, defaults to the unit hypercube [0, 1)^d.
    d:
        Dimension.

    Returns
    -------
    np.ndarray
        Float64 array of shape (d, 2).
    """
    if universe is None:
        return np.array([[0.0, 1.0]] * d, dtype=np.float64)

    U = np.asarray(universe, dtype=np.float64)
    if U.shape != (d, 2):
        raise ValueError(f"universe must have shape (d,2); got {U.shape}")
    if np.any(U[:, 1] <= U[:, 0]):
        raise ValueError("universe upper must be > lower in every dim")
    return U


def _normalize_volume_dist(volume_dist: str) -> str:
    vd = str(volume_dist).strip().lower()
    if vd not in {"fixed", "exponential", "normal", "lognormal"}:
        raise ValueError(
            "volume_dist must be one of {'fixed','exponential','normal','lognormal'}; "
            f"got {volume_dist!r}"
        )
    return vd


def sample_volumes(
    n: int,
    mean_vol: float,
    dist: VolumeDist,
    rng: np.random.Generator,
    cv: float = 0.25,
) -> np.ndarray:
    """Sample positive volumes with E[V] ≈ mean_vol.

    Parameters
    ----------
    n:
        Number of samples.
    mean_vol:
        Desired mean.
    dist:
        Distribution name.
    rng:
        NumPy random generator.
    cv:
        Coefficient of variation used for normal/lognormal (ignored otherwise).
    """
    mean_vol = float(mean_vol)
    if mean_vol <= 0:
        # Avoid exact zeros; extremely tiny boxes also help keep invariants under float32.
        return np.full(n, 1e-18, dtype=np.float64)

    if dist == "fixed":
        return np.full(n, mean_vol, dtype=np.float64)

    if dist == "exponential":
        # E = mean_vol
        return rng.exponential(scale=mean_vol, size=n).astype(np.float64)

    if dist == "normal":
        # Truncated normal (clip to positive)
        sigma = float(cv) * mean_vol
        vols = rng.normal(loc=mean_vol, scale=sigma, size=n).astype(np.float64)
        return np.clip(vols, a_min=mean_vol * 1e-12, a_max=None)

    if dist == "lognormal":
        # Choose sigma so that cv matches: cv^2 = exp(sigma^2) - 1
        sigma = math.sqrt(math.log1p(float(cv) ** 2))
        mu = math.log(mean_vol) - 0.5 * sigma**2
        return rng.lognormal(mean=mu, sigma=sigma, size=n).astype(np.float64)

    raise ValueError(f"Unknown volume distribution: {dist}")


def sample_shape_factors(
    n: int,
    d: int,
    shape_sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample shape factors g[i,k] with Π_k g[i,k] = 1 for each i.

    Then side lengths are:

        len[i,k] = (V[i]^(1/d)) * g[i,k]

    - shape_sigma = 0 -> hypercubes (all g=1)
    - shape_sigma > 0 -> log-normal aspect-ratio variation.
    """
    if shape_sigma <= 0:
        return np.ones((n, d), dtype=np.float64)

    z = rng.normal(loc=0.0, scale=float(shape_sigma), size=(n, d)).astype(np.float64)
    g = np.exp(z)
    g /= np.prod(g, axis=1, keepdims=True) ** (1.0 / d)
    return g


def sample_lengths_from_mean(
    n_samples: int,
    d: int,
    universe: np.ndarray,
    mean_vol: float,
    volume_dist: VolumeDist,
    volume_cv: float,
    shape_sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample side lengths (n_samples, d) given a desired mean volume."""
    spans = universe[:, 1] - universe[:, 0]

    vols = sample_volumes(n_samples, mean_vol, volume_dist, rng, cv=volume_cv)
    g = sample_shape_factors(n_samples, d, shape_sigma, rng)
    base = vols ** (1.0 / d)
    lengths = base[:, None] * g

    # Cap to slightly below spans to avoid degeneracy (and division by ~0 in formulas).
    eps = 1e-12
    lengths = np.minimum(lengths, spans * (1.0 - eps))
    return lengths


# ---------------------------------------------------------------------------
# Exact 1D overlap probability (given lengths)
# ---------------------------------------------------------------------------

def interval_overlap_prob(a: np.ndarray, b: np.ndarray, W: float) -> np.ndarray:
    """Exact overlap probability for two 1D intervals under uniform placement.

    Let X ~ Uniform(0, W-a) and Y ~ Uniform(0, W-b), independent.
    Consider the half-open intervals [X, X+a) and [Y, Y+b).

    If a+b >= W, overlap probability is 1.
    Else (a+b < W):

        p = 1 - (W-(a+b))^2 / ((W-a)(W-b))

    Parameters
    ----------
    a, b:
        Interval lengths (broadcastable arrays).
    W:
        Universe span in that dimension.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    W = float(W)

    a = np.clip(a, 0.0, W)
    b = np.clip(b, 0.0, W)

    c = W - (a + b)
    p = np.ones(np.broadcast(a, b).shape, dtype=np.float64)

    mask = c > 0.0
    if np.any(mask):
        denom = (W - a) * (W - b)  # positive when mask holds
        p[mask] = 1.0 - (c[mask] ** 2) / denom[mask]
    return p


def estimate_alpha_expected(
    *,
    nR: int,
    nS: int,
    alpha_out: float,
    d: int,
    universe: Optional[np.ndarray],
    coverage: float,
    volume_dist: VolumeDist,
    volume_cv: float,
    shape_sigma: float,
    num_samples: int,
    seed: int,
) -> Tuple[float, float]:
    """Estimate expected alpha_out for a given coverage (Monte Carlo over lengths).

    We estimate the pairwise intersection probability as:

        p = E[ Π_k p1D(lenR_k, lenS_k) ]

    where p1D is the exact 1D overlap probability under uniform placement.

    Returns
    -------
    (alpha_est, p_est)
    """
    _ = alpha_out  # (kept for API symmetry / logging)
    U = ensure_universe(universe, d)
    spans = U[:, 1] - U[:, 0]
    U_vol = float(np.prod(spans))

    mean_vol_R = float(coverage) * U_vol / int(nR)
    mean_vol_S = float(coverage) * U_vol / int(nS)

    rngR = np.random.default_rng(int(seed))
    rngS = np.random.default_rng(int(seed) + 1)

    lenR = sample_lengths_from_mean(
        num_samples, d, U, mean_vol_R, volume_dist, volume_cv, shape_sigma, rngR
    )
    lenS = sample_lengths_from_mean(
        num_samples, d, U, mean_vol_S, volume_dist, volume_cv, shape_sigma, rngS
    )

    p = np.ones(int(num_samples), dtype=np.float64)
    for k in range(int(d)):
        p *= interval_overlap_prob(lenR[:, k], lenS[:, k], spans[k])

    p_est = float(p.mean())
    alpha_est = p_est * (int(nR) * int(nS)) / (int(nR) + int(nS))
    return alpha_est, p_est


def solve_coverage_for_alpha(
    *,
    nR: int,
    nS: int,
    alpha_target: float,
    d: int = 2,
    universe: Optional[np.ndarray] = None,
    volume_dist: VolumeDist = "fixed",
    volume_cv: float = 0.25,
    shape_sigma: float = 0.0,
    num_samples: int = 200_000,
    seed: int = 0,
    tol_rel: float = 0.02,
    max_iter: int = 30,
) -> Tuple[float, List[Dict[str, float]]]:
    """Solve for coverage C such that E[alpha_out] ≈ alpha_target.

    Uses bracketing + log-space binary search. The objective is monotone in C
    for typical regimes.

    Returns
    -------
    (coverage, history)
        history is a list of dicts with keys: 'coverage', 'alpha_est'.
    """
    alpha_target = float(alpha_target)
    if alpha_target < 0:
        raise ValueError("alpha_target must be >= 0")
    if alpha_target == 0:
        return 0.0, [{"coverage": 0.0, "alpha_est": 0.0}]

    nR_i, nS_i = int(nR), int(nS)
    alpha_max = (nR_i * nS_i) / (nR_i + nS_i)  # since |J| <= nR*nS
    if alpha_target > alpha_max:
        raise ValueError(
            f"alpha_target={alpha_target} exceeds max possible {alpha_max:.6g} "
            f"for given |R|={nR_i}, |S|={nS_i}."
        )

    # Initial guess (small-rectangle regime, nR≈nS): alpha_out ≈ 2C
    C0 = max(1e-12, alpha_target / 2.0)

    def eval_alpha(C: float) -> float:
        a_est, _ = estimate_alpha_expected(
            nR=nR_i,
            nS=nS_i,
            alpha_out=alpha_target,
            d=int(d),
            universe=universe,
            coverage=float(C),
            volume_dist=volume_dist,
            volume_cv=float(volume_cv),
            shape_sigma=float(shape_sigma),
            num_samples=int(num_samples),
            seed=int(seed),
        )
        return float(a_est)

    alpha0 = eval_alpha(C0)
    history: List[Dict[str, float]] = [{"coverage": float(C0), "alpha_est": float(alpha0)}]

    if abs(alpha0 - alpha_target) <= float(tol_rel) * alpha_target:
        return float(C0), history

    # Bracket [lo, hi] s.t. alpha(lo) < target <= alpha(hi)
    lo = 0.0
    hi = float(C0)
    if alpha0 < alpha_target:
        lo = float(C0)
        hi = float(C0)
        for _ in range(60):
            hi *= 2.0
            a_hi = eval_alpha(hi)
            history.append({"coverage": float(hi), "alpha_est": float(a_hi)})
            if a_hi >= alpha_target:
                break
            lo = float(hi)
        else:
            raise RuntimeError(
                "Failed to bracket alpha_target; try increasing bounds or check parameters."
            )
    else:
        lo = 0.0
        hi = float(C0)

    # Log-space binary search
    lo_eps = 1e-15
    for _ in range(int(max_iter)):
        lo_for_log = max(lo, lo_eps)
        mid = math.exp(0.5 * (math.log(lo_for_log) + math.log(hi)))
        a_mid = eval_alpha(mid)
        history.append({"coverage": float(mid), "alpha_est": float(a_mid)})

        if abs(a_mid - alpha_target) <= float(tol_rel) * alpha_target:
            return float(mid), history

        if a_mid < alpha_target:
            lo = float(mid)
        else:
            hi = float(mid)

    best = min(history, key=lambda h: abs(h["alpha_est"] - alpha_target))
    return float(best["coverage"]), history


# ---------------------------------------------------------------------------
# Concrete box generation
# ---------------------------------------------------------------------------

def generate_boxset(
    n: int,
    *,
    d: int = 2,
    universe: Optional[np.ndarray] = None,
    coverage: float = 1.0,
    volume_dist: VolumeDist = "fixed",
    volume_cv: float = 0.25,
    shape_sigma: float = 0.0,
    seed: int = 0,
    dtype: np.dtype = np.float32,
) -> BoxSet:
    """Generate a set of n boxes at target coverage (in expectation).

    Placement model:
        For each box, sample side lengths, then sample the lower corner uniformly
        from the region where the box fits inside the universe.

    Notes
    -----
    - Coordinates are returned in `dtype` (default float32 for memory efficiency).
    - We enforce `lower < upper` *in the output dtype* to avoid degenerate boxes
      caused by float rounding.
    """
    U = ensure_universe(universe, int(d))
    spans = U[:, 1] - U[:, 0]
    U_vol = float(np.prod(spans))

    n_i = int(n)
    mean_vol = float(coverage) * U_vol / n_i
    rng = np.random.default_rng(int(seed))

    volume_dist = _normalize_volume_dist(volume_dist)  # type: ignore[assignment]
    lengths = sample_lengths_from_mean(
        n_i, int(d), U, mean_vol, volume_dist, float(volume_cv), float(shape_sigma), rng  # type: ignore[arg-type]
    )

    # Lower corner uniform so it fits.
    # NumPy's Generator.random(dtype=...) is not available in older NumPy versions,
    # so we fall back if needed.
    try:
        u = rng.random((n_i, int(d)), dtype=np.float64)  # type: ignore[call-arg]
    except TypeError:
        u = rng.random((n_i, int(d))).astype(np.float64, copy=False)

    lower64 = U[:, 0] + u * (spans - lengths)
    upper64 = lower64 + lengths

    # Cast to requested dtype and enforce strict lower < upper **in that dtype**.
    U_d = U.astype(dtype, copy=False)
    lo = U_d[:, 0]
    hi = U_d[:, 1]

    lower = lower64.astype(dtype, copy=False)
    upper = upper64.astype(dtype, copy=False)

    # Keep coordinates inside the universe in the output dtype.
    lower = np.maximum(lower, lo)
    upper = np.minimum(upper, hi)

    # Ensure lower < hi so there exists a representable value above it.
    hi_prev = np.nextafter(hi, lo)  # largest representable value < hi (per dim)
    lower = np.minimum(lower, hi_prev)

    # Fix any degeneracy caused by dtype rounding: enforce upper > lower by at least one ULP.
    hi_b = np.broadcast_to(hi, lower.shape)
    upper_fix = np.nextafter(lower, hi_b)
    mask = upper <= lower
    if np.any(mask):
        upper = np.where(mask, upper_fix, upper)

    return BoxSet(lower=lower, upper=upper, universe=U_d)


def make_rectangles_R_S(
    nR: int,
    nS: int,
    alpha_out: float,
    *,
    d: int = 2,
    universe: Optional[np.ndarray] = None,
    volume_dist: VolumeDist = "fixed",
    volume_cv: float = 0.25,
    shape_sigma: float = 0.0,
    tune_samples: int = 200_000,
    tune_tol_rel: float = 0.02,
    tune_max_iter: int = 30,
    seed: int = 0,
    dtype: np.dtype = np.float32,
) -> Tuple[BoxSet, BoxSet, Dict[str, object]]:
    """Generate (R, S) with target expected output density alpha_out.

    This is the main public entry point used in the README example:

        import alacarte_rectgen as ar
        R, S, info = ar.make_rectangles_R_S(...)

    Parameters
    ----------
    nR, nS:
        Sizes of the two sets.
    alpha_out:
        Target output density alpha_out = |J| / (|R|+|S|).
    d:
        Dimension (d>=1). For d=2 you get rectangles.
    universe:
        Universe bounds of shape (d,2). Default is [0,1)^d.
    volume_dist, volume_cv:
        Volume model for boxes. Use "fixed" to get equal volumes.
    shape_sigma:
        Aspect-ratio variation. 0 means hypercubes / squares.
    tune_samples, tune_tol_rel, tune_max_iter:
        Parameters for solving coverage C.
    seed:
        Random seed (controls both tuning and generation).
    dtype:
        Coordinate dtype (float32 is memory-friendly).

    Returns
    -------
    (R, S, info)
        info contains:
            - coverage
            - alpha_target
            - alpha_expected_est
            - pair_intersection_prob_est
            - tune_history
            - params
    """
    volume_dist = _normalize_volume_dist(volume_dist)  # type: ignore[assignment]

    C, history = solve_coverage_for_alpha(
        nR=int(nR),
        nS=int(nS),
        alpha_target=float(alpha_out),
        d=int(d),
        universe=universe,
        volume_dist=volume_dist,  # type: ignore[arg-type]
        volume_cv=float(volume_cv),
        shape_sigma=float(shape_sigma),
        num_samples=int(tune_samples),
        seed=int(seed) + 10_000,
        tol_rel=float(tune_tol_rel),
        max_iter=int(tune_max_iter),
    )

    R = generate_boxset(
        int(nR),
        d=int(d),
        universe=universe,
        coverage=float(C),
        volume_dist=volume_dist,  # type: ignore[arg-type]
        volume_cv=float(volume_cv),
        shape_sigma=float(shape_sigma),
        seed=int(seed) + 1,
        dtype=dtype,
    )
    S = generate_boxset(
        int(nS),
        d=int(d),
        universe=universe,
        coverage=float(C),
        volume_dist=volume_dist,  # type: ignore[arg-type]
        volume_cv=float(volume_cv),
        shape_sigma=float(shape_sigma),
        seed=int(seed) + 2,
        dtype=dtype,
    )

    # Report expected alpha using the tuned C (for logging)
    alpha_est, p_est = estimate_alpha_expected(
        nR=int(nR),
        nS=int(nS),
        alpha_out=float(alpha_out),
        d=int(d),
        universe=universe,
        coverage=float(C),
        volume_dist=volume_dist,  # type: ignore[arg-type]
        volume_cv=float(volume_cv),
        shape_sigma=float(shape_sigma),
        num_samples=int(min(200_000, int(tune_samples))),
        seed=int(seed) + 20_000,
    )

    info: Dict[str, object] = {
        "coverage": float(C),
        "alpha_target": float(alpha_out),
        "alpha_expected_est": float(alpha_est),
        "pair_intersection_prob_est": float(p_est),
        "tune_history": history,
        "params": {
            "nR": int(nR),
            "nS": int(nS),
            "d": int(d),
            "universe": None if universe is None else np.asarray(universe).tolist(),
            "volume_dist": str(volume_dist),
            "volume_cv": float(volume_cv),
            "shape_sigma": float(shape_sigma),
            "seed": int(seed),
            "dtype": str(np.dtype(dtype)),
        },
    }
    return R, S, info


# ---------------------------------------------------------------------------
# Optional: realized-alpha sanity check (pair sampling on generated sets)
# ---------------------------------------------------------------------------

def estimate_alpha_by_pair_sampling(
    R: BoxSet,
    S: BoxSet,
    *,
    num_pairs: int = 1_000_000,
    seed: int = 0,
) -> Tuple[float, float]:
    """Estimate realized alpha_out by random pair sampling on generated sets.

    Warning
    -------
    For very large N and very small alpha_out, true p is tiny (≈O(1/N)),
    so you may need many sampled pairs to observe enough intersections.
    """
    rng = np.random.default_rng(int(seed))
    idxR = rng.integers(0, int(R.n), size=int(num_pairs))
    idxS = rng.integers(0, int(S.n), size=int(num_pairs))

    lower_max = np.maximum(R.lower[idxR], S.lower[idxS])
    upper_min = np.minimum(R.upper[idxR], S.upper[idxS])
    hits = int(np.all(lower_max < upper_min, axis=1).sum())

    p_hat = hits / float(num_pairs)
    alpha_hat = p_hat * (int(R.n) * int(S.n)) / (int(R.n) + int(S.n))
    return float(alpha_hat), float(p_hat)
