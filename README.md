# Alacarte RectGen (`alacarte-rectgen`)

A **controllable output-density** synthetic generator for **axis-aligned hyper-rectangles** (boxes), inspired by the ideas in *Benchmarking Spatial Joins À La Carte*.

It generates two box sets **R** and **S** such that the expected spatial-join output density is close to a user-specified target:

$$
\alpha_{\mathrm{out}} = \frac{|J(R,S)|}{|R| + |S|},
\quad
J(R,S) = \{(r,s)\in R\times S\;|\; r\cap s \neq \varnothing\}.
$$

---

## Installation

```bash
pip install alacarte-rectgen
```

Import name:

```python
import alacarte_rectgen as ar
```

---

## Quickstart (matches the typical usage)

```python
import numpy as np
import alacarte_rectgen as ar

# Parameter settings
N_R, N_S = 500_000, 500_000
TARGET_ALPHA = 10.0

# 1. Generate data and solve parameters
R, S, info = ar.make_rectangles_R_S(
    nR=N_R,
    nS=N_S,
    alpha_out=TARGET_ALPHA,
    d=2,
    universe=None,          # Default is [0, 1)^2
    volume_dist="normal",   # Use normal distribution for volume
    volume_cv=0.25,         # Coefficient of variation for volume
    shape_sigma=0.5,        # Enable aspect ratio variation
    seed=42,
    tune_tol_rel=0.01       # Solving tolerance 1%
)

# 2. Access generation results
print(f"Generated R size: {R.n}, S size: {S.n}")
print(f"Coordinates shape: {R.lower.shape}")

# 3. Audit generation parameters
print("\n--- Generation Audit ---")
print(f"Solved Coverage (C): {info['coverage']:.6e}")
print(f"Target Alpha:        {info['alpha_target']:.4f}")
print(f"Expected Alpha:      {info['alpha_expected_est']:.4f}")
print(f"Intersection Prob:   {info['pair_intersection_prob_est']:.6e}")
```

---

## What you get back

### `R` and `S` are `BoxSet` objects

They store **half-open** boxes:

$$
\text{box}_i = \prod_{k} [\text{lower}_{i,k}, \text{upper}_{i,k})
$$

Key fields/properties:

- `R.lower`, `R.upper`: `np.ndarray` with shape `(n, d)`
- `R.universe`: `np.ndarray` with shape `(d, 2)` giving `[min, max]` per dimension
- `R.n`, `R.d`: sizes

### `info` is an audit dictionary

Common keys:

- `info["coverage"]`: solved coverage $C$
- `info["alpha_target"]`: requested target $\alpha_{out}$
- `info["alpha_expected_est"]`: Monte-Carlo estimate of expected $\alpha_{out}$ under the tuned coverage
- `info["pair_intersection_prob_est"]`: estimated pairwise intersection probability $p$
- `info["tune_history"]`: list of tried coverages + estimated alphas during tuning
- `info["params"]`: echo of the main generation parameters

---

## Parameters you will typically tune

- `universe` (default `None`): bounds of shape `(d,2)`. If `None`, uses `[0,1)^d`.
- `volume_dist`: `"fixed" | "exponential" | "normal" | "lognormal"`
- `volume_cv`: coefficient of variation for `"normal"` / `"lognormal"` volume distributions.
- `shape_sigma`: `0` gives squares/cubes; larger values increase aspect-ratio variation.
- `tune_samples`: Monte Carlo sample size used by the coverage solver (bigger = more accurate, slower).
- `tune_tol_rel`: relative tolerance of the solver, e.g. `0.01` for 1%.
- `dtype`: output coordinate dtype, default `np.float32` to save memory.

---

## Notes on scale & performance

- By default coordinates are `float32` to reduce memory use.
- Tuning uses Monte Carlo over **length samples** (not over all `nR*nS` pairs), so it stays practical even for large `nR`, `nS`.
- If you care about *realized* alpha on the concrete generated sets (not only the expectation), use:

```python
alpha_hat, p_hat = ar.estimate_alpha_by_pair_sampling(R, S, num_pairs=2_000_000, seed=0)
```

---

## License

MIT (see `LICENSE`).
