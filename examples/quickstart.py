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
