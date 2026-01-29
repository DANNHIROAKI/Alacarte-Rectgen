import numpy as np

import alacarte_rectgen as ar


def test_make_rectangles_R_S_shapes_and_info():
    R, S, info = ar.make_rectangles_R_S(
        nR=2000,
        nS=1500,
        alpha_out=10.0,
        d=2,
        universe=None,
        volume_dist="normal",
        volume_cv=0.25,
        shape_sigma=0.5,
        seed=123,
        tune_samples=20_000,
        tune_tol_rel=0.05,   # loose tolerance for a fast unit test
        dtype=np.float32,
    )

    assert R.lower.shape == (2000, 2)
    assert R.upper.shape == (2000, 2)
    assert S.lower.shape == (1500, 2)
    assert S.upper.shape == (1500, 2)

    # Invariants in the returned dtype
    assert np.all(R.lower < R.upper)
    assert np.all(S.lower < S.upper)

    # Info keys used in README / quickstart
    for k in [
        "coverage",
        "alpha_target",
        "alpha_expected_est",
        "pair_intersection_prob_est",
        "tune_history",
        "params",
    ]:
        assert k in info

    # Basic sanity: alpha estimate should not be wildly off the target.
    target = float(info["alpha_target"])
    est = float(info["alpha_expected_est"])
    assert est > 0
    assert abs(est - target) / target < 0.5  # coarse bound to avoid flaky tests
