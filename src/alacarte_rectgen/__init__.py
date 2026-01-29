"""alacarte_rectgen

Install
-------
    pip install alacarte-rectgen

Quickstart
----------
See README.md or `examples/quickstart.py`.

The primary public API is:
    - make_rectangles_R_S
    - BoxSet
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from .core import (
    BoxSet,
    VolumeDist,
    estimate_alpha_by_pair_sampling,
    estimate_alpha_expected,
    generate_boxset,
    interval_overlap_prob,
    make_rectangles_R_S,
    solve_coverage_for_alpha,
)

try:
    __version__ = version("alacarte-rectgen")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

__all__ = [
    "__version__",
    "BoxSet",
    "VolumeDist",
    "make_rectangles_R_S",
    "generate_boxset",
    "solve_coverage_for_alpha",
    "estimate_alpha_expected",
    "estimate_alpha_by_pair_sampling",
    "interval_overlap_prob",
]
