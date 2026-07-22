"""Root pytest configuration for the heart-disease-prediction test suite.

This module is imported by pytest before test collection (it sits at the
project root, which is the pytest rootdir), so the Hypothesis profiles it
registers are available to every property-based test under both ``tests`` and
``backend/tests``.

Per the design's Testing Strategy, each property-based test must run a minimum
of 100 examples. The ``default`` profile below enforces that floor; a heavier
``ci`` profile is provided for more thorough runs. Select a profile at runtime
with the ``HYPOTHESIS_PROFILE`` environment variable, e.g.::

    HYPOTHESIS_PROFILE=ci pytest
"""

import os

from hypothesis import HealthCheck, settings

# Minimum iteration count mandated by the Testing Strategy (>= 100 examples).
settings.register_profile(
    "default",
    max_examples=100,
    # Some property tests fit a small scikit-learn pipeline; don't fail them
    # merely for being slower than Hypothesis' default per-example budget.
    suppress_health_check=[HealthCheck.too_slow],
)

# A more exhaustive profile for CI or thorough local runs.
settings.register_profile(
    "ci",
    max_examples=500,
    suppress_health_check=[HealthCheck.too_slow],
)

# Load the requested profile, defaulting to the 100-example "default" profile.
settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "default"))
