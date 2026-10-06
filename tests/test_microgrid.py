from pathlib import Path
import sys

import numpy as np
import pytest

# Locate src relative to this test file.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from microgrid import MicroGrid


def test_known_daily_solution():
    """Check a solution calculated by hand."""
    # These demands should require 320 kWh solar and 120 kWh battery.
    centre = MicroGrid("Kasese Health Centre")
    result = centre.solve_day(1200, 1400)
    np.testing.assert_allclose(result, [320.0, 120.0])


def test_zero_demand():
    """Check that zero demand requires zero energy."""
    # Zero is a valid boundary case.
    centre = MicroGrid("Kasese Health Centre")
    result = centre.solve_day(0, 0)
    np.testing.assert_allclose(result, [0.0, 0.0], atol=1e-12)


def test_negative_demand():
    """Check that negative demand is rejected."""
    # Invalid demand should raise a clear error.
    centre = MicroGrid("Kasese Health Centre")
    with pytest.raises(ValueError, match="non-negative"):
        centre.solve_day(-100, 1400)