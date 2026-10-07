import sys
from pathlib import Path

import numpy as np
import pytest

# Locate the project folder so the reusable classes can be imported.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.fishstock import FishStock


def test_first_week_calculation():
    """Check the simulation against a hand-calculated result."""
    # Growth is 960 tonnes and harvest is 400 tonnes.
    stock = FishStock("Jinja Fish Cooperative")
    levels, harvests = stock.simulate(harvest_rate=0.10, weeks=1)

    np.testing.assert_allclose(levels, [4000.0, 4560.0])
    np.testing.assert_allclose(harvests, [400.0])


def test_zero_initial_stock():
    """Check that an empty fish stock produces no growth or harvest."""
    # Zero stock is a valid boundary case.
    stock = FishStock("Jinja Fish Cooperative", initial_stock=0)
    levels, harvests = stock.simulate(harvest_rate=0.10)

    np.testing.assert_allclose(levels, 0.0)
    np.testing.assert_allclose(harvests, 0.0)


def test_invalid_harvest_rate():
    """Check that harvesting more than the entire stock is rejected."""
    # A harvest fraction above one is invalid.
    stock = FishStock("Jinja Fish Cooperative")

    with pytest.raises(ValueError, match="Harvest rate"):
        stock.simulate(harvest_rate=1.20)