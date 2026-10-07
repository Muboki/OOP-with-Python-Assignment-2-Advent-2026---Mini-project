import sys
from pathlib import Path

import numpy as np
import pytest

# Locate the project folder to import the reusable classes.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rainfall import Region, CropRule


def test_uniform_rainfall():
    """Check totals and variability for constant monthly rainfall."""
    # Twelve months of 100 mm give 1,200 mm and zero variability.
    region = Region("Test region", np.full(12, 100.0))

    assert region.annual_total() == pytest.approx(1200.0)
    assert region.mean() == pytest.approx(100.0)
    assert region.coefficient_of_variation() == pytest.approx(0.0)


def test_negative_rainfall():
    """Check that negative monthly rainfall is rejected."""
    # A single invalid month should prevent object creation.
    values = np.full(12, 100.0)
    values[0] = -1.0

    with pytest.raises(ValueError, match="negative"):
        Region("Test region", values)


def test_crop_rule_boundaries():
    """Check classification at and outside the rainfall limits."""
    # Both endpoints belong to the suitable range.
    maize = CropRule("Maize", 125.0, 200.0)

    assert maize.classify(125.0) == "Good for Maize"
    assert maize.classify(200.0) == "Good for Maize"
    assert maize.classify(124.0) == "Drought risk"
    assert maize.classify(201.0) == "Waterlogging risk"


def test_cosine_ignores_scale():
    """Check that proportional rainfall vectors have similarity one."""
    # Doubling rainfall changes its quantity but preserves its pattern.
    values = np.arange(1, 13, dtype=float)
    first = Region("First region", values)
    second = Region("Second region", values * 2)

    assert first.cosine_similarity(second) == pytest.approx(1.0)