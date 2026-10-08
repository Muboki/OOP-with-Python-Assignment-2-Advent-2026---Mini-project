import sys
from pathlib import Path

import numpy as np
import pytest

# Locate the project folder to import the reusable classes.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.taxi import (
    Route,
    MovingAverageForecaster,
    ExponentialSmoothingForecaster,
    SeasonalNaiveForecaster,
    walk_forward
)


def test_route_revenue():
    """Check fare revenue, including a zero-passenger day."""
    route = Route("Kampala–Ntinda", np.array([35, 0, 40]), 2000)

    # Expected daily revenues are calculated directly by hand.
    np.testing.assert_allclose(route.daily_revenue(), [70000, 0, 80000])
    assert route.total_revenue() == pytest.approx(150000)


def test_empty_passengers():
    """Reject a route with no passenger observations."""
    with pytest.raises(ValueError, match="non-empty"):
        Route("Kampala–Ntinda", np.array([]), 2000)


def test_walk_forward():
    """Check that evaluation forecasts use only preceding days."""
    model = MovingAverageForecaster()

    # Day 4 uses [10, 20, 30]; day 5 uses [20, 30, 40].
    predictions, mae = walk_forward(
        model, np.array([10, 20, 30, 40, 50])
    )

    np.testing.assert_allclose(predictions, [20, 30])
    assert mae == pytest.approx(20)


def test_smoothing_alpha_one():
    """Check that alpha one predicts the latest observation."""
    model = ExponentialSmoothingForecaster(alpha=1.0)

    assert model.predict(np.array([35, 40, 42])) == pytest.approx(42)


def test_seasonal_naive():
    """Check that the next forecast uses the corresponding weekday."""
    model = SeasonalNaiveForecaster()

    # After eight days, day 9 should use day 2's passenger count.
    history = np.array([50, 52, 54, 56, 80, 60, 30, 51])

    assert model.predict(history) == pytest.approx(52)