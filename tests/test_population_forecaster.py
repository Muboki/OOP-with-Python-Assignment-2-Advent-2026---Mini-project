import numpy as np
import pytest

from src.population_forecaster import (
    DistrictPopulation,
    LinearTrendForecaster,
    CAGRForecaster
)


def test_district_population_length():
    district = DistrictPopulation(
        "Test District",
        np.array([2020, 2021, 2022]),
        np.array([100, 110, 120])
    )

    assert len(district) == 3


def test_negative_population_is_rejected():
    with pytest.raises(ValueError):
        DistrictPopulation(
            "Test District",
            np.array([2020, 2021]),
            np.array([100, -50])
        )


def test_linear_forecaster_returns_correct_horizon():
    model = LinearTrendForecaster()

    model.fit(
        np.array([2020, 2021, 2022]),
        np.array([100, 110, 120])
    )

    forecast = model.predict(3)

    assert len(forecast) == 3