import numpy as np


class DistrictPopulation:
    """Store population data for a district."""

    def __init__(
        self,
        district: str,
        years: np.ndarray,
        populations: np.ndarray
    ):
        self.district = district
        self.years = np.asarray(years, dtype=int)
        self.populations = np.asarray(populations, dtype=float)

        if len(self.years) != len(self.populations):
            raise ValueError("Years and populations must have equal lengths.")

        if len(self.years) == 0:
            raise ValueError("Population data cannot be empty.")

        if np.any(self.populations < 0):
            raise ValueError("Population values cannot be negative.")

    def __len__(self) -> int:
        """Return the number of population observations."""
        return len(self.populations)

    def __repr__(self) -> str:
        """Return a short description of the district."""
        return (
            f"DistrictPopulation("
            f"district='{self.district}', "
            f"observations={len(self)})"
        )

    def growth_rates(self) -> np.ndarray:
        """Calculate year-on-year population growth rates."""
        return np.diff(self.populations) / self.populations[:-1]

    def cagr(self) -> float:
        """Calculate the compound annual growth rate."""
        periods = self.years[-1] - self.years[0]
        return (self.populations[-1] / self.populations[0]) ** (1 / periods) - 1

#--- my forecasting base class ---

from abc import ABC, abstractmethod


class Forecaster(ABC):
    """Abstract base class for population forecasting."""

    @abstractmethod
    def fit(self, years: np.ndarray, populations: np.ndarray):
        """Fit the forecasting model."""
        pass

    @abstractmethod
    def predict(self, horizon: int) -> np.ndarray:
        """Predict future population values."""
        pass

#--- my linear trend forecasting class ---

class LinearTrendForecaster(Forecaster):
    """Forecast population using a linear trend."""

    def fit(
        self,
        years: np.ndarray,
        populations: np.ndarray
    ):
        self.slope, self.intercept = np.polyfit(
            years,
            populations,
            1
        )
        self.last_year = years[-1]
        return self

    def predict(self, horizon: int) -> np.ndarray:
        future_years = np.arange(
            self.last_year + 1,
            self.last_year + horizon + 1
        )

        return self.slope * future_years + self.intercept
    
#--- My CAGR Forecasting class ---

class CAGRForecaster(Forecaster):
    """Forecast population using compound annual growth."""

    def fit(
        self,
        years: np.ndarray,
        populations: np.ndarray
    ):
        periods = years[-1] - years[0]

        self.cagr = (
            populations[-1] / populations[0]
        ) ** (1 / periods) - 1

        self.last_population = populations[-1]
        return self

    def predict(self, horizon: int) -> np.ndarray:
        steps = np.arange(1, horizon + 1)

        return self.last_population * (
            1 + self.cagr
        ) ** steps

#--- My Fibonacci Forecasting Class ---

class FibonacciForecaster(Forecaster):
    """Forecast population using successive Fibonacci ratios."""

    def fit(
        self,
        years: np.ndarray,
        populations: np.ndarray
    ):
        self.last_population = populations[-1]
        return self

    def predict(self, horizon: int) -> np.ndarray:
        fibonacci = [1, 2]

        while len(fibonacci) < horizon + 1:
            fibonacci.append(
                fibonacci[-1] + fibonacci[-2]
            )

        ratios = [
            fibonacci[i + 1] / fibonacci[i]
            for i in range(horizon)
        ]

        forecasts = []
        current = self.last_population

        for ratio in ratios:
            current = current * ratio
            forecasts.append(current)

        return np.array(forecasts)

def bootstrap_prediction_interval(
    actual: np.ndarray,
    forecast: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42
):
    """Calculate a 95% bootstrap prediction interval."""

    residuals = actual - forecast

    rng = np.random.default_rng(seed)

    bootstrap_forecasts = []

    for _ in range(n_bootstrap):
        sampled_residuals = rng.choice(
            residuals,
            size=len(forecast),
            replace=True
        )

        bootstrap_forecasts.append(
            forecast + sampled_residuals
        )

    bootstrap_forecasts = np.array(bootstrap_forecasts)

    lower = np.percentile(
        bootstrap_forecasts,
        2.5,
        axis=0
    )

    upper = np.percentile(
        bootstrap_forecasts,
        97.5,
        axis=0
    )

    return lower, upper