import numpy as np


class DistrictPopulation:
    """Store a district's population history and calculate growth."""

    def __init__(
        self,
        district: str,
        years: np.ndarray,
        populations: np.ndarray
    ) -> None:
        # Store years as integers and populations as decimal values.
        self.district = district
        self.years = np.asarray(years, dtype=int)
        self.populations = np.asarray(populations, dtype=float)

        # Each year must have a corresponding population value.
        if len(self.years) != len(self.populations):
            raise ValueError("Years and populations must have equal lengths.")

        # Reject empty data and negative population values.
        if len(self.years) == 0:
            raise ValueError("Population data cannot be empty.")

        if np.any(self.populations < 0):
            raise ValueError("Population values cannot be negative.")

    def __len__(self) -> int:
        """Return the number of annual observations."""
        return len(self.populations)

    def __repr__(self) -> str:
        """Show the district name and observation count."""
        return (
            f"DistrictPopulation("
            f"district='{self.district}', "
            f"observations={len(self)})"
        )

    def growth_rates(self) -> np.ndarray:
        """Return annual growth rates as decimal fractions."""
        # Divide each population increase by the preceding year's value.
        return np.diff(self.populations) / self.populations[:-1]

    def cagr(self) -> float:
        """Return the compound annual growth rate as a decimal."""
        # Count elapsed years, rather than the number of observations.
        periods = self.years[-1] - self.years[0]

        # Find the constant annual rate connecting the endpoint values.
        return (
            (self.populations[-1] / self.populations[0])
            ** (1 / periods)
            - 1
        )
#--- my forecasting base class ---

# Import tools for defining a common interface for forecasting models.
from abc import ABC, abstractmethod


class Forecaster(ABC):
    """Define the methods that every forecasting model must implement."""

    @abstractmethod
    def fit(
        self,
        years: np.ndarray,
        populations: np.ndarray
    ) -> "Forecaster":
        """Learn model parameters from historical observations."""
        # Subclasses supply their own fitting calculations.
        pass

    @abstractmethod
    def predict(self, horizon: int) -> np.ndarray:
        """Return population forecasts for the requested future years."""
        # Subclasses supply their own forecasting calculations.
        pass

#--- my linear trend forecasting class ---

class LinearTrendForecaster(Forecaster):
    """Forecast population assuming a constant annual increase."""

    def fit(
        self,
        years: np.ndarray,
        populations: np.ndarray
    ) -> "LinearTrendForecaster":
        """Fit a straight line to the historical population values."""
        # Fit population = slope × year + intercept using least squares.
        self.slope, self.intercept = np.polyfit(years, populations, 1)

        # Remember the final observed year so forecasts start after it.
        self.last_year = years[-1]

        # Return the fitted object to support Model().fit(...) usage.
        return self

    def predict(self, horizon: int) -> np.ndarray:
        """Predict population for the next horizon years."""
        # Generate consecutive years after the last observation.
        future_years = np.arange(
            self.last_year + 1,
            self.last_year + horizon + 1
        )

        # Apply the fitted straight-line equation to each future year.
        return self.slope * future_years + self.intercept
    
#--- My CAGR Forecasting class ---

class CAGRForecaster(Forecaster):
    """Forecast population assuming a constant annual percentage growth."""

    def fit(
        self,
        years: np.ndarray,
        populations: np.ndarray
    ) -> "CAGRForecaster":
        """Estimate compound growth from the first and last observations."""
        # Count the annual intervals between the endpoint years.
        periods = years[-1] - years[0]

        # Calculate the annual growth rate as a decimal fraction.
        self.cagr = (
            populations[-1] / populations[0]
        ) ** (1 / periods) - 1

        # Future forecasts start from the latest observed population.
        self.last_population = populations[-1]
        return self

    def predict(self, horizon: int) -> np.ndarray:
        """Predict population for the next horizon years."""
        # Number the future annual steps from 1 to the forecast horizon.
        steps = np.arange(1, horizon + 1)

        # Compound the latest population by the same rate each year.
        return self.last_population * (1 + self.cagr) ** steps

#--- My Fibonacci Forecasting Class ---

class FibonacciForecaster(Forecaster):
    """Forecast population by applying successive Fibonacci ratios."""

    def fit(
        self,
        years: np.ndarray,
        populations: np.ndarray
    ) -> "FibonacciForecaster":
        """Store the latest population as the forecast starting point."""
        # This model uses fixed ratios rather than estimating growth.
        self.last_population = populations[-1]
        return self

    def predict(self, horizon: int) -> np.ndarray:
        """Generate forecasts using successive Fibonacci ratios."""
        # Start at 1 and 2 to avoid an initial ratio of 1.
        fibonacci = [1, 2]

        # Generate enough numbers to calculate one ratio per future year.
        while len(fibonacci) < horizon + 1:
            fibonacci.append(fibonacci[-1] + fibonacci[-2])

        # Divide each Fibonacci number by the preceding number.
        ratios = [
            fibonacci[i + 1] / fibonacci[i]
            for i in range(horizon)
        ]

        # Apply each ratio to the previous forecast population.
        forecasts = []
        current = self.last_population

        for ratio in ratios:
            current *= ratio
            forecasts.append(current)

        return np.array(forecasts)

def bootstrap_prediction_interval(
    actual: np.ndarray,
    fitted: np.ndarray,
    forecast: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate pointwise intervals by resampling historical residuals."""
    # Calculate historical errors and centre them around zero.
    residuals = np.asarray(actual) - np.asarray(fitted)
    residuals = residuals - np.mean(residuals)

    # Use a fixed seed to make the simulation reproducible.
    rng = np.random.default_rng(seed)

    # Draw one residual per forecast year in each simulation.
    sampled_residuals = rng.choice(
        residuals,
        size=(n_bootstrap, len(forecast)),
        replace=True
    )

    # Add the sampled errors to the future population forecasts.
    simulated_forecasts = np.asarray(forecast) + sampled_residuals

    # Take the central 95% range separately for each forecast year.
    lower = np.percentile(simulated_forecasts, 2.5, axis=0)
    upper = np.percentile(simulated_forecasts, 97.5, axis=0)

    return lower, upper