# Define a shared interface for the forecasting models.
from abc import ABC, abstractmethod

# Calculate descriptive statistics using Python's standard library.
from pyexpat import errors, model
import statistics

# Use NumPy for passenger arrays and numerical validation.
import numpy as np


class Route:
    """Store a taxi route's daily passenger counts and fare."""

    def __init__(
        self,
        name: str,
        passengers: np.ndarray,
        fare: float
    ) -> None:
        # Store passenger counts as a numerical array.
        self.name = name
        self.passengers = np.asarray(passengers, dtype=float)

        # Require a non-empty, one-dimensional series.
        if self.passengers.ndim != 1 or self.passengers.size == 0:
            raise ValueError("Provide a non-empty list of daily passengers.")

        # Passenger counts must be finite, non-negative whole numbers.
        if not np.all(np.isfinite(self.passengers)):
            raise ValueError("Passenger counts must be finite.")
        if np.any(self.passengers < 0):
            raise ValueError("Passenger counts cannot be negative.")
        if np.any(self.passengers != np.floor(self.passengers)):
            raise ValueError("Passenger counts must be whole numbers.")

        # Require a positive fare in Uganda shillings per passenger.
        if not np.isfinite(fare) or fare <= 0:
            raise ValueError("Fare must be finite and greater than zero.")

        self.fare = float(fare)
    def daily_revenue(self) -> np.ndarray:
        """Return each day's gross fare revenue in UGX."""
        # Multiply each day's passenger count by the fare.
        return self.passengers * self.fare
    def total_revenue(self) -> float:
        """Return gross fare revenue across all recorded days in UGX."""
        # Add the daily revenues for the full observation period.
        return float(np.sum(self.daily_revenue()))
    def passenger_statistics(self) -> dict[str, float]:
        """Summarise passenger counts across the recorded days."""
        # Convert the array into a list for the statistics module.
        counts = self.passengers.tolist()

        # Use population statistics to describe all recorded days.
        return {
            "Mean": statistics.mean(counts),
            "Variance": statistics.pvariance(counts),
            "Standard deviation": statistics.pstdev(counts)
        }

class Forecaster(ABC):
    """Define the interface for forecasting the next day's passengers."""

    @abstractmethod
    def predict(self, history: np.ndarray) -> float:
        """Predict the next day using only earlier passenger counts."""
        pass

    def validate_history(self, history: np.ndarray) -> np.ndarray:
        """Convert historical demand into a valid numerical array."""
        # Require a non-empty, one-dimensional sequence.
        values = np.asarray(history, dtype=float)
        if values.ndim != 1 or values.size == 0:
            raise ValueError("Provide a non-empty passenger history.")

        # Reject missing, infinite or negative demand values.
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError("Passenger history must be finite and non-negative.")

        return values

class MovingAverageForecaster(Forecaster):
    """Predict the next day's passengers from the latest three days."""

    def predict(self, history: np.ndarray) -> float:
        """Return the mean passenger count over the last three days."""
        # Validate the history and require at least three observations.
        values = self.validate_history(history)
        if len(values) < 3:
            raise ValueError("Moving average requires at least three days.")

        # Use only the three most recent passenger counts.
        return float(np.mean(values[-3:]))    

class ExponentialSmoothingForecaster(Forecaster):
    """Forecast passengers using a weighted update of the demand level."""

    def __init__(self, alpha: float = 0.5) -> None:
        # Alpha controls how strongly recent observations affect the level.
        if not np.isfinite(alpha) or not 0 < alpha <= 1:
            raise ValueError("Alpha must be greater than zero and at most one.")

        self.alpha = float(alpha)

    def predict(self, history: np.ndarray) -> float:
        """Return the final smoothed level as the next-day forecast."""
        # Validate the history and initialise from the first observation.
        values = self.validate_history(history)
        level = float(values[0])

        # Update the level using each subsequent passenger count.
        for passengers in values[1:]:
            level = (
                self.alpha * passengers
                + (1 - self.alpha) * level
            )

        return float(level)

class LinearTrendForecaster(Forecaster):
    """Forecast passengers by extending a fitted straight-line trend."""

    def predict(self, history: np.ndarray) -> float:
        """Fit the available history and predict the next day's demand."""
        # Require at least two observations to estimate a trend.
        values = self.validate_history(history)
        if len(values) < 2:
            raise ValueError("Linear trend requires at least two days.")

        # Fit passenger count = slope × day + intercept.
        days = np.arange(1, len(values) + 1)
        slope, intercept = np.polyfit(days, values, 1)

        # Extend the trend by one day.
        prediction = slope * (len(values) + 1) + intercept

        # Passenger demand cannot be negative.
        return max(0.0, float(prediction))


class SeasonalNaiveForecaster(Forecaster):
    """Predict demand using the same weekday from the previous week."""

    def predict(self, history: np.ndarray) -> float:
        """Return the passenger count from seven days before the forecast."""
        # Require a complete week of historical observations.
        values = self.validate_history(history)
        if len(values) < 7:
            raise ValueError("Seasonal naive requires at least seven days.")

        # The seventh value from the end matches the next day's weekday.
        return float(values[-7])

    
def walk_forward(
    model: Forecaster,
    passengers: np.ndarray
) -> tuple[np.ndarray, float]:
    """Return next-day predictions and their mean absolute error."""

    # Validate the data and require at least four days.
    values = model.validate_history(passengers)

    if len(values) < 4:
        raise ValueError(
            "Walk-forward evaluation requires at least four days."
        )

    # Predict each day using only earlier observations.
    daily_predictions = []

    for day_index in range(3, len(values)):
        history = values[:day_index]
        prediction = model.predict(history)
        daily_predictions.append(prediction)

    # Match the predictions with actual counts from day 4 onward.
    predictions = np.asarray(daily_predictions, dtype=float)
    actual = values[3:]

    # Calculate the average absolute prediction error.
    errors = np.abs(actual - predictions)
    mae = float(np.mean(errors))

    return predictions, mae