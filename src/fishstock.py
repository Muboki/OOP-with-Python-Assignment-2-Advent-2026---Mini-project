# Calculate revenue averages and variability.
import statistics

# Import NumPy for numerical calculations and input checks.
import numpy as np


class FishStock:
    """Model fish stock growth and harvesting for a cooperative."""

    def __init__(
        self,
        cooperative_name: str,
        growth_rate: float = 0.4,
        carrying_capacity: float = 10000.0,
        initial_stock: float = 4000.0
    ) -> None:
        # Reject missing or infinite numerical values.
        values = [growth_rate, carrying_capacity, initial_stock]
        if not np.all(np.isfinite(values)):
            raise ValueError("Stock parameters must be finite numbers.")

        # Check that the growth rate and stock values are meaningful.
        if growth_rate < 0:
            raise ValueError("Growth rate cannot be negative.")
        if carrying_capacity <= 0:
            raise ValueError("Carrying capacity must be greater than zero.")
        if initial_stock < 0:
            raise ValueError("Initial stock cannot be negative.")

        # Store the cooperative name and model parameters.
        self.cooperative_name = cooperative_name
        self.growth_rate = growth_rate
        self.carrying_capacity = carrying_capacity
        self.initial_stock = initial_stock
    def simulate(
        self,
        harvest_rate: float,
        weeks: int = 52,
        closed_season: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simulate stock and harvest, optionally closing eight weeks yearly."""
        # Require a valid harvest fraction and a positive whole week count.
        if not np.isfinite(harvest_rate) or not 0 <= harvest_rate <= 1:
            raise ValueError("Harvest rate must be between 0 and 1.")
        if isinstance(weeks, bool) or not isinstance(weeks, int) or weeks < 1:
            raise ValueError("Weeks must be a positive integer.")

        # Include the initial stock at week 0, plus each week's result.
        stock = np.zeros(weeks + 1)
        harvest = np.zeros(weeks)
        stock[0] = self.initial_stock

        for week in range(weeks):
            current_stock = stock[week]

            # Calculate natural growth using the logistic equation.
            growth = (
                self.growth_rate
                * current_stock
                * (1 - current_stock / self.carrying_capacity)
            )

            # Close fishing during weeks 1–8 of each 52-week year.
            fishing_closed = closed_season and (week % 52 < 8)

            # Allow natural growth during closure, but remove no fish.
            active_rate = 0.0 if fishing_closed else harvest_rate
            harvest[week] = active_rate * current_stock
            next_stock = current_stock + growth - harvest[week]

            # Flag parameters that produce an impossible negative stock.
            if not np.isfinite(next_stock) or next_stock < 0:
                raise ValueError("These parameters produce invalid fish stock.")

            stock[week + 1] = next_stock

        return stock, harvest

class PriceModel:
    """Simulate weekly fish prices in UGX per kilogram."""

    def __init__(
        self,
        starting_price: float = 12000.0,
        minimum_price: float = 9000.0,
        maximum_price: float = 16000.0,
        weekly_change_sd: float = 500.0
    ) -> None:
        # Reject non-finite price parameters.
        values = [
            starting_price, minimum_price,
            maximum_price, weekly_change_sd
        ]
        if not np.all(np.isfinite(values)):
            raise ValueError("Price parameters must be finite numbers.")

        # Ensure positive bounds and a starting price within them.
        if not 0 < minimum_price <= starting_price <= maximum_price:
            raise ValueError("Starting price must lie within positive bounds.")
        if weekly_change_sd < 0:
            raise ValueError("Weekly price-change SD cannot be negative.")

        # Store prices and the size of simulated weekly fluctuations.
        self.starting_price = starting_price
        self.minimum_price = minimum_price
        self.maximum_price = maximum_price
        self.weekly_change_sd = weekly_change_sd
    def simulate(self, weeks: int = 52, seed: int = 2026) -> np.ndarray:
        """Return weekly prices from a bounded random walk."""
        # Require a positive whole number of weeks.
        if isinstance(weeks, bool) or not isinstance(weeks, int) or weeks < 1:
            raise ValueError("Weeks must be a positive integer.")

        # Fix the seed so the same inputs reproduce the same price path.
        rng = np.random.default_rng(seed)
        prices = np.zeros(weeks)
        prices[0] = self.starting_price

        # Add a random change to the previous week's price.
        for week in range(1, weeks):
            change = rng.normal(0, self.weekly_change_sd)
            proposed_price = prices[week - 1] + change

            # Keep the resulting price within the chosen bounds.
            prices[week] = np.clip(
                proposed_price,
                self.minimum_price,
                self.maximum_price
            )

        return prices

class RiskAssessor:
    """Classify relative revenue variability using its CV."""

    def classify(self, revenues: np.ndarray) -> tuple[float, str]:
        """Return the coefficient of variation and a risk category."""
        # Require a non-empty, one-dimensional revenue series.
        values = np.asarray(revenues, dtype=float)
        if values.ndim != 1 or values.size == 0:
            raise ValueError("Provide a non-empty revenue series.")

        # Reject non-finite and negative revenue values.
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError("Revenues must be finite and non-negative.")

        # CV is undefined when mean revenue is zero.
        revenue_list = values.tolist()
        mean_revenue = statistics.mean(revenue_list)
        if mean_revenue == 0:
            raise ValueError("CV is undefined when mean revenue is zero.")

        # Measure variability relative to average revenue.
        cv = statistics.pstdev(revenue_list) / mean_revenue

        # Apply illustrative variability thresholds for this exercise.
        if cv < 0.10:
            category = "Low"
        elif cv < 0.20:
            category = "Moderate"
        else:
            category = "High"

        return cv, category