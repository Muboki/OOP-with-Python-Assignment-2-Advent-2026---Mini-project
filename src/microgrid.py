
# Find the closest solution with non-negative energy usage.
from scipy.optimize import nnls

# Import NumPy to work with arrays and matrices.

import numpy as np


class MicroGrid:
    """Plan solar and battery usage for a health centre."""

    def __init__(self, centre_name: str) -> None:
        # Store the name of the health centre.
        self.centre_name = centre_name

        # Store the coefficients of 3x + 2y = D1 and 4x + y = D2.
        self.coefficients = np.array([
            [3.0, 2.0],
            [4.0, 1.0]
        ])
    def determinant(self) -> float:
        """Return the determinant of the coefficient matrix."""
        # A non-zero determinant means there is a unique solution.
        return float(np.linalg.det(self.coefficients))

    def condition_number(self) -> float:
        """Measure how sensitive the solution is to demand changes."""
        # A large condition number suggests greater sensitivity.
        return float(np.linalg.cond(self.coefficients))
    def solve_day(self, d1: float, d2: float) -> np.ndarray:
        """Calculate solar and battery usage for one day."""
        # Convert both demand values into a NumPy array.
        demand = np.array([d1, d2], dtype=float)

        # Reject negative demands, NaN and infinity; zero is allowed.
        if not np.all(np.isfinite(demand)) or np.any(demand < 0):
            raise ValueError("Demand values must be finite and non-negative.")

        # Check that the matrix has a unique solution.
        if np.isclose(self.determinant(), 0.0):
            raise ValueError("The demand equations do not have a unique solution.")

        # Check numerical sensitivity before solving.
        if not np.isfinite(self.condition_number()):
            raise ValueError("The coefficient matrix is singular.")

        # Return solar usage first, followed by battery usage.
        return solve(self.coefficients, demand)
    def interactive_solve(self) -> np.ndarray:
        """Read valid demand values and calculate energy usage."""
        demand_values = []

        # Collect each demand separately so only invalid entries are repeated.
        for label in ["daytime", "critical-equipment"]:
            while True:
                entry = input(f"Enter {label} demand (kWh): ").strip()

                # Reject empty entries and text that cannot become a number.
                try:
                    value = float(entry)
                except ValueError:
                    print("Please enter a number, such as 1200.")
                    continue

                # Reject negative values, infinity and NaN.
                if not np.isfinite(value) or value < 0:
                    print("Enter a finite number that is zero or greater.")
                    continue

                # Keep the valid value and move to the next demand.
                demand_values.append(value)
                break

        # Solve after both demand values have passed validation.
        return self.solve_day(demand_values[0], demand_values[1])
    def load_demands(self, file_path: str) -> np.ndarray:
        """Load daytime and critical demands from a 30-day CSV."""
        # Read the numeric rows, skipping the column headings.
        data = np.loadtxt(
            file_path, delimiter=",", skiprows=1, ndmin=2
        )

        # Require 30 rows with day, daytime demand and critical demand.
        if data.shape != (30, 3):
            raise ValueError("The CSV must contain 30 rows and 3 columns.")

        # Check that all values are finite and days run from 1 to 30.
        if not np.all(np.isfinite(data)):
            raise ValueError("The CSV must contain only finite numbers.")

        if not np.array_equal(data[:, 0], np.arange(1, 31)):
            raise ValueError("Days must be numbered from 1 to 30 in order.")

        # Extract the two demand columns and reject negative values.
        demands = data[:, 1:]
        if np.any(demands < 0):
            raise ValueError("Demand values cannot be negative.")

        # Return one row per day, with two demand values per row.
        return demands
    def solve_month_loop(self, demands: np.ndarray) -> np.ndarray:
        """Calculate daily solar and battery usage using a loop."""
        # Require a non-empty array with two demand columns.
        demands = np.asarray(demands, dtype=float)
        if demands.ndim != 2 or demands.shape[1] != 2 or len(demands) == 0:
            raise ValueError("Provide one or more rows with two demand values.")

        # Solve each day's equations using the existing method.
        daily_results = []
        for daytime, critical in demands:
            usage = self.solve_day(daytime, critical)
            daily_results.append(usage)

        # Return one row per day: solar usage, then battery usage.
        return np.array(daily_results)
    def solve_month_vectorised(self, demands: np.ndarray) -> np.ndarray:
        """Solve all days together in one SciPy call."""
        # Require a non-empty array with two demand columns.
        demands = np.asarray(demands, dtype=float)
        if demands.ndim != 2 or demands.shape[1] != 2 or len(demands) == 0:
            raise ValueError("Provide one or more rows with two demand values.")

        # Reject invalid demand values before solving.
        if not np.all(np.isfinite(demands)) or np.any(demands < 0):
            raise ValueError("Demand values must be finite and non-negative.")

        # Check that the coefficient matrix can be solved.
        if np.isclose(self.determinant(), 0.0):
            raise ValueError("The demand equations do not have a unique solution.")
        if not np.isfinite(self.condition_number()):
            raise ValueError("The coefficient matrix is singular.")

        # Transpose demands to 2 × 30, solve, then return 30 × 2 results.
        return solve(self.coefficients, demands.T).T
    def solve_month_nonnegative(self, demands: np.ndarray) -> np.ndarray:
        """Replace infeasible solutions with non-negative approximations."""
        # Get the exact solutions using the existing validated method.
        usage = self.solve_month_vectorised(demands)
        demands = np.asarray(demands, dtype=float)

        # Identify days with materially negative solar or battery usage.
        flagged_days = np.any(usage < -1e-9, axis=1)

        # Find the closest non-negative solution for each flagged day.
        for index in np.flatnonzero(flagged_days):
            solution, _ = nnls(self.coefficients, demands[index])
            usage[index] = solution

        # Remove any tiny negative values caused by numerical rounding.
        return np.maximum(usage, 0.0)

# Import the solver for the simultaneous demand equations.
from scipy.linalg import solve
    


    