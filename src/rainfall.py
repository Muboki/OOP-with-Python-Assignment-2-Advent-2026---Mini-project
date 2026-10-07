# Use NumPy to store and analyse monthly rainfall.
import numpy as np


class Region:
    """Store twelve monthly rainfall values for a region."""

    def __init__(self, name: str, rainfall: np.ndarray) -> None:
        # Store the region name and rainfall in millimetres.
        self.name = name
        self.rainfall = np.asarray(rainfall, dtype=float)

        # Require one value for each month, ordered January to December.
        if self.rainfall.shape != (12,):
            raise ValueError("Provide exactly 12 monthly rainfall values.")

        # Reject missing, infinite and negative rainfall values.
        if not np.all(np.isfinite(self.rainfall)):
            raise ValueError("Rainfall values must be finite numbers.")
        if np.any(self.rainfall < 0):
            raise ValueError("Rainfall values cannot be negative.")
    def annual_total(self) -> float:
        """Return total rainfall across the twelve months in mm."""
        # Add all monthly rainfall values.
        return float(np.sum(self.rainfall))

    def mean(self) -> float:
        """Return average monthly rainfall in mm."""
        # Divide the annual total by the number of months.
        return float(np.mean(self.rainfall))
    def wettest_month(self) -> str:
        """Return the wettest month's name, choosing the first if tied."""
        # Match the largest rainfall value to its calendar month.
        months = [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ]
        return months[int(np.argmax(self.rainfall))]

    def driest_month(self) -> str:
        """Return the driest month's name, choosing the first if tied."""
        # Match the smallest rainfall value to its calendar month.
        months = [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ]
        return months[int(np.argmin(self.rainfall))]
    def coefficient_of_variation(self) -> float:
        """Return monthly rainfall variability relative to its mean."""
        # CV is undefined when all twelve months have zero rainfall.
        average = self.mean()
        if average == 0:
            raise ValueError("CV is undefined when mean rainfall is zero.")
        
        # Use population standard deviation for the full twelve months.
        return float(np.std(self.rainfall, ddof=0) / average)
    def cosine_similarity(self, other: "Region") -> float:
        """Compare the direction of two monthly rainfall vectors."""
        # Calculate the length (norm) of each region's rainfall vector.
        own_norm = np.linalg.norm(self.rainfall)
        other_norm = np.linalg.norm(other.rainfall)

        # Cosine similarity is undefined for an all-zero vector.
        if own_norm == 0 or other_norm == 0:
            raise ValueError("Cosine similarity requires non-zero rainfall.")

        # Divide the dot product by the product of the vector lengths.
        similarity = np.dot(self.rainfall, other.rainfall) / (
            own_norm * other_norm
        )

        # Keep tiny floating-point errors within the mathematical bounds.
        return float(np.clip(similarity, -1.0, 1.0))

class CropRule:
    """Classify monthly rainfall against a crop's chosen rainfall range."""

    def __init__(
        self,
        crop_name: str,
        minimum_rainfall: float,
        maximum_rainfall: float
    ) -> None:
        # Require finite, non-negative rainfall limits in the correct order.
        limits = [minimum_rainfall, maximum_rainfall]
        if not np.all(np.isfinite(limits)):
            raise ValueError("Rainfall limits must be finite numbers.")
        if not 0 <= minimum_rainfall < maximum_rainfall:
            raise ValueError("Require 0 <= minimum < maximum rainfall.")

        # Store the crop name and monthly rainfall limits in mm.
        self.crop_name = crop_name
        self.minimum_rainfall = minimum_rainfall
        self.maximum_rainfall = maximum_rainfall

    def classify(self, rainfall: float) -> str:
        """Classify a month's rainfall using the stored limits."""
        # Reject invalid rainfall before applying the rule.
        if not np.isfinite(rainfall) or rainfall < 0:
            raise ValueError("Rainfall must be finite and non-negative.")

        # Treat both endpoints as inside the suitable range.
        if rainfall < self.minimum_rainfall:
            return "Drought risk"
        if rainfall > self.maximum_rainfall:
            return "Waterlogging risk"
        return f"Good for {self.crop_name}"