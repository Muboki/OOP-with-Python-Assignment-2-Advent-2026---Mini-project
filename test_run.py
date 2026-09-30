import numpy as np

from src.population_forecaster import (
    DistrictPopulation,
    LinearTrendForecaster,
    CAGRForecaster,
    FibonacciForecaster
)

years = np.arange(2015, 2025)

population = np.array([
    1200, 1250, 1300, 1350, 1420,
    1500, 1580, 1650, 1720, 1800
])

district = DistrictPopulation(
    "Kampala",
    years,
    population
)

print(district)
print("Number of observations:", len(district))

linear = LinearTrendForecaster()
linear.fit(years, population)
print("Linear forecast:", linear.predict(5))

cagr = CAGRForecaster()
cagr.fit(years, population)
print("CAGR forecast:", cagr.predict(5))

fibonacci = FibonacciForecaster()
fibonacci.fit(years, population)
print("Fibonacci forecast:", fibonacci.predict(5))