## Mini-Project 1: UBOS District Population Forecaster

Forecasts district populations for 2025–2029 to estimate additional
primary-school classroom requirements.

- Notebook: notebooks/project1_population.ipynb
- Reusable classes: src/population_forecaster.py
- Tests: tests/test_population_forecaster.py

Run the notebook from top to bottom. It analyses illustrative population
data for Kampala, Wakiso, Gulu, Mbarara and Jinja over 2015–2024.

Linear trend, CAGR and Fibonacci-ratio models are trained on 2015–2021
and evaluated on 2022–2024 using MAE, RMSE and MAPE. CAGR achieves the
lowest test MAPE for all five districts and is refitted to the full
historical dataset before forecasting.

Wakiso has the fastest historical relative growth at 6.47% annually.
Estimated additional classroom requirements by 2029 are:

- Kampala: 1,544
- Wakiso: 2,088
- Gulu: 412
- Mbarara: 695
- Jinja: 712

These estimates assume 18% of the population is of primary-school age,
53 pupils per classroom and that the estimated 2024 classroom
requirement is already met.

The extension uses 1,000 seeded residual resamples to produce approximate
95% pointwise prediction intervals, shaded on the forecast charts.

The data are illustrative, not verified UBOS estimates. Forecasts assume
historical growth continues, and the intervals exclude growth-rate
uncertainty. Classroom estimates do not account for existing shortages.

Validation: three unit tests passed, and the notebook completed
Restart & Run All successfully.

## Mini-Project 2: Solar Micro-Grid Dispatch Planner

Models solar and battery usage for a rural health centre in Kasese.

- Notebook: notebooks/project2_microgrid.ipynb
- Reusable class: src/microgrid.py
- Generated data: data/kasese_demand.csv
- Tests: tests/test_microgrid.py

Run the notebook from top to bottom. It generates reproducible demand
data using seed 2026 and demonstrates input validation automatically.

The loop and vectorised solvers produced matching results. Eight days
had physically infeasible exact solutions. Non-negative least squares
provided approximate allocations, with remaining demand mismatches
reported explicitly.

The adjusted monthly energy cost was UGX 2,905,485.36, excluding backup
supply for unmet demand. Battery usage varied more than solar usage.
The extension explores demand sensitivity using 1,000 simulated scenarios.

Validation: three unit tests passed, and the notebook completed
Restart & Run All successfully.

## AI Assistance

I used ChatGPT for step-by-step guidance on code structure, input
validation, testing and interpretation. I ran the code and tests locally
and checked the resulting outputs.

## Mini-Project 3: Lake Victoria Fish Stock & Export Risk Model

Models fish stock, harvesting and revenue for an illustrative
fish-export cooperative in Jinja.

- Notebook: notebooks/project3_fishstock.ipynb
- Reusable classes: src/fishstock.py
- Tests: tests/test_fishstock.py

Run the notebook from top to bottom. Random price simulations use
fixed seeds for reproducibility.

The project compares four harvest fractions using logistic growth.
The 20% weekly fraction achieves the model's maximum sustainable
yield of 1,000 tonnes per week at a stock of 5,000 tonnes.

For the 10% scenario, 1,000 price paths produced a 5th-percentile annual
revenue of approximately UGX 372.90 billion. This is a lower-tail
revenue threshold, not a loss amount.

The extension compares five years with and without an eight-week
annual closure. At 20% harvesting, closure increased average stock
by approximately 14.7% but reduced revenue by 8.98% on the shared
price path.

All figures are illustrative. Revenue excludes costs, and biological
and market assumptions limit practical use.

Validation: three unit tests passed, and the notebook completed
Restart & Run All successfully.
