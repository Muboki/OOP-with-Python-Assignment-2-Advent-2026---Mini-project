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
