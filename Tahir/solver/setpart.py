"""
Phase 3: Set Partitioning solver using PuLP (CPLEX substitute).
Solves the Crew Pairing Problem:

  min  Σ_p c_p * x_p
  s.t. Σ_p a_{fp} * x_p = 1    ∀ flight f
       x_p ∈ {0, 1}

where a_{fp} = 1 if pairing p covers flight f.
"""

from __future__ import annotations
import time
from typing import List, Dict, Tuple

import pulp
import numpy as np


def solve_set_partitioning(
    flights: List[int],
    columns: List[List[int]],
    costs: List[float],
    time_limit: int = 300,
    relaxed: bool = False,
) -> Tuple[List[int], float, str]:
    """
    Solve the set partitioning problem.

    Args:
        flights:    list of all flight_ids to be covered
        columns:    list of pairings, each a list of flight_ids
        costs:      cost per pairing column
        time_limit: solver time limit (seconds)
        relaxed:    if True, solve LP relaxation

    Returns:
        (selected_column_indices, objective_value, status)
    """
    if not columns:
        return [], float("inf"), "INFEASIBLE_NO_COLUMNS"

    flight_set  = set(flights)
    n_cols      = len(columns)

    # Filter columns to only include flights in our flight set
    valid_cols  = []
    valid_costs = []
    for col, cost in zip(columns, costs):
        covered = [f for f in col if f in flight_set]
        if covered:
            valid_cols.append(covered)
            valid_costs.append(cost)

    if not valid_cols:
        return [], float("inf"), "INFEASIBLE_NO_VALID_COLUMNS"

    # Pre-index: for each flight, which columns cover it
    flight_to_cols: Dict[int, List[int]] = {f: [] for f in flight_set}
    for i, col in enumerate(valid_cols):
        for f in col:
            if f in flight_to_cols:
                flight_to_cols[f].append(i)

    # Only include flights that appear in at least one column
    coverable = sorted(f for f, cols in flight_to_cols.items() if cols)

    if not coverable:
        return [], float("inf"), "INFEASIBLE_NO_COVERABLE", 0.0

    # Build problem
    prob = pulp.LpProblem("CrewPairing_SP", pulp.LpMinimize)

    var_type = pulp.LpContinuous if relaxed else pulp.LpBinary
    x = [pulp.LpVariable(f"x_{i}", cat=var_type, lowBound=0, upBound=1)
         for i in range(len(valid_cols))]

    # Objective: minimise total cost
    prob += pulp.lpSum(valid_costs[i] * x[i] for i in range(len(valid_cols)))

    # Coverage constraints: each coverable flight must be covered exactly once
    for f in coverable:
        cols_covering_f = flight_to_cols[f]
        # Set partitioning: = 1 (each flight covered exactly once)
        prob += pulp.lpSum(x[i] for i in cols_covering_f) == 1, f"cover_{f}"

    # Solve
    solver = pulp.PULP_CBC_CMD(timeLimit=time_limit, msg=0)
    t0     = time.time()
    prob.solve(solver)
    solve_time = time.time() - t0

    status = pulp.LpStatus[prob.status]
    obj    = pulp.value(prob.objective) or float("inf")

    selected = [i for i in range(len(valid_cols)) if pulp.value(x[i]) and pulp.value(x[i]) > 0.5]

    return selected, obj, status, solve_time


def compute_coverage_gap(
    flights: List[int],
    selected_columns: List[int],
    columns: List[List[int]],
) -> float:
    """
    Compute the fraction of flights covered by selected pairings.
    Returns gap = 1 - coverage_rate (0 = perfect, 1 = nothing covered).
    """
    covered = set()
    for idx in selected_columns:
        covered.update(columns[idx])
    n_covered = len(covered.intersection(set(flights)))
    return 1.0 - n_covered / len(flights) if flights else 0.0
