"""
LP/MIP solver for CPP Set Partitioning.

LP relaxation  → scipy HiGHS (always available, returns dual values)
MIP (integer)  → Gurobi WLS (preferred) or PuLP CBC (fallback)

Gurobi credentials are loaded from the environment variable
GRB_LICENSE_FILE or from the project-level gurobi.lic file.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Gurobi setup ──────────────────────────────────────────────────────────────

# Point to the project-level WLS license if not already set
_PROJECT_LIC = Path(__file__).parent.parent / "gurobi.lic"
if _PROJECT_LIC.exists() and "GRB_LICENSE_FILE" not in os.environ:
    os.environ["GRB_LICENSE_FILE"] = str(_PROJECT_LIC)

_GUROBI_ENV = None          # shared Env; created once


def _gurobi_env():
    global _GUROBI_ENV
    if _GUROBI_ENV is not None:
        return _GUROBI_ENV
    try:
        import gurobipy as gp
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.setParam("LogToConsole", 0)
        env.start()
        _GUROBI_ENV = env
        return env
    except Exception:
        return None


def _gurobi_available() -> bool:
    return _gurobi_env() is not None


# ── LP relaxation with dual values (scipy HiGHS) ─────────────────────────────

def solve_lp(
    flight_ids: List[int],
    columns:    List[List[int]],
    costs:      List[float],
    M:          float = 1e7,
) -> Tuple[float, float, Dict[int, float], float]:
    """
    Solve LP relaxation of set-partitioning with artificial variables.

    For each flight f we add an artificial variable a_f (cost M) so the LP
    is always feasible:
        min  sum_p c_p x_p  +  M sum_f a_f
        s.t. sum_p A_{fp} x_p + a_f = 1   for all f
             x_p >= 0,  a_f >= 0

    Args:
        flight_ids: list of all flight IDs to cover
        columns:    list of pairings (each a list of flight IDs)
        costs:      cost per pairing
        M:          big-M cost for artificial variables

    Returns:
        (lp_real_obj, lp_total_obj, duals, solve_time)
        lp_real_obj:  sum(c_p * x_p) for real columns only — valid LP lower bound
                      for the Gap calculation (excludes artificial penalty M * a_f)
        lp_total_obj: full LP objective including M * a_f terms; > lp_real_obj
                      whenever some flights are uncovered (artificials active)
        duals:        {flight_id: dual_value}
    """
    from scipy.optimize import linprog

    n_flights = len(flight_ids)
    fid2row   = {fid: r for r, fid in enumerate(flight_ids)}
    n_cols    = len(columns)

    # Build coverage matrix A  (n_flights x n_cols)
    # Each column p: A[f, p] = 1 if f in columns[p]
    c_real = np.array(costs, dtype=float)
    c_art  = np.full(n_flights, M, dtype=float)
    c_all  = np.concatenate([c_real, c_art])

    # Equality constraints: A_eq @ x = b_eq  (b_eq = 1 for each flight)
    # Real columns: sparse; artificial: identity
    rows, col_idx, vals = [], [], []
    for j, col in enumerate(columns):
        for fid in col:
            r = fid2row.get(fid)
            if r is not None:
                rows.append(r)
                col_idx.append(j)
                vals.append(1.0)
    # Artificial columns: identity block appended after real columns
    for r in range(n_flights):
        rows.append(r)
        col_idx.append(n_cols + r)
        vals.append(1.0)

    from scipy.sparse import csr_matrix
    A_eq = csr_matrix(
        (vals, (rows, col_idx)),
        shape=(n_flights, n_cols + n_flights),
    ).toarray()
    b_eq = np.ones(n_flights)

    bounds = [(0, None)] * (n_cols + n_flights)

    t0 = time.time()
    res = linprog(c_all, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    solve_time = time.time() - t0

    if not res.success:
        return float("inf"), float("inf"), {fid: 0.0 for fid in flight_ids}, solve_time

    lp_total_obj = float(res.fun)
    # Real LP obj = pairing-cost portion only (excludes M * artificials)
    lp_real_obj  = float(np.dot(c_real, res.x[:n_cols]))
    # res.eqlin.marginals[r] = dual value for constraint r (= dual for flight fid2row[r])
    marginals = res.eqlin.marginals
    duals = {fid: float(marginals[r]) for fid, r in fid2row.items()}

    return lp_real_obj, lp_total_obj, duals, solve_time


# ── MIP (set partitioning) ────────────────────────────────────────────────────

def solve_mip(
    flight_ids:  List[int],
    columns:     List[List[int]],
    costs:       List[float],
    time_limit:  int  = 300,
    verbose:     bool = False,
    A_ub:        Optional[np.ndarray] = None,
    b_ub:        Optional[np.ndarray] = None,
) -> Tuple[List[int], float, str, float]:
    """
    Solve integer set-partitioning problem.

    Optional A_ub @ x <= b_ub adds availability constraints (crew limits).
    Tries Gurobi first; falls back to PuLP CBC.

    Returns:
        (selected_column_indices, obj_value, status_str, solve_time)
    """
    env = _gurobi_env()
    if env is not None:
        return _solve_mip_gurobi(
            flight_ids, columns, costs, time_limit, verbose, env,
            A_ub=A_ub, b_ub=b_ub,
        )
    return _solve_mip_pulp(flight_ids, columns, costs, time_limit)


def _solve_mip_gurobi(
    flight_ids: List[int],
    columns:    List[List[int]],
    costs:      List[float],
    time_limit: int,
    verbose:    bool,
    env,
    M:     float                    = 1e7,
    A_ub:  Optional[np.ndarray]     = None,
    b_ub:  Optional[np.ndarray]     = None,
) -> Tuple[List[int], float, str, float]:
    """
    Set-partitioning MIP with binary artificial variables (big-M penalty).

    Formulation:
        min  Σ c_p x_p  +  M Σ a_f
        s.t. Σ_{p: f∈p} x_p + a_f = 1   ∀ f with at least one covering column
             x_p ∈ {0,1},  a_f ∈ {0,1}

    The artificial variable a_f=1 means flight f is LEFT UNCOVERED (cost M).
    This ensures feasibility when the column pool alone cannot partition all
    covered flights (e.g. when newly priced columns overlap existing ones).
    """
    import gurobipy as gp
    from gurobipy import GRB

    fid2row  = {fid: r for r, fid in enumerate(flight_ids)}
    n_cols   = len(columns)

    # Build coverage index: {fid: [col_indices]}
    fid2cols: Dict[int, List[int]] = {}
    for j, col in enumerate(columns):
        for fid in col:
            if fid in fid2row:
                fid2cols.setdefault(fid, []).append(j)

    # Only flights with at least one covering column get a constraint
    constrained_fids = [fid for fid in flight_ids if fid in fid2cols]

    t0 = time.time()
    m  = gp.Model(env=env)
    m.setParam("OutputFlag",   int(verbose))
    m.setParam("TimeLimit",    time_limit)
    m.setParam("LogToConsole", int(verbose))

    x = m.addVars(n_cols, vtype=GRB.BINARY, obj=costs, name="x")
    # Binary artificial: a_f=1 means flight uncovered (penalised by M)
    a = m.addVars(len(constrained_fids), vtype=GRB.BINARY,
                  obj=[M] * len(constrained_fids), name="a")

    for k, fid in enumerate(constrained_fids):
        m.addConstr(
            gp.quicksum(x[j] for j in fid2cols[fid]) + a[k] == 1,
            name=f"cov_{fid}",
        )

    # Availability inequalities: Σ_{p starting at base b on day d} x_p <= cap
    if A_ub is not None and b_ub is not None:
        import gurobipy as gp2
        for row_i in range(len(b_ub)):
            col_indices = [int(j) for j in range(n_cols) if A_ub[row_i, j] > 0.5]
            if col_indices:
                m.addConstr(
                    gp.quicksum(x[j] for j in col_indices) <= float(b_ub[row_i]),
                    name=f"avail_{row_i}",
                )

    m.ModelSense = GRB.MINIMIZE
    m.optimize()

    solve_time = time.time() - t0
    status_map = {
        GRB.OPTIMAL:    "Optimal",
        GRB.INFEASIBLE: "Infeasible",
        GRB.INF_OR_UNBD:"InfOrUnbd",
        GRB.TIME_LIMIT: "TimeLimit",
    }
    status = status_map.get(m.Status, f"Unknown({m.Status})")

    if m.SolCount == 0:
        return [], float("inf"), status, solve_time

    selected = [j for j in range(n_cols) if x[j].X > 0.5]
    # Report true pairing cost only (exclude artificial penalty from displayed obj)
    obj = float(sum(costs[j] for j in selected))
    return selected, obj, status, solve_time


def _solve_mip_pulp(
    flight_ids: List[int],
    columns:    List[List[int]],
    costs:      List[float],
    time_limit: int,
) -> Tuple[List[int], float, str, float]:
    import pulp

    flight_set = set(flight_ids)
    valid_cols, valid_costs = [], []
    for col, cost in zip(columns, costs):
        covered = [f for f in col if f in flight_set]
        if covered:
            valid_cols.append(covered)
            valid_costs.append(cost)

    if not valid_cols:
        return [], float("inf"), "NoColumns", 0.0

    fid2cols: Dict[int, List[int]] = {f: [] for f in flight_ids}
    for j, col in enumerate(valid_cols):
        for f in col:
            if f in fid2cols:
                fid2cols[f].append(j)

    coverable = [f for f in flight_ids if fid2cols[f]]

    prob = pulp.LpProblem("CPP_MIP", pulp.LpMinimize)
    x = [pulp.LpVariable(f"x_{j}", cat=pulp.LpBinary) for j in range(len(valid_cols))]
    prob += pulp.lpSum(valid_costs[j] * x[j] for j in range(len(valid_cols)))
    for f in coverable:
        prob += pulp.lpSum(x[j] for j in fid2cols[f]) == 1, f"cov_{f}"

    t0 = time.time()
    pulp.PULP_CBC_CMD(timeLimit=time_limit, msg=0).solve(prob)
    solve_time = time.time() - t0

    status = pulp.LpStatus[prob.status]
    obj    = pulp.value(prob.objective) or float("inf")
    selected = [j for j in range(len(valid_cols))
                if pulp.value(x[j]) and pulp.value(x[j]) > 0.5]
    return selected, obj, status, solve_time


# ── Convenience: solve set covering (>= 1) for quick coverage check ──────────

def solve_lp_full(
    flight_ids: List[int],
    columns:    List[List[int]],
    costs:      List[float],
    M:          float = 1e7,
) -> Tuple[float, float, Dict[int, float], List[float], float]:
    """
    Like solve_lp but also returns LP solution values for real columns.

    Returns:
        (lp_real_obj, lp_total_obj, duals, x_real, solve_time)
        x_real: list of LP solution values for each column in `columns`
    """
    from scipy.optimize import linprog

    n_flights = len(flight_ids)
    fid2row   = {fid: r for r, fid in enumerate(flight_ids)}
    n_cols    = len(columns)

    c_real = np.array(costs, dtype=float)
    c_art  = np.full(n_flights, M, dtype=float)
    c_all  = np.concatenate([c_real, c_art])

    rows, col_idx, vals = [], [], []
    for j, col in enumerate(columns):
        for fid in col:
            r = fid2row.get(fid)
            if r is not None:
                rows.append(r)
                col_idx.append(j)
                vals.append(1.0)
    for r in range(n_flights):
        rows.append(r)
        col_idx.append(n_cols + r)
        vals.append(1.0)

    from scipy.sparse import csr_matrix
    A_eq = csr_matrix(
        (vals, (rows, col_idx)),
        shape=(n_flights, n_cols + n_flights),
    ).toarray()
    b_eq = np.ones(n_flights)
    bounds = [(0, None)] * (n_cols + n_flights)

    t0 = time.time()
    res = linprog(c_all, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    solve_time = time.time() - t0

    if not res.success:
        return (
            float("inf"), float("inf"),
            {fid: 0.0 for fid in flight_ids},
            [0.0] * n_cols,
            solve_time,
        )

    lp_total_obj = float(res.fun)
    lp_real_obj  = float(np.dot(c_real, res.x[:n_cols]))
    marginals     = res.eqlin.marginals
    duals = {fid: float(marginals[r]) for fid, r in fid2row.items()}
    x_real = list(res.x[:n_cols])

    return lp_real_obj, lp_total_obj, duals, x_real, solve_time


def solve_lp_partition(
    flight_ids:   List[int],
    columns:      List[List[int]],
    costs:        List[float],
    A_ub:         Optional[np.ndarray] = None,
    b_ub:         Optional[np.ndarray] = None,
) -> Tuple[float, Dict[int, float], List[float], float]:
    """
    Pure LP relaxation of set-partitioning (NO artificials).

    Optionally accepts availability inequality constraints  A_ub @ x <= b_ub
    (crew-availability constraints from the CPPSC/CPP instances).  Adding
    these makes the LP genuinely fractional, giving meaningful LP-MIP gaps.

    All flight_ids must be covered by at least one column; caller is responsible
    for filtering to coverable flights before calling this.

    Returns:
        (lp_obj, duals, x_vals, solve_time)
        lp_obj : LP objective (valid lower bound for set-partitioning MIP)
        duals  : {flight_id: dual_value} (from equality constraints only)
        x_vals : LP solution values for each column
    """
    from scipy.optimize import linprog

    n_flights = len(flight_ids)
    fid2row   = {fid: r for r, fid in enumerate(flight_ids)}
    n_cols    = len(columns)

    c_vec = np.array(costs, dtype=float)

    rows, col_idx, vals = [], [], []
    for j, col in enumerate(columns):
        for fid in col:
            r = fid2row.get(fid)
            if r is not None:
                rows.append(r)
                col_idx.append(j)
                vals.append(1.0)

    from scipy.sparse import csr_matrix
    A_eq = csr_matrix(
        (vals, (rows, col_idx)), shape=(n_flights, n_cols)
    ).toarray()
    b_eq = np.ones(n_flights)
    bounds = [(0, None)] * n_cols

    t0 = time.time()
    res = linprog(
        c_vec,
        A_ub=A_ub, b_ub=b_ub,
        A_eq=A_eq, b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )
    solve_time = time.time() - t0

    if not res.success:
        return float("inf"), {fid: 0.0 for fid in flight_ids}, [0.0] * n_cols, solve_time

    lp_obj    = float(res.fun)
    marginals = res.eqlin.marginals
    duals     = {fid: float(marginals[r]) for fid, r in fid2row.items()}
    x_vals    = list(res.x)

    return lp_obj, duals, x_vals, solve_time


def solve_covering_lp(
    flight_ids: List[int],
    columns:    List[List[int]],
    costs:      List[float],
) -> Tuple[float, Dict[int, float], float]:
    """
    LP relaxation of set COVERING (>= constraints).
    Dual values are non-negative (shadow prices for coverage constraints).
    """
    from scipy.optimize import linprog

    n_flights = len(flight_ids)
    fid2row   = {fid: r for r, fid in enumerate(flight_ids)}
    n_cols    = len(columns)

    rows, col_idx, vals = [], [], []
    for j, col in enumerate(columns):
        for fid in col:
            r = fid2row.get(fid)
            if r is not None:
                rows.append(r)
                col_idx.append(j)
                vals.append(-1.0)   # >= -> -A <= -b

    from scipy.sparse import csr_matrix
    A_ub = csr_matrix(
        (vals, (rows, col_idx)), shape=(n_flights, n_cols)
    ).toarray()
    b_ub = -np.ones(n_flights)

    t0  = time.time()
    res = linprog(costs, A_ub=A_ub, b_ub=b_ub,
                  bounds=[(0, None)] * n_cols, method="highs")
    dt  = time.time() - t0

    if not res.success:
        return float("inf"), {fid: 0.0 for fid in flight_ids}, dt

    # Duals for >= (ineqlin): negate because we flipped the sign
    marginals = -res.ineqlin.marginals
    duals = {fid: float(marginals[r]) for fid, r in fid2row.items()}
    return float(res.fun), duals, dt
