"""
Set Covering for Crew Pairing

Paper Sec. "Problem Setting and Overview", Eq. (2) -- the restricted
set-covering formulation solved here (with d_i playing the role of the
excess-coverage variable z_i, and lambda_dh playing the role of lambda_exc):

  min  sum_j  c_j * x_j  +  lambda_dh * sum_i d_i
  s.t. sum_{j: i in j} x_j  >=  1   for all flight i   (coverage constraint)
       d_i = sum_{j: i in j} x_j - 1                    (excess coverage / deadhead count)
       x_j in {0, 1},  d_i >= 0

Because coverage is >=1 rather than ==1, the same flight may be covered by
multiple selected pairings -> deadheading is allowed rather than forbidden.

3-stage pipeline:
  [1] solve_lp_relaxation : solve the LP relaxation (x_j in [0,1]) -> extract
      the coverage/excess-coverage duals mu_i^cov, nu_i^exc used by Phase II
      LP-dual refinement (Eq. 9-10, Algorithm 1)
  [2] column_reduction     : keep only pairings with reduced cost <= 0 (+ a
      coverage safeguard)
  [3] solve_set_covering   : solve the restricted MIP (x_j in {0,1}) over the
      remaining pairings -- this is the "final mixed-integer program" that
      selects the operational pairing set (Sec. "Scalable Inference and
      Global Selection")

solver: PuLP + CBC (default) | Gurobi (use_gurobi=True)
"""

import pulp
from collections import defaultdict
from typing import Dict, List, Optional


def solve_lp_relaxation(
    pairings: List[Dict],
    lambda_dh: float = 1.0,
    verbose: bool = False,
    flight_ids=None,
    artificial_cost: float = 1000.0,
) -> Optional[Dict]:
    """
    Set-covering LP relaxation of Eq. (2): x_j in [0,1] (continuous) -> extract
    dual variables mu_i^cov (coverage) and nu_i^exc (excess coverage) -> reduced costs.

    The LP is built with the same objective as the IP (including the DH
    penalty) so that column reduction has a valid theoretical justification.

    reduced cost: rc_j = c_j - sum_{i in legs_j} mu_i^cov
      - rc_j < 0: including this pairing would lower the objective -> worth
        keeping for the IP
      - rc_j >= 0: unlikely to be in the optimal solution -> eligible for
        removal by column_reduction

    Returns: { lp_value, dual_vars, reduced_costs, status }
    None: if the LP fails to solve
    """
    if not pairings and flight_ids is None:
        return None

    real_pairing_count = len(pairings)
    if flight_ids is not None:
        present = {leg for p in pairings for leg in p["legs"]}
        pairings = list(pairings) + [
            {"legs": [flight_id], "cost": artificial_cost, "_artificial": True}
            for flight_id in flight_ids if flight_id not in present
        ]

    flight_to_pairings: Dict[int, List[int]] = defaultdict(list)
    for j, p in enumerate(pairings):
        for leg in p["legs"]:
            flight_to_pairings[leg].append(j)

    covered_flights = set(flight_to_pairings.keys())
    M = len(pairings)

    prob = pulp.LpProblem("crew_pairing_lp", pulp.LpMinimize)

    x = [
        pulp.LpVariable(f"x_{j}", lowBound=0, upBound=1, cat="Continuous")
        for j in range(M)
    ]
    d = {i: pulp.LpVariable(f"d_{i}", lowBound=0, cat="Continuous") for i in covered_flights}

    prob += (
        pulp.lpSum(pairings[j]["cost"] * x[j] for j in range(M))
        + lambda_dh * pulp.lpSum(d.values())
    )

    for i in covered_flights:
        cover_sum = pulp.lpSum(x[j] for j in flight_to_pairings[i])
        prob += (cover_sum >= 1,          f"cover_{i}")
        prob += (d[i] >= cover_sum - 1,   f"dh_{i}")

    prob.solve(pulp.PULP_CBC_CMD(msg=int(verbose)))

    if prob.status != 1:
        return None

    mu_cov: Dict[int, float] = {}
    nu_exc: Dict[int, float] = {}
    for i in covered_flights:
        mu_cov[i] = prob.constraints[f"cover_{i}"].pi or 0.0
        # Dual of the dh_i constraint (d[i] >= cover_sum - 1) -- the shadow
        # price of how much this flight is currently over-covered
        # (deadheaded) by the pool; equals lambda_dh when binding. Paper
        # Eq. (9): delta_i = mu_i^cov - nu_i^exc. mu_cov pushes the policy
        # toward under-covered flights and nu_exc is subtracted to suppress
        # redundant coverage of already-saturated flights.
        nu_exc[i] = prob.constraints[f"dh_{i}"].pi or 0.0

    reduced_costs = [
        pairings[j]["cost"] - sum(mu_cov.get(i, 0.0) for i in pairings[j]["legs"])
        for j in range(real_pairing_count)
    ]

    return {
        "lp_value":      pulp.value(prob.objective),
        "dual_vars":     mu_cov,
        "dh_dual_vars":  nu_exc,
        "reduced_costs": reduced_costs,
        "status":        pulp.LpStatus[prob.status],
    }


def column_reduction(
    pairings: List[Dict],
    reduced_costs: List[float],
    threshold: float = 1e-6,
) -> List[Dict]:
    """
    Reduced-cost-based column reduction.

    Keep only pairings with rc_j <= threshold, while guaranteeing that at
    least one pairing covering each flight remains.

    Args:
        threshold: default 1e-6 ~= 0 (numerical-error tolerance)
    """
    kept_set = {j for j, rc in enumerate(reduced_costs) if rc <= threshold}

    flight_to_pairings: Dict[int, List[int]] = defaultdict(list)
    for j, p in enumerate(pairings):
        for leg in p["legs"]:
            flight_to_pairings[leg].append(j)

    for pairing_ids in flight_to_pairings.values():
        if not any(j in kept_set for j in pairing_ids):
            best_j = min(pairing_ids, key=lambda j: reduced_costs[j])
            kept_set.add(best_j)

    return [pairings[j] for j in sorted(kept_set)]


def solve_set_covering(
    pairings: List[Dict],
    n_flights: int,
    lambda_dh: float = 1.0,
    time_limit: int  = 300,
    use_gurobi: bool = False,
    verbose: bool    = False,
) -> Dict:
    """
    Solve the set-covering MIP (Eq. 2, x_j in {0,1}) to select the final
    pairing subset -- this is the restricted mixed-integer program described
    in Sec. "Scalable Inference and Global Selection" that performs the
    final selection over the generated candidate pool.

    Coverage is >=1 rather than ==1, so the same flight may be covered by
    multiple pairings -> deadheading is allowed. lambda_dh controls the DH
    penalty weight (0 disables DH suppression).

    Args:
        pairings:   list of pairing dicts (requires "legs", "cost" fields)
        n_flights:  total number of flights (flight IDs: 0 .. n_flights-1)
        lambda_dh:  DH penalty weight
        time_limit: solver time limit (seconds)
        use_gurobi: if True, use Gurobi, falling back to CBC on failure
        verbose:    whether to print solver logs

    Returns:
        selected, n_pairings, total_cost, coverage, status,
        uncoverable, deadhead_count, deadhead_flights
    """
    if not pairings:
        return {
            "selected": [], "n_pairings": 0, "total_cost": 0.0,
            "coverage": 0.0, "status": "no_pairings", "uncoverable": n_flights,
            "deadhead_count": 0, "deadhead_flights": [],
        }

    flight_to_pairings: Dict[int, List[int]] = defaultdict(list)
    for j, p in enumerate(pairings):
        for leg in p["legs"]:
            flight_to_pairings[leg].append(j)

    covered_flights = set(flight_to_pairings.keys())
    uncoverable     = set(range(n_flights)) - covered_flights

    M = len(pairings)
    prob = pulp.LpProblem("crew_pairing_sc", pulp.LpMinimize)

    x = [pulp.LpVariable(f"x_{j}", cat="Binary") for j in range(M)]
    d = {i: pulp.LpVariable(f"d_{i}", lowBound=0) for i in covered_flights}

    prob += (
        pulp.lpSum(pairings[j]["cost"] * x[j] for j in range(M))
        + lambda_dh * pulp.lpSum(d.values())
    )

    for i in covered_flights:
        cover_sum = pulp.lpSum(x[j] for j in flight_to_pairings[i])
        prob += (cover_sum >= 1,          f"cover_{i}")
        prob += (d[i] >= cover_sum - 1,   f"dh_{i}")

    if use_gurobi:
        try:
            solver = pulp.GUROBI(timeLimit=time_limit, msg=int(verbose))
        except Exception:
            print("[warn] Gurobi unavailable -> falling back to CBC")
            solver = pulp.PULP_CBC_CMD(timeLimit=time_limit, msg=int(verbose))
    else:
        solver = pulp.PULP_CBC_CMD(timeLimit=time_limit, msg=int(verbose))

    prob.solve(solver)

    selected = [pairings[j] for j in range(M) if (x[j].value() or 0) > 0.5]

    covered_legs = set()
    for p in selected:
        covered_legs.update(p["legs"])
    covered_count = len(covered_legs & set(range(n_flights)))

    deadhead_flights = [
        i for i in covered_flights if (d[i].value() or 0) > 0.5
    ]

    total_legs = sum(len(p["legs"]) for p in selected)
    avg_legs   = total_legs / len(selected) if selected else 0.0

    return {
        "selected":            selected,
        "n_pairings":          len(selected),
        "total_cost":          sum(p["cost"] for p in selected),
        "mip_obj":             pulp.value(prob.objective),
        "coverage":            covered_count / n_flights if n_flights > 0 else 0.0,
        "status":              pulp.LpStatus[prob.status],
        "uncoverable":         len(uncoverable),
        "deadhead_count":      len(deadhead_flights),
        "deadhead_flights":    deadhead_flights,
        "avg_legs_per_pairing": avg_legs,
    }
