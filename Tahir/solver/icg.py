"""
I2CG / I2CGp: Iterative Column Generation for CPP.
Full faithful implementation of Algorithm 1 in Tahir, Desaulniers, El Hallaoui (2021).

Algorithm structure (Figure 3 of paper):
-----------------------------------------
Outer loop (max_iter, controlled by nb_fail):
  1. Solve RP   : MIP on full column pool C_S → integer solution S
  2. CP loop    : multi-phase incompatibility-filtered LP
       l = 0
       while l <= L_MAX:
         eligible = {j in C_S : deg(j, S) <= l}
         (lp_obj, duals, x) = LP(eligible)
         z_CP = lp_obj - cost(S)
         if z_CP >= 0          : break  (no descent direction)
         if integral(x)        : pivot S <- x; break
         if l < L_MAX          : l += 1; continue
         else                  : zoom (small MIP on LP support); break
  3. Pricing    : SPPRC once per outer iteration using CP duals
  4. nb_fail    : update; switch SP type if needed; stop if nb_fail >= maxFail

Key differences from prior version:
  - CP uses incompatibility-degree filtering (multi-phase, l = 0 .. L_MAX)
  - Integral LP solution -> pivot (update S without a new full MIP)
  - Fractional LP at l_max -> zoom (small MIP on LP support)
  - SPPRC pricing is called ONCE per outer iteration, after the CP loop
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

from .constraints import pairing_cost
from .spprc import solve_pricing, enumerate_pairings
from .lp_solver import solve_lp, solve_lp_full, solve_lp_partition, solve_mip

# ── Hyper-parameters (Table 6 / Section 4 of Tahir 2021) ─────────────────────
EPSILON_MIN = 1e-4   # min relative improvement to reset nb_fail
MAX_FAIL    = 3      # paper default
L_MAX       = 3      # max CP phase (incompatibility level)


# ── Public API ────────────────────────────────────────────────────────────────

def run_i2cg(
    inst:             Dict,
    initial_columns:  Optional[List[List[int]]] = None,
    max_fail:         int  = MAX_FAIL,
    max_iter:         int  = 200,
    time_limit_mip:   int  = 300,
    time_limit_total: int  = 3600,
    max_labels:       int  = 300,
    max_pricing_cols: int  = 500,
    verbose:          bool = True,
) -> Dict:
    """I2CG: full subproblem (no DNN). Baseline from Table 6."""
    return _i2cgx(
        inst=inst,
        initial_columns=initial_columns,
        P=None, Psi=None, class_max=1,
        max_fail=max_fail, max_iter=max_iter,
        time_limit_mip=time_limit_mip,
        time_limit_total=time_limit_total,
        max_labels=max_labels,
        max_pricing_cols=max_pricing_cols,
        verbose=verbose,
        method_name="I2CG",
    )


def run_i2cgp(
    inst:             Dict,
    P:                Dict[int, Dict[int, float]],
    Psi:              Dict[int, Dict[int, int]],
    class_max:        int,
    initial_columns:  Optional[List[List[int]]] = None,
    max_fail:         int  = MAX_FAIL,
    max_iter:         int  = 200,
    time_limit_mip:   int  = 300,
    time_limit_total: int  = 3600,
    max_labels:       int  = 300,
    max_pricing_cols: int  = 500,
    verbose:          bool = True,
) -> Dict:
    """I2CGp: DNN-guided reduced subproblem (Algorithm 1, Tahir et al. 2021)."""
    return _i2cgx(
        inst=inst,
        initial_columns=initial_columns,
        P=P, Psi=Psi, class_max=class_max,
        max_fail=max_fail, max_iter=max_iter,
        time_limit_mip=time_limit_mip,
        time_limit_total=time_limit_total,
        max_labels=max_labels,
        max_pricing_cols=max_pricing_cols,
        verbose=verbose,
        method_name="I2CGp",
    )


# ── Core algorithm ────────────────────────────────────────────────────────────

def _i2cgx(
    inst:             Dict,
    initial_columns:  Optional[List[List[int]]],
    P:                Optional[Dict],
    Psi:              Optional[Dict],
    class_max:        int,
    max_fail:         int,
    max_iter:         int,
    time_limit_mip:   int,
    time_limit_total: int,
    max_labels:       int,
    max_pricing_cols: int,
    verbose:          bool,
    method_name:      str,
) -> Dict:
    legs        = inst["legs"]
    bases       = inst["bases"]
    leg_map     = {leg["flight_id"]: leg for leg in legs}
    flights     = [leg["flight_id"] for leg in legs]
    availability = inst.get("availability", {})
    t0          = time.time()

    # ── Pre-build availability matrix (reused every LP/MIP call) ─────────────
    # avail_info = (builder function) to keep icg.py clean
    avail_builder = _make_avail_builder(availability, leg_map)

    # ── Initialise column pool ────────────────────────────────────────────────
    columns, costs = _init_pool(
        legs, bases, leg_map, initial_columns,
        P, Psi, class_max, max_labels, max_pricing_cols, verbose, method_name,
    )
    if not columns:
        return _empty_result(method_name)

    nb_fail    = 0
    sp_type    = "reduced" if P is not None else "full"
    outer_iter = 0
    S_cost     = float("inf")
    S_selected: List[int] = []
    mip_status = "Init"

    for outer_iter in range(1, max_iter + 1):
        if time.time() - t0 > time_limit_total:
            if verbose:
                print(f"  [{method_name}] time limit reached.", flush=True)
            break

        S_old_cost = S_cost

        # ── 1. Solve RP (MIP on full column pool) ─────────────────────────────
        avail_A, avail_b = avail_builder(columns)
        sel, new_cost, mip_status, mip_dt = solve_mip(
            flights, columns, costs, time_limit=time_limit_mip,
            A_ub=avail_A, b_ub=avail_b,
        )
        if sel:
            S_selected = sel
            S_cost     = new_cost

        if verbose:
            print(f"  [{method_name}] iter {outer_iter:3d} | "
                  f"RP={S_cost:.2f} ({mip_dt:.1f}s) | "
                  f"pool={len(columns)} | nb_fail={nb_fail} | sp={sp_type}",
                  flush=True)

        # ── 2. CP loop: multi-phase incompatibility filtering ─────────────────
        # Incompatibility degrees are fixed for the entire CP loop
        # (S only changes via pivot/zoom at which point we break immediately)
        degrees  = _compute_incompatibility_degrees(columns, S_selected)
        last_duals: Dict[int, float] = {f: 0.0 for f in flights}
        # Flights currently covered by S — LP domain for CP.
        # S pairings are always in eligible (deg=0), so S is always a
        # feasible LP point: no artificial variables needed → LP can be
        # genuinely fractional, giving meaningful z_cp < 0 values.
        # Only operated (positive) flight IDs count for coverage/domain.
        S_flights: set = set(f for i in S_selected for f in columns[i] if f >= 0)
        S_cost_cov = float(sum(costs[i] for i in S_selected))
        l = 0

        while l <= L_MAX:
            eligible = [j for j, d in enumerate(degrees) if d <= l]
            if not eligible:
                l += 1
                continue

            lp_real, duals, x_vals, is_integral = _solve_cp(
                flights, columns, costs, eligible, S_flights,
                avail_builder=avail_builder,
            )
            last_duals = duals
            z_cp = lp_real - S_cost_cov

            if verbose:
                gap_pct = abs(z_cp / max(abs(S_cost), 1.0)) * 100
                print(f"    CP l={l:d} | elig={len(eligible):4d} "
                      f"LP={lp_real:.2f} z_cp={z_cp:+.4f} "
                      f"({'integral' if is_integral else 'frac':8s}) "
                      f"gap={gap_pct:.4f}%",
                      flush=True)

            if z_cp >= -1e-6:
                # No improvement at this phase.
                # Try the next phase (larger eligible set) unless already at L_MAX.
                # Rationale: eligible(l+1) ⊇ eligible(l), so LP(l+1) ≤ LP(l).
                # Higher-deg pricing columns may offer a cheaper direction at l+1.
                if l < L_MAX:
                    l += 1
                    continue
                else:
                    break  # exhausted all phases, no improvement found

            if is_integral:
                # Integer descent direction found → pivot
                new_S = [eligible[k] for k, xv in enumerate(x_vals) if xv > 0.5]
                if new_S:
                    S_selected = new_S
                    S_cost     = lp_real
                    S_cost_cov = lp_real
                    S_flights  = set(f for i in new_S for f in columns[i] if f >= 0)
                    if verbose:
                        print(f"    CP: integral pivot -> S_cost={S_cost:.2f}",
                              flush=True)
                break  # exit CP loop; next outer iteration will Solve RP

            else:
                # Fractional LP direction
                if l < L_MAX:
                    l += 1
                else:
                    # l == L_MAX and still fractional -> Zoom
                    zoom_sel, zoom_cost, zoom_status = _zoom(
                        list(S_flights), columns, costs, eligible, x_vals,
                        time_limit=min(time_limit_mip, 60),
                    )
                    if zoom_sel and zoom_cost < S_cost - 1e-6:
                        S_selected = zoom_sel
                        S_cost     = zoom_cost
                    if verbose:
                        print(f"    CP: zoom -> cost={zoom_cost:.2f} "
                              f"status={zoom_status}", flush=True)
                    break

        # ── 3. Pricing: SPPRC once per outer iteration ────────────────────────
        use_P   = P   if sp_type == "reduced" else None
        use_Psi = Psi if sp_type == "reduced" else None

        new_cols = solve_pricing(
            legs, bases, last_duals,
            P=use_P, Psi=use_Psi, class_max=class_max,
            threshold=-1e-6,
            max_labels=max_labels,
            max_cols=max_pricing_cols,
        )
        added = _add_unique(columns, costs, new_cols, leg_map)

        if verbose:
            print(f"    pricing ({sp_type}) -> {len(new_cols)} cols "
                  f"(added {added})", flush=True)

        # ── 4. Improvement check ──────────────────────────────────────────────
        if S_old_cost < float("inf"):
            improvement = (S_old_cost - S_cost) / max(abs(S_old_cost), 1.0)
        else:
            improvement = float("inf")

        if improvement >= EPSILON_MIN:
            nb_fail = 0
            sp_type = "reduced" if P is not None else "full"
        else:
            nb_fail += 1
            if sp_type == "reduced" and P is not None:
                sp_type = "full"
                if verbose:
                    print(f"    nb_fail={nb_fail}: switching to full SP",
                          flush=True)

        if nb_fail >= max_fail:
            if verbose:
                print(f"  [{method_name}] nb_fail={nb_fail} >= maxFail={max_fail}"
                      f" -> stop", flush=True)
            break

        if added == 0 and nb_fail > 0:
            if verbose:
                print(f"  [{method_name}] no new columns (nb_fail={nb_fail})"
                      f" -> stop", flush=True)
            break

    # ── Final LP for gap calculation (full column pool) ───────────────────────
    total_time = time.time() - t0
    final_lp_real, _, _, _ = solve_lp(flights, columns, costs)

    covered = set()
    for idx in S_selected:
        covered.update(f for f in columns[idx] if f >= 0)
    covered_flights = covered.intersection(set(flights))
    coverage    = len(covered_flights) / len(flights) if flights else 0.0
    n_uncovered = len(flights) - len(covered_flights)

    if final_lp_real < float("inf"):
        gap_pct = abs(
            (S_cost - final_lp_real) / max(abs(final_lp_real), 1.0) * 100
        )
    else:
        gap_pct = float("inf")

    if verbose:
        gap_str = f"{gap_pct:.4f}%" if gap_pct < float("inf") else "N/A"
        print(f"  [{method_name}] FINAL | MIP={S_cost:.2f} LP={final_lp_real:.2f} "
              f"gap={gap_str} | coverage={coverage:.3f} ({n_uncovered} uncovered) | "
              f"pairings={len(S_selected)} | iters={outer_iter} | "
              f"total={total_time:.1f}s", flush=True)

    return {
        "method":            method_name,
        "selected_pairings": [columns[i] for i in S_selected],
        "lp_obj":            final_lp_real,
        "mip_obj":           S_cost,
        "gap_pct":           gap_pct,
        "n_iters":           outer_iter,
        "n_columns":         len(columns),
        "coverage":          coverage,
        "n_uncovered":       n_uncovered,
        "status":            mip_status,
        "cg_time":           total_time,
        "mip_time":          0.0,
        "total_time":        total_time,
    }


# ── CP helpers ────────────────────────────────────────────────────────────────

def _make_avail_builder(availability: Dict, leg_map: Dict):
    """
    Return a callable avail_builder(columns) -> (A_ub, b_ub) or (None, None).

    The availability constraint: for each (base, day, cap) triple,
    the sum of x_p for pairings p that start from `base` on `dep_day`=`day`
    must be <= cap.

    Returns a function so the (base, day) structure is computed once, and
    only the column mapping is rebuilt when columns grow.
    """
    import numpy as np

    if not availability or not leg_map:
        return lambda cols: (None, None)

    # Enumerate (base, day, cap) triples
    constraints = [
        (base, int(day), float(cap))
        for base, day_counts in availability.items()
        for day, cap in day_counts.items()
    ]
    if not constraints:
        return lambda cols: (None, None)

    def builder(columns):
        n_cols = len(columns)
        rows, col_idx, vals = [], [], []
        b_list = []
        for row_i, (base, day, cap) in enumerate(constraints):
            any_entry = False
            for k, col in enumerate(columns):
                if not col:
                    continue
                # col[0] may be negative (deadhead); use abs() to get real fid
                fl = leg_map.get(abs(col[0]))
                if fl is None:
                    continue
                if fl["origin"] == base and fl["dep_day"] == day:
                    rows.append(row_i)
                    col_idx.append(k)
                    vals.append(1.0)
                    any_entry = True
            b_list.append(cap)

        if not any(v for v in vals):
            return None, None

        from scipy.sparse import csr_matrix
        A_ub = csr_matrix(
            (vals, (rows, col_idx)),
            shape=(len(constraints), n_cols),
        ).toarray()
        return A_ub, np.array(b_list)

    return builder


def _compute_incompatibility_degrees(
    columns:   List[List[int]],
    S_indices: List[int],
) -> List[int]:
    """
    Compute incompatibility degree of each column w.r.t. current solution S.

    deg(j, S) = |{s in S : flights(j) ∩ flights(s) != {} AND flights(s) ⊄ flights(j)}|

    - 0  : j is fully compatible with S (disjoint or a superset of every s it touches)
    - k>0: j partially disrupts k pairings in S

    Columns belonging to S always have deg=0 because S is a valid set partition
    (no two pairings share flights, so no partial overlap is possible).
    """
    if not S_indices:
        return [0] * len(columns)

    # Use only operated (positive) flight IDs for incompatibility computation.
    S_sets = [frozenset(f for f in columns[i] if f >= 0) for i in S_indices]
    degrees: List[int] = []
    for col in columns:
        j_set = frozenset(f for f in col if f >= 0)
        deg = 0
        for s_set in S_sets:
            shared = j_set & s_set
            if shared and not s_set.issubset(j_set):
                deg += 1
        degrees.append(deg)
    return degrees


def _solve_cp(
    flights:          List[int],
    columns:          List[List[int]],
    costs:            List[float],
    eligible_indices: List[int],
    S_flights:        set,
    avail_builder     = None,
) -> Tuple[float, Dict[int, float], List[float], bool]:
    """
    Solve CP(l): pure LP relaxation on S-covered flights.

    LP domain = flights currently covered by S.
    S pairings are always in eligible (deg=0) → LP always feasible (no big-M).
    Availability constraints (if provided via avail_builder) are added as
    inequalities, making the LP genuinely fractional when binding.

    Returns:
        lp_real    : LP objective (lower bound on S-covered flights)
        duals      : {flight_id: dual} (0 for flights outside LP domain)
        x_vals     : LP solution values for each eligible column
        is_integral: True iff all x_vals in {0, 1} (within tol=1e-4)
    """
    if not eligible_indices or not S_flights:
        return float("inf"), {f: 0.0 for f in flights}, [], False

    elig_cols  = [columns[i] for i in eligible_indices]
    elig_costs = [costs[i]   for i in eligible_indices]

    # LP only over flights in S (guarantees feasibility without artificials)
    lp_flights = [f for f in flights if f in S_flights]
    if not lp_flights:
        return float("inf"), {f: 0.0 for f in flights}, [], False

    # Availability constraints on eligible columns
    A_ub = b_ub_vec = None
    if avail_builder is not None:
        A_ub, b_ub_vec = avail_builder(elig_cols)

    lp_real, cp_duals, x_vals, _ = solve_lp_partition(
        lp_flights, elig_cols, elig_costs,
        A_ub=A_ub, b_ub=b_ub_vec,
    )

    # Extend duals to all flights (flights outside LP domain → dual=0)
    duals = {f: cp_duals.get(f, 0.0) for f in flights}

    tol = 1e-4
    is_integral = all(xv < tol or xv > 1.0 - tol for xv in x_vals)

    return lp_real, duals, x_vals, is_integral


def _zoom(
    flights:          List[int],
    columns:          List[List[int]],
    costs:            List[float],
    eligible_indices: List[int],
    x_vals:           List[float],
    time_limit:       int = 60,
) -> Tuple[List[int], float, str]:
    """
    Zoom (Zaghrouti et al. 2018): solve small MIP on LP solution support.

    When the LP is fractional at l=L_MAX, restrict to columns with x_j > 0
    and find the best integer solution in that neighbourhood.

    Returns:
        sel_global : selected column indices (global, into `columns`)
        obj        : MIP objective
        status     : solver status string
    """
    tol = 1e-4
    support = [
        eligible_indices[k]
        for k, xv in enumerate(x_vals)
        if xv > tol
    ]
    if not support:
        return [], float("inf"), "NoSupport"

    supp_cols  = [columns[i] for i in support]
    supp_costs = [costs[i]   for i in support]

    sel_in_supp, obj, status, _ = solve_mip(
        flights, supp_cols, supp_costs, time_limit=time_limit
    )
    if not sel_in_supp:
        return [], float("inf"), status

    sel_global = [support[k] for k in sel_in_supp]
    return sel_global, obj, status


# ── Pool helpers ──────────────────────────────────────────────────────────────

def _init_pool(
    legs, bases, leg_map, initial_columns,
    P, Psi, class_max, max_labels, max_pricing_cols, verbose, method_name,
) -> Tuple[List[List[int]], List[float]]:
    """Build initial column pool from provided columns or greedy reference."""
    if initial_columns:
        columns = [list(c) for c in initial_columns]
        costs   = [
            pairing_cost(
                [leg_map[abs(f)] for f in col],
                dh_set=frozenset(abs(f) for f in col if f < 0),
            )
            for col in columns
        ]
        if verbose:
            print(f"  [{method_name}] using {len(columns)} provided initial columns",
                  flush=True)
        return columns, costs

    from dnn.reference import generate_reference_pairings
    inst_stub = {
        "legs": legs, "bases": bases,
        "airports": list(set(l["origin"] for l in legs) |
                        set(l["dest"]   for l in legs)),
        "aircraft_type": legs[0]["aircraft_type"] if legs else "?",
        "instance_id": 0, "source": "CPP", "availability": {},
    }
    greedy_ref = generate_reference_pairings(inst_stub, method="greedy",
                                             verbose=False)
    columns = [list(c) for c in greedy_ref]
    costs   = [
        pairing_cost(
            [leg_map[abs(f)] for f in col],
            dh_set=frozenset(abs(f) for f in col if f < 0),
        )
        for col in columns
    ]

    if verbose:
        covered = len(set(f for c in columns for f in c if f >= 0))
        print(f"  [{method_name}] initial pool: {len(columns)} greedy columns "
              f"(coverage {covered}/{len(legs)})", flush=True)
    return columns, costs


def _add_unique(
    columns:  List[List[int]],
    costs:    List[float],
    new_cols: List[List[int]],
    leg_map:  Dict,
) -> int:
    """Add non-duplicate columns; return number added."""
    existing = {tuple(c) for c in columns}
    added = 0
    for col in new_cols:
        key = tuple(col)
        if key not in existing:
            existing.add(key)
            columns.append(col)
            costs.append(pairing_cost(
                [leg_map[abs(f)] for f in col],
                dh_set=frozenset(abs(f) for f in col if f < 0),
            ))
            added += 1
    return added


def _empty_result(method_name: str) -> Dict:
    return {
        "method":            method_name,
        "selected_pairings": [],
        "lp_obj":            float("inf"),
        "mip_obj":           float("inf"),
        "gap_pct":           float("inf"),
        "n_iters":           0,
        "n_columns":         0,
        "coverage":          0.0,
        "n_uncovered":       0,
        "status":            "NoColumns",
        "cg_time":           0.0,
        "mip_time":          0.0,
        "total_time":        0.0,
    }


# ── Reference solution via I2CG (for DNN training labels) ────────────────────

def generate_reference_via_cg(
    inst:       Dict,
    max_fail:   int  = 3,
    max_iter:   int  = 50,
    max_labels: int  = 200,
    verbose:    bool = False,
) -> List[List[int]]:
    """
    Near-optimal reference pairings via I2CG (no DNN).
    Used in dnn/reference.py for high-quality DNN training labels.
    """
    res      = run_i2cg(
        inst, max_fail=max_fail, max_iter=max_iter,
        time_limit_mip=120, max_labels=max_labels,
        max_pricing_cols=300, verbose=verbose,
    )
    pairings = res["selected_pairings"]

    # Only operated (positive) flight IDs count for coverage
    covered = {f for p in pairings for f in p if f >= 0}
    bases   = set(inst["bases"])
    for leg in inst["legs"]:
        fid = leg["flight_id"]
        if fid not in covered and leg["origin"] in bases and leg["dest"] in bases:
            pairings.append([fid])

    return pairings
