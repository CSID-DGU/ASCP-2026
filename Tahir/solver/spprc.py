"""
SPPRC: Shortest Path with Resource Constraints
Forward label-setting algorithm for CPP pricing subproblem.

For each base b, finds feasible pairings (starting and ending at b) with
negative reduced cost given LP dual values from the master problem.

Resources tracked per label:
  rc       float  reduced cost accumulated
  n_d      int    number of duties used          [1, D_BAR_DUTY=4]
  d_start  int    abs-minutes when current duty started
  work     int    work minutes in current duty   [0, T_BAR_W=480]
  n_df     int    flights in current duty        [0, F_MAX=5]
  cpbs     float  CPBS = product of Psi ranks    [1, inf)

Dominance: A dominates B at the same arrival node iff
  A.rc      <= B.rc
  A.n_d     <= B.n_d
  A.work    <= B.work
  A.d_start >= B.d_start   (later start => less duty-time consumed)
  A.n_df    <= B.n_df
  A.cpbs    <= B.cpbs
"""
from __future__ import annotations

import bisect
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from .constraints import (
    T_C_MIN, T_C_MAX, T_R_MIN, T_R_MAX, D_BAR, D_BAR_DUTY,
    T_BAR_D, T_BAR_W, F_MAX, GAMMA_DH, LAMBDA_DH,
    T_HAT_C, T_HAT_R, LAMBDA_C, LAMBDA_R, T_MIN_DUTY_PAY,
)


def _conn_penalty(gap: int) -> float:
    return LAMBDA_C * (T_HAT_C - gap) if gap < T_HAT_C else 0.0


def _rest_penalty(gap: int) -> float:
    return LAMBDA_R * (T_HAT_R - gap) if gap < T_HAT_R else 0.0


def _duty_min240_correction(work: float) -> float:
    """rc correction when a duty ends: T_p guarantees max(240, work) per duty."""
    return max(0.0, float(T_MIN_DUTY_PAY) - work)


# ── Label ─────────────────────────────────────────────────────────────────────

class Label:
    """Forward label at the arrival event of the last flight in path."""

    __slots__ = ('rc', 'n_d', 'd_start', 'work', 'n_df', 'cpbs',
                 'path', 'first_dep', 'last_arr')

    def __init__(self, rc, n_d, d_start, work, n_df, cpbs,
                 path, first_dep, last_arr):
        self.rc        = rc
        self.n_d       = n_d
        self.d_start   = d_start
        self.work      = work
        self.n_df      = n_df
        self.cpbs      = cpbs
        self.path      = path       # tuple of flight_id
        self.first_dep = first_dep  # dep_abs of first flight (D_BAR check)
        self.last_arr  = last_arr   # arr_abs of last flight

    def dominates(self, other: "Label") -> bool:
        """True iff self is at least as good as other in every resource."""
        return (
            self.rc      <= other.rc      and
            self.n_d     <= other.n_d     and
            self.work    <= other.work    and
            self.d_start >= other.d_start and  # higher => less duty-time used
            self.n_df    <= other.n_df    and
            self.cpbs    <= other.cpbs
        )


def _prune(labels: List[Label], max_labels: int = 300) -> List[Label]:
    """Remove dominated labels; keep at most max_labels sorted by rc."""
    if len(labels) <= 1:
        return labels
    labels.sort(key=lambda l: l.rc)
    survivors: List[Label] = []
    for lab in labels:
        if not any(s.dominates(lab) for s in survivors):
            survivors.append(lab)
            if len(survivors) >= max_labels:
                break
    return survivors


# ── Airport-indexed departure lookup ──────────────────────────────────────────

def _build_airport_index(
    legs: List[Dict],
) -> Dict[str, Tuple[List[int], List[int]]]:
    """
    Returns {airport: (sorted dep_abs list, fid list)} for O(log N) range queries.
    """
    by_ap: Dict[str, Tuple[List[int], List[int]]] = {}
    for leg in sorted(legs, key=lambda l: l["dep_abs"]):
        ap = leg["origin"]
        if ap not in by_ap:
            by_ap[ap] = ([], [])
        by_ap[ap][0].append(leg["dep_abs"])
        by_ap[ap][1].append(leg["flight_id"])
    return by_ap


# ── Core SPPRC ────────────────────────────────────────────────────────────────

def solve_pricing(
    legs:       List[Dict],
    bases:      List[str],
    duals:      Dict[int, float],
    P:          Optional[Dict[int, Dict[int, float]]] = None,
    Psi:        Optional[Dict[int, Dict[int, int]]]   = None,
    class_max:  int   = 1,
    threshold:  float = -1e-6,
    max_labels: int   = 300,
    max_cols:   int   = 1000,
) -> List[List[int]]:
    """
    Solve the SPPRC pricing subproblem for all bases.

    Args:
        legs:       list of flight dicts (flight_id, origin, dest, dep_abs,
                    arr_abs, duration)
        bases:      list of base airport names
        duals:      {flight_id: dual_value} from LP relaxation
        P:          DNN probability matrix. If given, only arcs with P[i][j]>0
                    are allowed (I2CGp reduced subproblem).
        Psi:        DNN rank matrix. Required when P is given.
        class_max:  maximum rank used in CPBS computation
        threshold:  only return columns with rc < threshold
        max_labels: max non-dominated labels kept per node
        max_cols:   total column cap across all bases

    Returns:
        List of path lists ([flight_id, ...]) with rc < threshold.
    """
    leg_map      = {leg["flight_id"]: leg for leg in legs}
    legs_sorted  = sorted(legs, key=lambda l: (l["dep_abs"], l["flight_id"]))
    ap_idx       = _build_airport_index(legs_sorted)

    all_cols: List[List[int]] = []
    for base in bases:
        cols = _spprc_base(
            legs_sorted, leg_map, ap_idx, base, duals,
            P, Psi, class_max, threshold, max_labels,
            max_cols - len(all_cols),
        )
        all_cols.extend(cols)
        if len(all_cols) >= max_cols:
            break
    return all_cols


def _spprc_base(
    legs_sorted: List[Dict],
    leg_map:     Dict[int, Dict],
    ap_idx:      Dict[str, Tuple[List[int], List[int]]],
    base:        str,
    duals:       Dict[int, float],
    P:           Optional[Dict],
    Psi:         Optional[Dict],
    class_max:   int,
    threshold:   float,
    max_labels:  int,
    max_cols:    int,
) -> List[List[int]]:
    """SPPRC forward label-setting for a single base."""

    labels_at: Dict[int, List[Label]] = defaultdict(list)

    # Initialise: one label per flight departing from base
    # Note: initial flights at the base are always operated (positive fid).
    for leg in legs_sorted:
        if leg["origin"] != base:
            continue
        fid = leg["flight_id"]
        labels_at[fid].append(Label(
            rc        = leg["duration"] - duals.get(fid, 0.0),
            n_d       = 1,
            d_start   = leg["dep_abs"],
            work      = leg["duration"],
            n_df      = 1,
            cpbs      = 1.0,
            path      = (fid,),
            first_dep = leg["dep_abs"],
            last_arr  = leg["arr_abs"],
        ))

    # Process nodes in topological order (arr_abs ascending).
    # Because any feasible successor j has:
    #   j.dep_abs >= i.arr_abs + T_C_MIN  =>  j.arr_abs > i.arr_abs
    # labels are never added to already-processed nodes.
    proc_order = sorted(legs_sorted, key=lambda l: (l["arr_abs"], l["flight_id"]))

    results:    List[List[int]] = []
    seen_paths: set = set()

    for leg in proc_order:
        fid = leg["flight_id"]
        if fid not in labels_at:
            continue

        labels = _prune(labels_at.pop(fid), max_labels)

        arr_i  = leg["arr_abs"]
        dest_i = leg["dest"]

        for lab in labels:

            # ── Record complete pairing (at base, >=2 operated flights) ─────
            # Apply final duty's min-240 correction to rc (Eq. 2: max(240, work)).
            n_operated = sum(1 for f in lab.path if f >= 0)
            final_rc   = lab.rc + _duty_min240_correction(lab.work)
            if dest_i == base and n_operated >= 2 and final_rc < threshold:
                pkey = lab.path
                if pkey not in seen_paths:
                    seen_paths.add(pkey)
                    results.append(list(lab.path))
                    if len(results) >= max_cols:
                        return results

            # ── Expand to successors ─────────────────────────────────────────
            ap_data = ap_idx.get(dest_i)
            if ap_data is None:
                continue
            dep_times, fids = ap_data

            # Set of absolute flight IDs already visited (to prevent revisits)
            visited_abs = {abs(f) for f in lab.path}

            # Connection arcs: same duty
            lo = bisect.bisect_left(dep_times,  arr_i + T_C_MIN)
            hi = bisect.bisect_right(dep_times, arr_i + T_C_MAX)
            for k in range(lo, hi):
                jid  = fids[k]
                if jid in visited_abs:
                    continue
                legj = leg_map[jid]

                if P is not None and jid not in P.get(fid, {}):
                    continue

                gap      = legj["dep_abs"] - arr_i
                new_dlen = legj["arr_abs"] - lab.d_start
                if (legj["arr_abs"] - lab.first_dep) / 1440.0 > D_BAR:
                    continue

                cpbs_j = (lab.cpbs * Psi.get(fid, {}).get(jid, class_max + 1)
                          if Psi is not None else 1.0)

                # ── Operated extension ──────────────────────────────────────
                new_work = lab.work + legj["duration"]
                new_ndf  = lab.n_df + 1
                if new_work <= T_BAR_W and new_dlen <= T_BAR_D and new_ndf <= F_MAX:
                    labels_at[jid].append(Label(
                        rc        = lab.rc + _conn_penalty(gap)
                                    + legj["duration"] - duals.get(jid, 0.0),
                        n_d       = lab.n_d,
                        d_start   = lab.d_start,
                        work      = new_work,
                        n_df      = new_ndf,
                        cpbs      = cpbs_j,
                        path      = lab.path + (jid,),
                        first_dep = lab.first_dep,
                        last_arr  = legj["arr_abs"],
                    ))

                # ── Deadhead extension ──────────────────────────────────────
                # DH work time = duration // 2; n_df unchanged; no dual credit.
                dh_work = legj["duration"] // 2
                new_work_dh = lab.work + dh_work
                if new_work_dh <= T_BAR_W and new_dlen <= T_BAR_D:
                    labels_at[jid].append(Label(
                        rc        = lab.rc + _conn_penalty(gap)
                                    + GAMMA_DH + LAMBDA_DH * legj["duration"],
                        n_d       = lab.n_d,
                        d_start   = lab.d_start,
                        work      = new_work_dh,
                        n_df      = lab.n_df,  # DH does not count toward F_MAX
                        cpbs      = cpbs_j,
                        path      = lab.path + (-jid,),  # negative = deadhead
                        first_dep = lab.first_dep,
                        last_arr  = legj["arr_abs"],
                    ))

            # Rest arcs: new duty (only if duty budget allows)
            if lab.n_d >= D_BAR_DUTY:
                continue

            lo = bisect.bisect_left(dep_times,  arr_i + T_R_MIN)
            hi = bisect.bisect_right(dep_times, arr_i + T_R_MAX)
            for k in range(lo, hi):
                jid  = fids[k]
                if jid in visited_abs:
                    continue
                legj = leg_map[jid]

                if P is not None and jid not in P.get(fid, {}):
                    continue

                gap = legj["dep_abs"] - arr_i
                if (legj["arr_abs"] - lab.first_dep) / 1440.0 > D_BAR:
                    continue

                cpbs_j = (lab.cpbs * Psi.get(fid, {}).get(jid, class_max + 1)
                          if Psi is not None else 1.0)

                # min-240 correction for the duty that just ended (Eq. 2)
                corr = _duty_min240_correction(lab.work)

                # ── Operated extension (new duty) ───────────────────────────
                labels_at[jid].append(Label(
                    rc        = lab.rc + corr + _rest_penalty(gap)
                                + legj["duration"] - duals.get(jid, 0.0),
                    n_d       = lab.n_d + 1,
                    d_start   = legj["dep_abs"],
                    work      = legj["duration"],
                    n_df      = 1,
                    cpbs      = cpbs_j,
                    path      = lab.path + (jid,),
                    first_dep = lab.first_dep,
                    last_arr  = legj["arr_abs"],
                ))

                # ── Deadhead extension (new duty) ───────────────────────────
                dh_work = legj["duration"] // 2
                labels_at[jid].append(Label(
                    rc        = lab.rc + corr + _rest_penalty(gap)
                                + GAMMA_DH + LAMBDA_DH * legj["duration"],
                    n_d       = lab.n_d + 1,
                    d_start   = legj["dep_abs"],
                    work      = dh_work,
                    n_df      = 0,  # DH does not count toward F_MAX
                    cpbs      = cpbs_j,
                    path      = lab.path + (-jid,),  # negative = deadhead
                    first_dep = lab.first_dep,
                    last_arr  = legj["arr_abs"],
                ))

    return results


# ── Enumerate all feasible pairings (reference solution helper) ───────────────

def enumerate_pairings(
    legs:       List[Dict],
    bases:      List[str],
    P:          Optional[Dict] = None,
    Psi:        Optional[Dict] = None,
    class_max:  int            = 1,
    max_labels: int            = 300,
    max_cols:   int            = 10000,
) -> Tuple[List[List[int]], List[float]]:
    """
    Enumerate all feasible pairings without pricing (threshold=+inf).
    Returns (columns, costs). Used for reference solution generation.
    """
    from .constraints import pairing_cost
    leg_map = {leg["flight_id"]: leg for leg in legs}
    cols = solve_pricing(
        legs, bases, duals={}, P=P, Psi=Psi, class_max=class_max,
        threshold=float("inf"), max_labels=max_labels, max_cols=max_cols,
    )
    costs = [
        pairing_cost(
            [leg_map[abs(f)] for f in path],
            dh_set=frozenset(abs(f) for f in path if f < 0),
        )
        for path in cols
    ]
    return cols, costs
