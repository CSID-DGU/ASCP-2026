# base_reach.py -- backward reachability DP that determines whether a pairing
# can still return to base
#
# Paper Sec. "Constraint-Conditioned Pairing Generator": "This mechanism
# additionally removes any leg from which the base can no longer be reached
# within the remaining pairing-duration and overnight budget, computed via
# backward reachability from the base." This module computes that backward
# reachability. For each flight's arrival point, we precompute an admissible
# lower bound on (a) the additional elapsed time and (b) the additional
# number of duty-boundary crossings needed to return to base_ap; any leg
# whose lower bound already exceeds max_pairing_days or max_duty_periods is
# masked out in RL/environment.py::get_mask. Because both bounds are
# admissible (never overestimate feasibility), this pruning never removes a
# path that is actually feasible -- the same logic as resource-infeasible
# label pruning in SPPRC.
#
# The duty-boundary-crossing bound (duty_crossings) is checked against
# max_duty_periods rather than counting raw hops against max_legs, because
# EndDuty always grants a fresh per-duty leg budget: the real resource that
# can block base return is the number of remaining overnight rests
# (max_duty_periods), not legs-per-duty.

INF = float("inf")


def build_base_reach(flights, base_ap, constraint):
    """Compute two lower bounds on the cost of returning to base_ap from each flight's arrival point.

    Returns: {"time": {id: min additional elapsed time}, "duty_crossings": {id: min additional duty-boundary crossings}}

    The time bound alone rarely prunes anything -- max_pairing_days is
    usually large enough (e.g. 5 days) that most paths fit inside it. The
    duty_crossings bound is what actually keeps the policy near the base,
    since max_duty_periods is a much tighter budget (typically 2-4). Both
    bounds are lower bounds, hence admissible.

    To keep both bounds admissible, connectivity is modeled as loosely as possible:
      - Only gap >= min_conn is required; no upper bound (max_conn) is
        applied, since a large gap can still be legalized via EndDuty
        (overnight rest). A connection with gap > max_conn is instead
        counted as a duty-boundary crossing (it needs rest), since only
        min_conn..max_conn connections are valid within the same duty. If a
        gap exceeds max_conn but is still below min_rest, it is neither a
        legal same-duty connection nor a legal rest, so that connection is
        excluded from the bound entirely.
      - Duty-time and legs-per-duty limits (max_duty, max_legs) are ignored,
        since including them could only increase the bound (i.e. would break
        admissibility). Only duty_crossings is tracked precisely, since it
        corresponds to the actual binding resource (EndDuty count, bounded by
        max_duty_periods).
      - Already-assigned flights are still allowed in the path (ignore
        `assigned`) -- again optimistic, preserving the lower-bound property.

    Computation: a single DP pass in descending dep_time order. Any successor
    leg g satisfies dep_g >= arr_f + min_conn > dep_f, so g is always
    processed before f in descending order -> O(N * average out-degree).

    Args:
        flights:    list of flight dicts (keys: id, origin, dest, dep_time, arr_time)
        base_ap:    target base airport ID
        constraint: reads min_conn/max_conn/min_rest
    Returns:
        dict: {"time": {...}, "duty_crossings": {...}}
    """
    min_conn = constraint.get("min_conn", 0.65)
    max_conn = constraint.get("max_conn", 9.0)
    min_rest = constraint.get("min_rest", 10.0)

    by_origin = {}
    for f in flights:
        by_origin.setdefault(f["origin"], []).append(f)

    D, C = {}, {}
    for f in sorted(flights, key=lambda x: -x["dep_time"]):
        if f["dest"] == base_ap:
            D[f["id"]], C[f["id"]] = 0.0, 0
            continue

        best_t, best_c = INF, INF
        arr = f["arr_time"]
        for g in by_origin.get(f["dest"], ()):
            gap = g["dep_time"] - arr
            if gap < min_conn:
                continue
            # A gap that exceeds max_conn but is still below min_rest is
            # legal neither as a same-duty connection nor as rest, so this
            # connection must be excluded from the bound entirely -- otherwise
            # duty_crossings would optimistically treat an infeasible path as feasible.
            if gap > max_conn and gap < min_rest:
                continue
            d_next = D.get(g["id"])
            if d_next is None or d_next == INF:
                continue
            cand_t = gap + (g["arr_time"] - g["dep_time"]) + d_next
            if cand_t < best_t:
                best_t = cand_t
            c_next = C[g["id"]]
            cand_c = c_next + (1 if gap > max_conn else 0)
            if cand_c < best_c:
                best_c = cand_c
        D[f["id"]], C[f["id"]] = best_t, best_c

    return {"time": D, "duty_crossings": C}


def can_reach_base(reach, flight, pairing_start_time, max_pairing_days,
                   duty_period=None, max_duty_periods=None):
    """Check whether base return remains feasible (by the admissible lower bound) after selecting `flight`.

    If reach is None (not computed), always returns True.

    When duty_period/max_duty_periods are given, also checks the remaining
    rest-count budget:
        duty_period so far + (min additional duty-boundary crossings needed to reach base) <= max_duty_periods
    Legs-per-duty (max_legs) is not checked here -- EndDuty can always grant a
    fresh budget, so it is not the real bottleneck (RL/environment.py already
    enforces max_legs correctly within the current duty at every step).
    """
    if reach is None:
        return True
    fid = flight["id"]
    d = reach["time"].get(fid, INF)
    if d == INF:
        return False
    if (flight["arr_time"] + d - pairing_start_time) / 24.0 > max_pairing_days:
        return False
    if duty_period is not None and max_duty_periods is not None:
        c = reach["duty_crossings"].get(fid, INF)
        if duty_period + c > max_duty_periods:
            return False
    return True
