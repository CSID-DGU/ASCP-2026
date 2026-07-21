"""
CPP feasibility checker (Tahir Table 5 constraints).
Used by both the greedy solver and the set partitioning verifier.
"""

from __future__ import annotations
from typing import List, Dict

# [2026-07-09] ASCP-2026 Delta 실제 규정(RL/airline_constraints/delta.py)에 맞춰 override.
# 원래는 Tahir et al. 2021 Table 5 값(min_conn=30, min_rest=570, max_duty=720, max_legs=5)이었음 —
# 우리 RL+IP 모델(min_conn=39, min_rest=600, max_duty=780, max_legs=8)과 비교할 때 Tahir가 더
# 빡빡한 규정으로 도는 confound가 있어서, 우리 데이터/모델 기준에 맞춰 조정.
T_C_MIN    = 39     # min_conn 0.65h → 39min (DELTA_CONSTRAINTS["min_conn"])
T_R_MIN    = 600    # min_rest 10.0h → 600min (DELTA_CONSTRAINTS["min_rest"])
T_C_MAX    = T_R_MIN - 1   # 599: connection arc covers [t^C, t^R) per paper Sec 4.2.2
T_R_MAX    = 48 * 60
D_BAR      = 5
D_BAR_DUTY = 4
T_BAR_D    = 780    # max_duty 13.0h → 780min (DELTA_CONSTRAINTS["max_duty"])
T_BAR_W    = 480
F_MAX      = 8      # max_legs 8 (DELTA_CONSTRAINTS["max_legs"])
T_MIN_DUTY_PAY = 240       # minimum guaranteed paid time per duty (4 hours, Eq. 2)

# Deadhead penalty parameters (Table 5, Tahir et al. 2021)
GAMMA_DH   = 400          # fixed deadhead penalty (minutes equivalent)
LAMBDA_DH  = 5.0 / 6.0   # per-minute deadhead penalty

# Connection/rest penalty parameters (Table 5)
T_HAT_C  = 60
T_HAT_R  = 690
LAMBDA_C = 6
LAMBDA_R = 5.0 / 3.0


class DutyState:
    """Accumulates resource consumption within one duty."""
    def __init__(self, start_abs: int):
        self.start_abs  = start_abs
        self.work_time  = 0
        self.n_flights  = 0

    def can_add(self, leg: Dict, conn_gap: int) -> bool:
        new_work = self.work_time + leg["duration"]
        new_duty = leg["arr_abs"] - self.start_abs
        return (
            new_work      <= T_BAR_W
            and new_duty  <= T_BAR_D
            and self.n_flights < F_MAX
            and conn_gap  >= T_C_MIN
            and conn_gap  <= T_C_MAX
        )

    def add(self, leg: Dict):
        self.work_time += leg["duration"]
        self.n_flights += 1


def is_feasible_pairing(legs: List[Dict], bases: List[str]) -> bool:
    """
    Check if a sequence of legs constitutes a feasible pairing:
    - Starts and ends at the same base airport
    - Satisfies duty/rest constraints
    """
    if not legs:
        return False

    if legs[0]["origin"] not in bases:
        return False

    n_duties   = 1
    duty       = DutyState(legs[0]["dep_abs"])
    duty.work_time = legs[0]["duration"]
    duty.n_flights = 1
    pairing_start  = legs[0]["dep_abs"]

    for k in range(1, len(legs)):
        prev = legs[k - 1]
        curr = legs[k]
        gap  = curr["dep_abs"] - prev["arr_abs"]

        if gap < T_C_MIN:
            return False  # impossible overlap or too-short connection

        if gap <= T_C_MAX:   # T_C_MAX = T_R_MIN - 1, so no gap zone
            # same duty (connection arc)
            if not duty.can_add(curr, gap):
                return False
            duty.add(curr)
        else:
            # rest → new duty (gap >= T_R_MIN)
            n_duties += 1
            if n_duties > D_BAR_DUTY:
                return False
            duty = DutyState(curr["dep_abs"])
            duty.work_time = curr["duration"]
            duty.n_flights = 1

        pairing_days = (curr["arr_abs"] - pairing_start) / 1440
        if pairing_days > D_BAR:
            return False

    # Must end at a base
    return legs[-1]["dest"] in bases


def pairing_cost(legs: List[Dict], dh_set: "frozenset" = frozenset()) -> float:
    """
    Pairing cost per Tahir et al. (2021) Equations (1)-(4).

    c_p = T_p + sum(phi^DH) + sum(phi^C)

    T_p = max(delta_p / 4,  sum_d max(240, fly_d + 0.5 * dh_d))   [Eq. 2]
    phi^DH(delta) = gamma^DH + lambda^DH * delta                   [Eq. 3]
    phi^C(delta)  = lambda^C*(t_hat_C - delta) if t_C <= delta < t_hat_C
                    lambda^R*(t_hat_R - delta) if t_R <= delta < t_hat_R
                    0 otherwise                                      [Eq. 4]

    Args:
        legs:   time-ordered list of leg dicts (operated + deadhead)
        dh_set: frozenset of flight_ids that are deadheads
    """
    if not legs:
        return 0.0

    # ── Split into duties ────────────────────────────────────────────────────
    duties: List[List[Dict]] = []
    cur: List[Dict] = [legs[0]]
    for k in range(1, len(legs)):
        gap = legs[k]["dep_abs"] - legs[k - 1]["arr_abs"]
        if gap >= T_R_MIN:
            duties.append(cur)
            cur = [legs[k]]
        else:
            cur.append(legs[k])
    duties.append(cur)

    # ── T_p (Eq. 2) ─────────────────────────────────────────────────────────
    delta_p = legs[-1]["arr_abs"] - legs[0]["dep_abs"]
    sum_duty_paid = 0.0
    for duty_legs in duties:
        fly = sum(l["duration"] for l in duty_legs if l["flight_id"] not in dh_set)
        dh  = sum(l["duration"] for l in duty_legs if l["flight_id"] in dh_set)
        sum_duty_paid += max(float(T_MIN_DUTY_PAY), fly + 0.5 * dh)
    T_p = max(delta_p / 4.0, sum_duty_paid)

    # ── phi^DH: deadhead penalties ───────────────────────────────────────────
    dh_penalty = sum(
        GAMMA_DH + LAMBDA_DH * l["duration"]
        for l in legs if l["flight_id"] in dh_set
    )

    # ── phi^C: short-connection / short-rest penalties ───────────────────────
    phi_c = 0.0
    for k in range(1, len(legs)):
        gap = legs[k]["dep_abs"] - legs[k - 1]["arr_abs"]
        if T_C_MIN <= gap < T_R_MIN:        # connection (same duty)
            if gap < T_HAT_C:
                phi_c += LAMBDA_C * (T_HAT_C - gap)
        else:                               # rest (new duty)
            if gap < T_HAT_R:
                phi_c += LAMBDA_R * (T_HAT_R - gap)

    return T_p + dh_penalty + phi_c
