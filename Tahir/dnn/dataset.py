"""
Phase 1: Data pipeline
CPP instance → successor sets D_ib → X_i matrices → (X, y) training dataset

Implements Tahir paper's feature engineering (Table 2):
  9 features per flight, input row = [base_feats | flight_i_feats | successor_j_feats]
"""

from __future__ import annotations
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple

# ── Crew Pairing Constraints (Tahir Table 5) ────────────────────────────────
# [2026-07-09] solver/constraints.py와 동일하게 ASCP-2026 Delta 실제 규정에 맞춰 override
# (원래 Table 5 값: min_conn=30, min_rest=570, max_duty=720, max_legs=5).
# T_C_MAX: dnn/reference.py의 CG reference pairing 생성 로직이 "같은 duty로 묶는 조건"
# (gap <= T_C_MAX)과 "rest로 넘어가는 조건"(gap >= T_R_MIN)을 따로 쓰기 때문에, 이걸
# 고정값(240)으로 두면 T_R_MIN=600과의 사이(240~600분)에 "같은 duty도 rest도 아닌"
# 사각지대가 생겨 후보가 버려진다(2026-07-14 재구성 중 실측 확인 — Reference pairings
# 447개가 아니라 354개로 나와서 발견). solver/constraints.py와 동일하게 T_R_MIN-1로
# 자동 계산해 사각지대를 없앰.
T_C_MIN    = 39      # min connection time (minutes) — 0.65h
T_R_MIN    = 600     # min rest time (minutes) — 10.0 h
T_C_MAX    = T_R_MIN - 1   # 599 — solver/constraints.py와 동일한 자동 계산
T_R_MAX    = 48 * 60 # successor search window, 48 h
D_BAR      = 5       # max pairing days
D_BAR_DUTY = 4       # max duties per pairing
T_BAR_D    = 780     # max duty length (minutes) — 13.0 h
T_BAR_W    = 480     # max duty work time (minutes)  8 h
F_MAX      = 8       # max flights per duty


# ── Feature encoding ────────────────────────────────────────────────────────

def build_encoders(instances: List[Dict]) -> Dict[str, Dict[str, int]]:
    """
    Build integer encoders from a list of loaded instances.
    Categorical: airports, aircraft_types
    """
    airports = set()
    aircraft_types = set()
    for inst in instances:
        airports.update(inst["airports"])
        aircraft_types.add(inst["aircraft_type"])

    return {
        "airport": {a: i for i, a in enumerate(sorted(airports))},
        "aircraft": {t: i for i, t in enumerate(sorted(aircraft_types))},
    }


def _encode_leg(leg: Dict, base: str, enc: Dict) -> np.ndarray:
    """
    Encode one leg into 9-dim feature vector (Tahir Table 2):
      [dep_airport, arr_airport, base, aircraft_type,
       dep_date, dep_time, arr_date, arr_time, duration]
    """
    return np.array([
        enc["airport"].get(leg["origin"], 0),      # 0 departure_airport
        enc["airport"].get(leg["dest"],   0),      # 1 arrival_airport
        enc["airport"].get(base,          0),      # 2 base
        enc["aircraft"].get(leg["aircraft_type"], 0),  # 3 aircraft_type
        float(leg["dep_day"]),                     # 4 departure_date
        float(leg["dep_min"]),                     # 5 departure_time
        float(leg["arr_day"]),                     # 6 arrival_date
        float(leg["arr_min"]),                     # 7 arrival_time
        float(leg["duration"]),                    # 8 duration
    ], dtype=np.float32)


# ── Successor set construction ──────────────────────────────────────────────

def build_successor_sets(legs: List[Dict]) -> Dict[int, List[int]]:
    """
    D_ib: for each leg i, all feasible successors j satisfying:
      - j.origin == i.dest
      - T_C_MIN <= j.dep_abs - i.arr_abs <= T_R_MAX  (48 h window)
    Returns {flight_id: [successor_flight_ids sorted by dep_abs]}
    """
    by_origin: Dict[str, List[int]] = defaultdict(list)
    for leg in legs:
        by_origin[leg["origin"]].append(leg["flight_id"])

    successors: Dict[int, List[int]] = {}
    arr_abs = {leg["flight_id"]: leg["arr_abs"] for leg in legs}
    dep_abs = {leg["flight_id"]: leg["dep_abs"] for leg in legs}

    for leg in legs:
        fid   = leg["flight_id"]
        t_arr = leg["arr_abs"]
        cands = by_origin.get(leg["dest"], [])
        succ  = []
        for jid in cands:
            gap = dep_abs[jid] - t_arr
            if T_C_MIN <= gap <= T_R_MAX:
                succ.append(jid)
        succ.sort(key=lambda j: dep_abs[j])
        successors[fid] = succ

    return successors


MAX_SUCCESSORS = 8   # hard cap after filtering (paper achieves 4.81 with ICG ref)


def filter_successors_by_pattern(
    legs: List[Dict],
    successors: Dict[int, List[int]],
    ref_pairings: List[List[int]],
) -> Dict[int, List[int]]:
    """
    Tahir's training-time filter (Section 4 of paper):
    Build (prev_airport, conn_airport, ampm, dur_bucket, day_hour) → known_next_airports
    from reference pairings, then restrict D_ib accordingly.
    Target: ~4.81 average (Table 4). We cap at MAX_SUCCESSORS.

    Two-tier filter:
      1. Pattern match (origin, dest, ampm, dur_bucket, day_hour) → known next origins
      2. Prefer short-connection candidates (within T_C_MAX) over rest candidates
    """
    leg_map = {leg["flight_id"]: leg for leg in legs}
    dep_abs = {leg["flight_id"]: leg["dep_abs"] for leg in legs}

    def pattern(leg: Dict) -> tuple:
        ampm       = "AM" if leg["dep_min"] < 720 else "PM"
        dur_bucket = leg["duration"] // 60
        day_hour   = (leg["dep_day"], leg["dep_min"] // 60)
        return (leg["origin"], leg["dest"], ampm, dur_bucket, day_hour)

    # Build mapping from reference pairings
    pattern_next_airports: Dict[tuple, set] = defaultdict(set)
    for pairing in ref_pairings:
        for k in range(len(pairing) - 1):
            fi_leg = leg_map[pairing[k]]
            fj_leg = leg_map[pairing[k + 1]]
            pattern_next_airports[pattern(fi_leg)].add(fj_leg["origin"])

    filtered: Dict[int, List[int]] = {}
    for fid, succ in successors.items():
        leg_i = leg_map[fid]
        t_arr = leg_i["arr_abs"]

        # Tier 1: pattern match
        key   = pattern(leg_i)
        known = pattern_next_airports.get(key)
        if known:
            narrowed = [j for j in succ if leg_map[j]["origin"] in known]
        else:
            narrowed = succ

        # Tier 2: prefer same-duty connections (gap < T_C_MAX) first, then rest
        same_duty = [j for j in narrowed if dep_abs[j] - t_arr <= T_C_MAX]
        rest_conn = [j for j in narrowed if dep_abs[j] - t_arr >  T_C_MAX]

        # Take up to MAX_SUCCESSORS: prioritise same-duty
        combined = same_duty[:MAX_SUCCESSORS] + rest_conn[:(MAX_SUCCESSORS - len(same_duty[:MAX_SUCCESSORS]))]
        if not combined:
            combined = succ[:MAX_SUCCESSORS]

        filtered[fid] = combined[:MAX_SUCCESSORS]

    return filtered


# ── X_i matrix construction ─────────────────────────────────────────────────

def build_xi_matrix(
    leg_i: Dict,
    successors: List[int],
    leg_map: Dict[int, Dict],
    enc: Dict,
    base: str,
) -> np.ndarray:
    """
    X_i: shape (|D_ib|, 27)
    Each row = [base_feats(9) | leg_i_feats(9) | leg_j_feats(9)]
    """
    # base feats: dummy leg at base (dep=arr=base, times=0)
    base_dummy = {
        "origin": base, "dest": base, "aircraft_type": leg_i["aircraft_type"],
        "dep_day": 0, "dep_min": 0, "arr_day": 0, "arr_min": 0, "duration": 0,
    }
    base_feats = _encode_leg(base_dummy, base, enc)
    fi_feats   = _encode_leg(leg_i, base, enc)

    rows = []
    for jid in successors:
        fj_feats = _encode_leg(leg_map[jid], base, enc)
        rows.append(np.concatenate([base_feats, fi_feats, fj_feats]))
    return np.array(rows, dtype=np.float32)


def build_dataset(
    inst: Dict,
    successors: Dict[int, List[int]],
    ref_pairings: List[List[int]],
    enc: Dict,
    base: str,
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Build (X_i, y_i) pairs for one instance/base.
    X_i: (|D_ib|, 27), y_i: index of true successor in D_ib.
    Only includes flights that appear mid-pairing (have a true next flight).
    """
    leg_map = {leg["flight_id"]: leg for leg in inst["legs"]}

    true_next: Dict[int, int] = {}
    for pairing in ref_pairings:
        for k in range(len(pairing) - 1):
            true_next[pairing[k]] = pairing[k + 1]

    X_list, y_list = [], []
    for fid, succ in successors.items():
        if fid not in true_next:
            continue
        true_j = true_next[fid]
        if true_j not in succ:
            continue
        y = succ.index(true_j)
        X = build_xi_matrix(leg_map[fid], succ, leg_map, enc, base)
        X_list.append(X)
        y_list.append(y)

    return X_list, y_list
