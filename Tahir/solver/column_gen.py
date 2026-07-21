"""
Phase 3: DNN-guided column generation
Builds pairing columns using DNN connection probabilities (P^b matrix).
Implements:
  1. Reduced subproblem: remove arcs where P^b[i][j] == 0
  2. CPBS resource: priority = product of Psi_ij_b ranks (lower = better)
  3. Greedy beam search over reduced successor graph
"""

from __future__ import annotations
import heapq
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

from .constraints import T_C_MIN, T_C_MAX, T_R_MIN, T_R_MAX, D_BAR, D_BAR_DUTY, T_BAR_D, T_BAR_W, F_MAX, pairing_cost


def build_probability_matrix(
    legs: List[Dict],
    successors: Dict[int, List[int]],
    model,           # keras Model
    enc: Dict,
    base: str,
    norm_mean: np.ndarray = None,
    norm_std:  np.ndarray = None,
) -> Dict[int, Dict[int, float]]:
    """
    Compute P^b: for each leg i, probability distribution over D_ib.
    Returns {flight_id_i: {flight_id_j: p_ij}} where p_ij > 0.
    """
    import tensorflow as tf
    from dnn.dataset import build_xi_matrix

    leg_map = {leg["flight_id"]: leg for leg in legs}
    num_cols = list(range(4, 9)) + list(range(13, 18)) + list(range(22, 27))
    P: Dict[int, Dict[int, float]] = {}

    for fid, succ in successors.items():
        if not succ:
            P[fid] = {}
            continue

        X = build_xi_matrix(leg_map[fid], succ, leg_map, enc, base)
        # Normalise
        if norm_mean is not None:
            X[:, num_cols] = (X[:, num_cols] - norm_mean) / norm_std

        X_input = tf.constant(X[np.newaxis])  # (1, K, 27)
        probs   = model(X_input, training=False).numpy()[0]  # (K,)

        P[fid] = {succ[k]: float(probs[k]) for k in range(len(succ))}

    return P


def compute_psi(
    P: Dict[int, Dict[int, float]]
) -> Dict[int, Dict[int, int]]:
    """
    Compute Psi_ij_b: rank of j in D+_ib (probability descending, 1-based).
    classMax_b + 1 for j not in D+_ib.
    """
    # classMax = max |D+_ib| across all flights
    class_max = max((len(v) for v in P.values()), default=1)

    Psi: Dict[int, Dict[int, int]] = {}
    for fid, prob_dict in P.items():
        if not prob_dict:
            Psi[fid] = {}
            continue
        # Sort by probability descending
        sorted_j = sorted(prob_dict, key=lambda j: prob_dict[j], reverse=True)
        Psi[fid] = {j: r + 1 for r, j in enumerate(sorted_j)}

    return Psi, class_max


def generate_columns_beam(
    legs: List[Dict],
    bases: List[str],
    P: Dict[int, Dict[int, float]],
    Psi: Dict[int, Dict[int, int]],
    class_max: int,
    beam_width: int = 10,
    max_columns: int = 500,
) -> List[List[int]]:
    """
    Beam search over the DNN-reduced successor graph.
    State: (cpbs_score, pairing_so_far, cur_airport, cur_abs,
            n_duties, duty_start, work_time, n_duty_flights)
    Lower CPBS = higher probability path → priority queue (min-heap).

    Generates feasible pairing columns for the set partitioning problem.
    """
    leg_map  = {leg["flight_id"]: leg for leg in legs}
    dep_abs  = {leg["flight_id"]: leg["dep_abs"] for leg in legs}
    columns  = []
    seen     = set()

    for base in bases:
        # Starting legs: depart from base
        start_legs = [leg for leg in legs if leg["origin"] == base]
        start_legs.sort(key=lambda l: l["dep_abs"])

        for start_leg in start_legs:
            sid = start_leg["flight_id"]

            # Initial state: (cpbs=1, pairing=[sid], ...)
            heap = [(
                1,                      # cpbs (product of Psi ranks)
                [sid],                  # pairing
                start_leg["dest"],      # cur_airport
                start_leg["arr_abs"],   # cur_abs
                1,                      # n_duties
                start_leg["dep_abs"],   # duty_start
                start_leg["duration"],  # work_time
                1,                      # n_duty_flights
            )]

            best_per_state = {}  # (cur_airport, cur_abs) → best cpbs

            while heap and len(columns) < max_columns:
                cpbs, pairing, cur_ap, cur_abs, n_duties, d_start, work, n_df = heapq.heappop(heap)

                state_key = (cur_ap, cur_abs, n_duties, d_start)
                if best_per_state.get(state_key, float("inf")) <= cpbs:
                    continue
                best_per_state[state_key] = cpbs

                # Prune if back at base with ≥ 2 flights → record column
                if cur_ap == base and len(pairing) >= 2:
                    key = tuple(pairing)
                    if key not in seen:
                        seen.add(key)
                        columns.append(list(pairing))

                if len(pairing) >= 20:  # safety cap
                    continue

                last_fid = pairing[-1]
                prob_dict = P.get(last_fid, {})

                # Expand successors (reduced: only j where P[i][j] > 0)
                succ_items = sorted(prob_dict.items(), key=lambda x: -x[1])[:beam_width]

                for jid, p_ij in succ_items:
                    if jid in set(pairing):
                        continue
                    j_leg = leg_map[jid]
                    gap   = j_leg["dep_abs"] - cur_abs

                    if gap < T_C_MIN:
                        continue

                    psi_val = Psi.get(last_fid, {}).get(jid, class_max + 1)
                    new_cpbs = cpbs * psi_val

                    pairing_days = (j_leg["arr_abs"] - leg_map[sid]["dep_abs"]) / 1440
                    if pairing_days > D_BAR:
                        continue

                    if gap <= T_C_MAX:
                        # Same duty
                        new_work = work + j_leg["duration"]
                        new_duty = j_leg["arr_abs"] - d_start
                        new_ndf  = n_df + 1
                        if new_work > T_BAR_W or new_duty > T_BAR_D or new_ndf > F_MAX:
                            continue
                        heapq.heappush(heap, (
                            new_cpbs,
                            pairing + [jid],
                            j_leg["dest"], j_leg["arr_abs"],
                            n_duties, d_start, new_work, new_ndf,
                        ))
                    else:
                        # Rest → new duty
                        if gap < T_R_MIN:
                            continue
                        new_nd = n_duties + 1
                        if new_nd > D_BAR_DUTY:
                            continue
                        heapq.heappush(heap, (
                            new_cpbs,
                            pairing + [jid],
                            j_leg["dest"], j_leg["arr_abs"],
                            new_nd, j_leg["dep_abs"], j_leg["duration"], 1,
                        ))

    return columns
