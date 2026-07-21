"""
Reference pairing generator for CPP instances.

Two modes:
  1. CG-based (default, recommended):
     Uses I2CG column generation to produce near-optimal reference pairings.
     These become the DNN training labels (as in the paper: gap ~0.12%).
     Falls back to greedy if CG fails or is too slow.

  2. Greedy (legacy fallback):
     Fast but lower quality (~87% coverage). Kept for compatibility.

Usage:
    from dnn.reference import generate_reference_pairings
    pairings = generate_reference_pairings(inst)               # CG (default)
    pairings = generate_reference_pairings(inst, method="greedy")  # fast
"""

from __future__ import annotations
from collections import defaultdict
from typing import List, Dict, Tuple

from .dataset import T_C_MIN, T_R_MIN, T_C_MAX, T_R_MAX, D_BAR, D_BAR_DUTY, T_BAR_D, T_BAR_W, F_MAX


# ── CG-based reference (recommended) ─────────────────────────────────────────

def generate_reference_pairings(
    inst:    Dict,
    method:  str = "cg",
    verbose: bool = False,
) -> List[List[int]]:
    """
    Generate reference pairings for one instance.

    Args:
        inst:   instance dict with keys 'legs', 'bases', 'aircraft_type'
        method: 'cg'     -> I2CG column generation (near-optimal, recommended)
                'greedy' -> legacy greedy (fast, ~87% coverage)
        verbose: print CG progress

    Returns:
        List of pairings (each = list of flight_ids).
    """
    if method == "cg":
        try:
            from solver.icg import generate_reference_via_cg
            pairings = generate_reference_via_cg(
                inst, max_fail=3, max_iter=50, max_labels=200, verbose=verbose
            )
            if pairings:
                return pairings
        except Exception as e:
            print(f"  [reference] CG failed ({e}), falling back to greedy", flush=True)

    return _greedy_reference(inst)


# ── Legacy greedy (kept for fallback / fast mode) ─────────────────────────────

def _greedy_reference(inst: Dict) -> List[List[int]]:
    builder     = GreedyPairingBuilder(inst["legs"])
    all_pairings: List[List[int]] = []
    global_used = set()
    for base in inst["bases"]:
        pairings = builder.build_pairings_for_base(base, used_global=global_used)
        for p in pairings:
            global_used.update(p)
        all_pairings.extend(pairings)
    return all_pairings


class GreedyPairingBuilder:

    def __init__(self, legs: List[Dict]):
        self.legs    = legs
        self.leg_map = {leg["flight_id"]: leg for leg in legs}

        # Index by origin airport: sorted list of (dep_abs, flight_id)
        self.by_origin: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        for leg in legs:
            self.by_origin[leg["origin"]].append((leg["dep_abs"], leg["flight_id"]))
        for k in self.by_origin:
            self.by_origin[k].sort()

    def _successors(self, airport: str, after_abs: int,
                    max_gap: int, used: set) -> List[int]:
        result = []
        for dep_abs, fid in self.by_origin.get(airport, []):
            if fid in used:
                continue
            gap = dep_abs - after_abs
            if gap < T_C_MIN:
                continue
            if gap > max_gap:
                break
            result.append(fid)
        return result

    def _build_duty(
        self, cur_airport: str, cur_abs: int, used: set
    ) -> Tuple[List[int], str, int]:
        """
        Build one duty greedily.
        Returns (flight_ids_in_duty, last_airport, last_arr_abs).
        """
        flights    = []
        duty_start = cur_abs
        work_time  = 0

        while len(flights) < F_MAX:
            remaining_duty = T_BAR_D - (cur_abs - duty_start)
            if remaining_duty <= 0:
                break

            cands = self._successors(cur_airport, cur_abs,
                                     max_gap=min(T_C_MAX, remaining_duty),
                                     used=used)
            if not cands:
                break

            fid  = cands[0]
            leg  = self.leg_map[fid]
            new_work = work_time + leg["duration"]
            new_duty = leg["arr_abs"] - duty_start
            if new_work > T_BAR_W or new_duty > T_BAR_D:
                break

            flights.append(fid)
            used.add(fid)
            work_time   = new_work
            cur_airport = leg["dest"]
            cur_abs     = leg["arr_abs"]

        return flights, cur_airport, cur_abs

    def build_pairings_for_base(
        self, base: str, max_pairings: int = 500, used_global: set = None
    ) -> List[List[int]]:
        """Build all pairings starting (and ideally ending) at base."""
        used     = set(used_global) if used_global else set()
        pairings = []

        starts = [fid for dep_abs, fid in self.by_origin.get(base, [])]

        for start_fid in starts:
            if start_fid in used or len(pairings) >= max_pairings:
                continue

            leg0         = self.leg_map[start_fid]
            pairing      = [start_fid]
            used.add(start_fid)

            cur_airport  = leg0["dest"]
            cur_abs      = leg0["arr_abs"]
            pairing_start = leg0["dep_abs"]
            n_duties     = 1

            while n_duties < D_BAR_DUTY:
                pairing_days = (cur_abs - pairing_start) / 1440
                if pairing_days >= D_BAR:
                    break

                # First try short connection (same duty continuation)
                duty_flights, after_airport, after_abs = self._build_duty(
                    cur_airport, cur_abs, used
                )
                if duty_flights:
                    pairing.extend(duty_flights)
                    cur_airport = after_airport
                    cur_abs     = after_abs
                    if cur_airport == base:
                        break
                    # rest
                    rest_cands = self._successors(cur_airport, cur_abs,
                                                  max_gap=T_R_MAX, used=used)
                    if not rest_cands:
                        break
                    # pick first rest candidate with gap >= T_R_MIN
                    rest_fid = None
                    for rfid in rest_cands:
                        gap = self.leg_map[rfid]["dep_abs"] - cur_abs
                        if gap >= T_R_MIN:
                            rest_fid = rfid
                            break
                    if rest_fid is None:
                        break
                    rest_leg = self.leg_map[rest_fid]
                    pairing.append(rest_fid)
                    used.add(rest_fid)
                    cur_airport = rest_leg["dest"]
                    cur_abs     = rest_leg["arr_abs"]
                    n_duties   += 1
                else:
                    break

                if cur_airport == base:
                    break

            if len(pairing) >= 2:
                pairings.append(pairing)

        return pairings


