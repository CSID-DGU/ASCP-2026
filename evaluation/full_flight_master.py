"""전체 운항편을 명시적으로 다루는 V2 full-flight master.

기존 ``set_partition.solve_set_covering``은 AAAI restricted-master 재현용으로
유지하고, 이 모듈은 candidate가 없는 운항편도 문제에서 제외하지 않는 별도 경로를 제공함.
"""

from __future__ import annotations

import math
from collections import defaultdict

import pulp
from typing import Dict, Iterable, List, Sequence, Tuple


LEGAL_SOURCE_TYPES = frozenset({"policy", "salvage", "rescue"})
OPERATIONAL_SOURCE_TYPES = frozenset({"reposition", "reserve"})
SUPPORTED_SOURCE_TYPES = LEGAL_SOURCE_TYPES | OPERATIONAL_SOURCE_TYPES


class FullFlightInputError(ValueError):
    """Full-flight master 입력 계약 위반."""


def validate_master_inputs(
    columns: Sequence[Dict],
    all_flight_ids: Iterable[int],
) -> Tuple[List[Dict], Tuple[int, ...]]:
    """입력 universe와 column을 검증하고 결정적인 순서로 정규화함."""
    universe = tuple(all_flight_ids)
    if len(set(universe)) != len(universe):
        raise FullFlightInputError("all_flight_ids에 중복 ID가 있음")

    universe_set = set(universe)
    normalized: List[Dict] = []
    seen_column_ids = set()

    for index, raw in enumerate(columns):
        column = dict(raw)
        column_id = column.get("column_id", f"column_{index}")
        if column_id in seen_column_ids:
            raise FullFlightInputError(f"중복 column_id: {column_id}")
        seen_column_ids.add(column_id)

        legs = list(column.get("legs", []))
        if not legs:
            raise FullFlightInputError(f"{column_id}: legs가 비어 있음")
        if len(set(legs)) != len(legs):
            raise FullFlightInputError(f"{column_id}: column 내부 중복 flight가 있음")
        unknown = sorted(set(legs) - universe_set)
        if unknown:
            raise FullFlightInputError(f"{column_id}: universe 밖 flight ID {unknown}")

        source_type = column.get("source_type")
        if source_type not in SUPPORTED_SOURCE_TYPES:
            raise FullFlightInputError(f"{column_id}: 지원하지 않는 source_type {source_type!r}")
        if source_type in LEGAL_SOURCE_TYPES and column.get("is_legal") is not True:
            raise FullFlightInputError(f"{column_id}: legal column은 is_legal=True여야 함")

        try:
            cost = float(column["cost"])
        except (KeyError, TypeError, ValueError) as exc:
            raise FullFlightInputError(f"{column_id}: 유효한 cost가 필요함") from exc
        if not math.isfinite(cost) or cost < 0:
            raise FullFlightInputError(f"{column_id}: cost는 0 이상의 유한값이어야 함")

        column.update(column_id=column_id, legs=legs, cost=cost)
        normalized.append(column)

    return normalized, universe



def _solver(time_limit: int, use_gurobi: bool, verbose: bool):
    if use_gurobi:
        try:
            return pulp.GUROBI(timeLimit=time_limit, msg=int(verbose))
        except Exception:
            pass
    return pulp.PULP_CBC_CMD(timeLimit=time_limit, msg=int(verbose))


def solve_full_flight_master(
    columns: Sequence[Dict],
    all_flight_ids: Iterable[int],
    *,
    lambda_excess: float = 1.0,
    time_limit: int = 300,
    use_gurobi: bool = False,
    verbose: bool = False,
) -> Dict:
    """모든 global flight ID에 coverage constraint를 생성하는 최소 master."""
    normalized, universe = validate_master_inputs(columns, all_flight_ids)
    if lambda_excess < 0 or not math.isfinite(lambda_excess):
        raise FullFlightInputError("lambda_excess는 0 이상의 유한값이어야 함")
    if not universe:
        return {
            "selected": [], "selected_column_ids": [], "n_pairings": 0,
            "status": "Empty", "is_feasible": True, "mip_objective": 0.0,
            "pairing_cost": 0.0, "excess_cost": 0.0,
            "covered_flight_ids": [], "uncovered_flight_ids": [],
            "coverage": 1.0, "excess_flight_ids": [], "excess_count": 0,
        }

    by_flight = defaultdict(list)
    for j, column in enumerate(normalized):
        for flight_id in column["legs"]:
            by_flight[flight_id].append(j)

    problem = pulp.LpProblem("full_flight_master", pulp.LpMinimize)
    x = [pulp.LpVariable(f"x_{j}", cat="Binary") for j in range(len(normalized))]
    excess = {
        flight_id: pulp.LpVariable(f"excess_{flight_id}", lowBound=0)
        for flight_id in universe
    }
    pairing_term = pulp.lpSum(normalized[j]["cost"] * x[j] for j in range(len(x)))
    excess_term = lambda_excess * pulp.lpSum(excess.values())
    problem += pairing_term + excess_term

    for flight_id in universe:
        cover_sum = pulp.lpSum(x[j] for j in by_flight[flight_id])
        problem += cover_sum >= 1, f"cover_{flight_id}"
        problem += excess[flight_id] >= cover_sum - 1, f"excess_{flight_id}"

    problem.solve(_solver(time_limit, use_gurobi, verbose))
    status = pulp.LpStatus[problem.status]
    is_feasible = status in {"Optimal", "Feasible"}
    selected = [
        normalized[j] for j, variable in enumerate(x)
        if is_feasible and (variable.value() or 0.0) > 0.5
    ]
    covered = set()
    for column in selected:
        covered.update(column["legs"])
    excess_ids = [
        flight_id for flight_id in universe
        if is_feasible and (excess[flight_id].value() or 0.0) > 0.5
    ]
    pairing_cost = sum(column["cost"] for column in selected)
    excess_cost = lambda_excess * sum(
        excess[flight_id].value() or 0.0 for flight_id in universe
    ) if is_feasible else 0.0

    return {
        "selected": selected,
        "selected_column_ids": [column["column_id"] for column in selected],
        "n_pairings": len(selected),
        "status": status,
        "is_feasible": is_feasible,
        "mip_objective": pulp.value(problem.objective) if is_feasible else None,
        "pairing_cost": pairing_cost,
        "excess_cost": excess_cost,
        "covered_flight_ids": sorted(covered),
        "uncovered_flight_ids": sorted(set(universe) - covered),
        "coverage": len(covered) / len(universe),
        "excess_flight_ids": sorted(excess_ids),
        "excess_count": len(excess_ids),
    }
