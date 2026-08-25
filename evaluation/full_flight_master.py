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



def _solver(time_limit: int, use_gurobi: bool, verbose: bool, threads: int = 1):
    if use_gurobi:
        try:
            return pulp.GUROBI(timeLimit=time_limit, msg=int(verbose))
        except Exception:
            pass
    return pulp.PULP_CBC_CMD(timeLimit=time_limit, threads=threads, msg=int(verbose))


def calibrate_completion_penalties(columns: Sequence[Dict]) -> Dict[str, float]:
    """현재 pool cost scale을 기준으로 completion penalty 초기값을 산정함.

    scale은 순수 policy pairing의 cost로만 계산함. salvage/rescue column은 둘 다
    RL/rollout.py::emit_prefix()/completion/rescue_generator.py::_compute_cost()에서
    IP_DEADHEAD_PENALTY+IP_PAIRING_FIXED_COST가 무조건 더해지는 "억지로 만들어진"
    column이라(is_deadhead=True) cost가 policy pairing보다 크게 나올 수 있음 --
    특히 rescue는 BFS가 타이트한 연결을 보장하지 않아 dead_time까지 훨씬 커질 수
    있음. 이런 column을 포함해서 scale을 잡으면 그중 하나의 비정상적으로 높은
    cost가 그것과 무관한 모든 flight의 reposition/reserve/artificial penalty까지
    같이 부풀려버림(실측: rescue candidate 하나가 cost=80대면 scale이 그만큼 뛰어
    페널티 구조 전체가 왜곡됨).
    """
    finite_costs = []
    for column in columns:
        if column.get("source_type") != "policy":
            continue
        try:
            cost = float(column.get("cost", 0.0))
        except (TypeError, ValueError):
            continue
        if math.isfinite(cost) and cost >= 0:
            finite_costs.append(cost)
    scale = max([1.0] + finite_costs)
    return {
        "reposition_penalty": scale * 2.0,
        "reserve_penalty": scale * 4.0,
        "artificial_penalty": scale * 8.0,
    }


def solve_full_flight_master(
    columns: Sequence[Dict],
    all_flight_ids: Iterable[int],
    *,
    lambda_excess: float = 1.0,
    allow_reposition: bool = False,
    allow_reserve: bool = False,
    allow_artificial: bool = False,
    reposition_penalty: float | None = None,
    reserve_penalty: float | None = None,
    artificial_penalty: float | None = None,
    reposition_flight_ids: Iterable[int] | None = None,
    reserve_flight_ids: Iterable[int] | None = None,
    time_limit: int = 300,
    threads: int = 1,
    use_gurobi: bool = False,
    verbose: bool = False,
) -> Dict:
    """전체 flight를 legal·operational·artificial 중 하나로 명시 처리함."""
    normalized, universe = validate_master_inputs(columns, all_flight_ids)
    calibrated = calibrate_completion_penalties(normalized)
    reposition_penalty = calibrated["reposition_penalty"] if reposition_penalty is None else reposition_penalty
    reserve_penalty = calibrated["reserve_penalty"] if reserve_penalty is None else reserve_penalty
    artificial_penalty = calibrated["artificial_penalty"] if artificial_penalty is None else artificial_penalty
    penalties = {
        "lambda_excess": lambda_excess,
        "reposition_penalty": reposition_penalty,
        "reserve_penalty": reserve_penalty,
        "artificial_penalty": artificial_penalty,
    }
    for name, value in penalties.items():
        if value < 0 or not math.isfinite(value):
            raise FullFlightInputError(f"{name}는 0 이상의 유한값이어야 함")

    universe_set = set(universe)
    reposition_targets = set() if reposition_flight_ids is None else set(reposition_flight_ids)
    reserve_targets = set() if reserve_flight_ids is None else set(reserve_flight_ids)
    for name, targets in (("reposition_flight_ids", reposition_targets), ("reserve_flight_ids", reserve_targets)):
        unknown = targets - universe_set
        if unknown:
            raise FullFlightInputError(f"{name}에 universe 밖 flight ID {sorted(unknown)}")

    enabled_columns = [
        column for column in normalized
        if column["source_type"] in LEGAL_SOURCE_TYPES
        or (column["source_type"] == "reposition" and allow_reposition)
        or (column["source_type"] == "reserve" and allow_reserve)
    ]
    if not universe:
        return {
            "selected": [], "selected_column_ids": [], "n_pairings": 0,
            "status": "Empty", "is_feasible": True, "mip_objective": 0.0,
            "pairing_cost": 0.0, "excess_cost": 0.0,
            "reposition_cost": 0.0, "reserve_cost": 0.0, "artificial_cost": 0.0,
            "covered_flight_ids": [], "uncovered_flight_ids": [],
            "coverage": 1.0, "operational_completion_coverage": 1.0,
            "completion_coverage": 1.0, "excess_flight_ids": [], "excess_count": 0,
            "reposition_flight_ids": [], "reserve_flight_ids": [],
            "artificial_flight_ids": [], "artificial_count": 0,
            "penalties": penalties,
            "selected_count_by_source": {}, "selected_cost_by_source": {},
            "objective_breakdown": {"pairing": 0.0, "excess": 0.0, "reposition": 0.0, "reserve": 0.0, "artificial": 0.0},
        }

    available = {flight_id for column in enabled_columns for flight_id in column["legs"]}
    if allow_reposition:
        available.update(reposition_targets)
    if allow_reserve:
        available.update(reserve_targets)
    impossible = universe_set - available
    if impossible and not allow_artificial:
        zero_by_source = {source: 0 for source in sorted(SUPPORTED_SOURCE_TYPES)}
        return {
            "selected": [], "selected_column_ids": [], "n_pairings": 0,
            "status": "Infeasible", "is_feasible": False, "mip_objective": None,
            "pairing_cost": 0.0, "excess_cost": 0.0,
            "reposition_cost": 0.0, "reserve_cost": 0.0, "artificial_cost": 0.0,
            "covered_flight_ids": [], "operational_covered_flight_ids": [],
            "uncovered_flight_ids": sorted(universe_set), "penalties": penalties,
            "selected_count_by_source": zero_by_source,
            "selected_cost_by_source": {source: 0.0 for source in zero_by_source},
            "objective_breakdown": {"pairing": 0.0, "excess": 0.0, "reposition": 0.0, "reserve": 0.0, "artificial": 0.0},
            "coverage": 0.0, "operational_completion_coverage": 0.0,
            "completion_coverage": 0.0, "excess_flight_ids": [], "excess_count": 0,
            "reposition_flight_ids": [], "reserve_flight_ids": [],
            "artificial_flight_ids": [], "artificial_count": 0,
            "solve_skipped": True,
            "structural_infeasible_flight_ids": sorted(impossible),
        }

    by_flight = defaultdict(list)
    for j, column in enumerate(enabled_columns):
        for flight_id in column["legs"]:
            by_flight[flight_id].append(j)

    problem = pulp.LpProblem("full_flight_master", pulp.LpMinimize)
    x = [pulp.LpVariable(f"x_{j}", cat="Binary") for j in range(len(enabled_columns))]
    excess = {flight_id: pulp.LpVariable(f"excess_{flight_id}", lowBound=0) for flight_id in universe}
    reposition = {
        flight_id: pulp.LpVariable(f"reposition_{flight_id}", cat="Binary")
        for flight_id in universe if allow_reposition and flight_id in reposition_targets
    }
    reserve = {
        flight_id: pulp.LpVariable(f"reserve_{flight_id}", cat="Binary")
        for flight_id in universe if allow_reserve and flight_id in reserve_targets
    }
    artificial = {
        flight_id: pulp.LpVariable(f"artificial_{flight_id}", cat="Binary")
        for flight_id in universe
    } if allow_artificial else {}

    pairing_term = pulp.lpSum(enabled_columns[j]["cost"] * x[j] for j in range(len(x)))
    excess_term = lambda_excess * pulp.lpSum(excess.values())
    reposition_term = reposition_penalty * pulp.lpSum(reposition.values())
    reserve_term = reserve_penalty * pulp.lpSum(reserve.values())
    artificial_term = artificial_penalty * pulp.lpSum(artificial.values())
    problem += pairing_term + excess_term + reposition_term + reserve_term + artificial_term

    for flight_id in universe:
        cover_sum = pulp.lpSum(x[j] for j in by_flight[flight_id])
        completion_sum = reposition.get(flight_id, 0) + reserve.get(flight_id, 0) + artificial.get(flight_id, 0)
        problem += cover_sum + completion_sum >= 1, f"cover_{flight_id}"
        problem += excess[flight_id] >= cover_sum - 1, f"excess_{flight_id}"

    if threads < 1:
        raise FullFlightInputError("threads는 1 이상이어야 함")
    problem.solve(_solver(time_limit, use_gurobi, verbose, threads))
    status = pulp.LpStatus[problem.status]
    is_feasible = status in {"Optimal", "Feasible"}
    selected = [
        enabled_columns[j] for j, variable in enumerate(x)
        if is_feasible and (variable.value() or 0.0) > 0.5
    ]
    def chosen_ids(variables):
        return sorted(
            flight_id for flight_id, variable in variables.items()
            if is_feasible and (variable.value() or 0.0) > 0.5
        )

    legal_covered = set()
    operational_column_covered = set()
    for column in selected:
        target = legal_covered if column["source_type"] in LEGAL_SOURCE_TYPES else operational_column_covered
        target.update(column["legs"])
    reposition_ids = chosen_ids(reposition)
    reserve_ids = chosen_ids(reserve)
    artificial_ids = chosen_ids(artificial)
    operational_covered = legal_covered | operational_column_covered | set(reposition_ids) | set(reserve_ids)
    completed = operational_covered | set(artificial_ids)
    excess_values = {
        flight_id: (excess[flight_id].value() or 0.0) if is_feasible else 0.0
        for flight_id in universe
    }
    excess_ids = sorted(flight_id for flight_id, value in excess_values.items() if value > 0.5)
    pairing_cost = sum(column["cost"] for column in selected)
    excess_cost = lambda_excess * sum(excess_values.values())
    reposition_cost = reposition_penalty * len(reposition_ids)
    reserve_cost = reserve_penalty * len(reserve_ids)
    artificial_cost = artificial_penalty * len(artificial_ids)
    selected_count_by_source = {
        source: sum(column["source_type"] == source for column in selected)
        for source in sorted(SUPPORTED_SOURCE_TYPES)
    }
    selected_cost_by_source = {
        source: sum(column["cost"] for column in selected if column["source_type"] == source)
        for source in sorted(SUPPORTED_SOURCE_TYPES)
    }
    objective_breakdown = {
        "pairing": pairing_cost,
        "excess": excess_cost,
        "reposition": reposition_cost,
        "reserve": reserve_cost,
        "artificial": artificial_cost,
    }

    return {
        "selected": selected,
        "selected_column_ids": [column["column_id"] for column in selected],
        "n_pairings": len(selected), "status": status, "is_feasible": is_feasible,
        "mip_objective": pulp.value(problem.objective) if is_feasible else None,
        "pairing_cost": pairing_cost, "excess_cost": excess_cost,
        "reposition_cost": reposition_cost, "reserve_cost": reserve_cost,
        "artificial_cost": artificial_cost,
        "covered_flight_ids": sorted(legal_covered),
        "operational_covered_flight_ids": sorted(operational_covered),
        "uncovered_flight_ids": sorted(universe_set - completed),
        "penalties": penalties,
        "selected_count_by_source": selected_count_by_source,
        "selected_cost_by_source": selected_cost_by_source,
        "objective_breakdown": objective_breakdown,
        "coverage": len(legal_covered) / len(universe),
        "operational_completion_coverage": len(operational_covered) / len(universe),
        "completion_coverage": len(completed) / len(universe),
        "excess_flight_ids": excess_ids, "excess_count": len(excess_ids),
        "reposition_flight_ids": reposition_ids, "reserve_flight_ids": reserve_ids,
        "artificial_flight_ids": artificial_ids, "artificial_count": len(artificial_ids),
    }
