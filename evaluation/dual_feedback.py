"""Full-universe LP dual과 반복 candidate 생성을 연결하는 모듈."""

from __future__ import annotations

from collections import defaultdict
import math
import random
import time
from typing import Callable, Dict, Iterable, List, Sequence

import pulp


class DualFeedbackError(ValueError):
    """Dual feedback 입력 또는 solver 계약 위반."""


def solve_full_universe_lp(
    columns: Sequence[Dict],
    all_flight_ids: Iterable[int],
    *,
    lambda_excess: float = 1.0,
    artificial_penalty: float = 1000.0,
    solver: str = "cbc",
    threads: int = 1,
    time_limit: float | None = None,
    verbose: bool = False,
) -> Dict:
    """후보가 없는 flight도 artificial slack으로 포함한 LP를 풂."""
    universe = tuple(all_flight_ids)
    if len(set(universe)) != len(universe):
        raise DualFeedbackError("all_flight_ids에 중복이 있음")
    universe_set = set(universe)
    if lambda_excess < 0 or artificial_penalty <= 0:
        raise DualFeedbackError("penalty는 유효한 양수 범위여야 함")

    normalized = []
    by_flight = defaultdict(list)
    for index, raw in enumerate(columns):
        column = dict(raw)
        legs = list(column.get("legs", []))
        if not legs or len(set(legs)) != len(legs):
            raise DualFeedbackError(f"column {index}: legs가 비었거나 중복됨")
        unknown = set(legs) - universe_set
        if unknown:
            raise DualFeedbackError(f"column {index}: universe 밖 flight {sorted(unknown)}")
        cost = float(column["cost"])
        if not math.isfinite(cost) or cost < 0:
            raise DualFeedbackError(f"column {index}: 음수 cost")
        column.update(legs=legs, cost=cost)
        normalized.append(column)
        for flight_id in legs:
            by_flight[flight_id].append(index)

    problem = pulp.LpProblem("full_universe_dual_master", pulp.LpMinimize)
    x = [
        pulp.LpVariable(f"x_{j}", lowBound=0, upBound=1, cat="Continuous")
        for j in range(len(normalized))
    ]
    excess = {
        flight_id: pulp.LpVariable(f"excess_{flight_id}", lowBound=0)
        for flight_id in universe
    }
    artificial = {
        flight_id: pulp.LpVariable(
            f"artificial_{flight_id}", lowBound=0, upBound=1, cat="Continuous"
        )
        for flight_id in universe
    }
    problem += (
        pulp.lpSum(normalized[j]["cost"] * x[j] for j in range(len(x)))
        + lambda_excess * pulp.lpSum(excess.values())
        + artificial_penalty * pulp.lpSum(artificial.values())
    )
    for flight_id in universe:
        cover_sum = pulp.lpSum(x[j] for j in by_flight[flight_id])
        problem += cover_sum + artificial[flight_id] >= 1, f"cover_{flight_id}"
        problem += excess[flight_id] >= cover_sum - 1, f"excess_{flight_id}"

    if solver == "gurobi":
        lp_solver = pulp.GUROBI(
            mip=False, msg=bool(verbose), timeLimit=time_limit,
            Threads=max(1, int(threads)),
        )
    elif solver == "cbc":
        lp_solver = pulp.PULP_CBC_CMD(
            mip=False, msg=int(verbose), timeLimit=time_limit,
            threads=max(1, int(threads)),
        )
    else:
        raise DualFeedbackError(f"지원하지 않는 LP solver: {solver}")

    started = time.monotonic()
    print(
        f"[dual-lp] solver={solver}, columns={len(normalized)}, "
        f"flights={len(universe)}, threads={max(1, int(threads))}, "
        f"time_limit={time_limit}",
        flush=True,
    )
    problem.solve(lp_solver)
    elapsed = time.monotonic() - started
    print(
        f"[dual-lp] status={pulp.LpStatus[problem.status]}, elapsed={elapsed:.1f}s",
        flush=True,
    )
    if problem.status != pulp.LpStatusOptimal:
        raise DualFeedbackError(
            f"LP solve 실패: {pulp.LpStatus[problem.status]} "
            f"(solver={solver}, elapsed={elapsed:.1f}s)"
        )

    coverage_dual = {
        flight_id: float(problem.constraints[f"cover_{flight_id}"].pi or 0.0)
        for flight_id in universe
    }
    excess_dual = {
        flight_id: float(problem.constraints[f"excess_{flight_id}"].pi or 0.0)
        for flight_id in universe
    }
    net_dual = {
        flight_id: coverage_dual[flight_id] - excess_dual[flight_id]
        for flight_id in universe
    }
    reduced_costs = [float(variable.dj or 0.0) for variable in x]
    formula_reduced_costs = [
        normalized[j]["cost"] - sum(net_dual[i] for i in normalized[j]["legs"])
        for j in range(len(normalized))
    ]
    artificial_ids = sorted(
        flight_id for flight_id, variable in artificial.items()
        if (variable.value() or 0.0) > 1e-7
    )
    return {
        "status": pulp.LpStatus[problem.status],
        "lp_objective": float(pulp.value(problem.objective) or 0.0),
        "coverage_dual": coverage_dual,
        "excess_dual": excess_dual,
        "net_dual": net_dual,
        "reduced_costs": reduced_costs,
        "formula_reduced_costs": formula_reduced_costs,
        "artificial_flight_ids": artificial_ids,
        "artificial_count": len(artificial_ids),
        "zero_cost_count": sum(column["cost"] == 0 for column in normalized),
        "zero_cost_fraction": (
            sum(column["cost"] == 0 for column in normalized) / len(normalized)
            if normalized else 0.0
        ),
    }


def normalize_dual(dual: Dict[int, float], clip: float = 5.0) -> Dict[int, float]:
    """Instance 내부 max-abs 정규화 후 범위를 제한함."""
    scale = max([abs(float(value)) for value in dual.values()] + [1.0])
    return {
        flight_id: max(-clip, min(clip, float(value) / scale))
        for flight_id, value in dual.items()
    }


def build_dual_signal(lp_result: Dict, mode: str, rng=None) -> Dict[int, float]:
    """동일 LP 결과를 실험 대조군별 action signal로 변환함."""
    signal = normalize_dual(lp_result["net_dual"])
    if mode == "real":
        return signal
    if mode == "zero":
        return {flight_id: 0.0 for flight_id in signal}
    if mode == "uncovered-only":
        uncovered = set(lp_result["artificial_flight_ids"])
        return {
            flight_id: 1.0 if flight_id in uncovered else 0.0
            for flight_id in signal
        }
    if mode == "uniform":
        mean_signal = sum(signal.values()) / len(signal) if signal else 0.0
        return {flight_id: mean_signal for flight_id in signal}
    if mode == "shuffled":
        keys = sorted(signal)
        values = [signal[key] for key in keys]
        (rng or random).shuffle(values)
        return dict(zip(keys, values))
    raise DualFeedbackError(f"지원하지 않는 dual mode: {mode}")


def merge_unique_columns(existing: Sequence[Dict], new_columns: Sequence[Dict]) -> List[Dict]:
    """순서가 같은 leg sequence를 하나로 합치고 더 낮은 cost를 유지함."""
    merged = {tuple(column["legs"]): dict(column) for column in existing}
    for raw in new_columns:
        column = dict(raw)
        key = tuple(column["legs"])
        if key not in merged or float(column["cost"]) < float(merged[key]["cost"]):
            merged[key] = column
    return list(merged.values())


def run_iterative_dual_feedback(
    initial_columns: Sequence[Dict],
    all_flight_ids: Iterable[int],
    generate_columns: Callable[[Dict[int, float], int], Sequence[Dict]],
    *,
    max_iterations: int = 3,
    lambda_excess: float = 1.0,
    artificial_penalty: float = 1000.0,
    solver: str = "cbc",
    threads: int = 1,
    time_limit: float | None = None,
) -> Dict:
    """LP→dual→candidate 생성→병합을 반복하고 trace를 반환함."""
    pool = [dict(column) for column in initial_columns]
    trace: List[Dict] = []
    for iteration in range(max_iterations + 1):
        lp = solve_full_universe_lp(
            pool, all_flight_ids,
            lambda_excess=lambda_excess,
            artificial_penalty=artificial_penalty,
            solver=solver,
            threads=threads,
            time_limit=time_limit,
        )
        entry = {
            "iteration": iteration,
            "pool_size": len(pool),
            "lp_objective": lp["lp_objective"],
            "artificial_count": lp["artificial_count"],
            "zero_cost_count": lp["zero_cost_count"],
            "zero_cost_fraction": lp["zero_cost_fraction"],
        }
        trace.append(entry)
        if iteration == max_iterations:
            break
        signal = normalize_dual(lp["net_dual"])
        generated = list(generate_columns(signal, iteration + 1))
        before = len(pool)
        pool = merge_unique_columns(pool, generated)
        entry["generated_count"] = len(generated)
        entry["new_unique_count"] = len(pool) - before
        if len(pool) == before:
            break
    return {"columns": pool, "trace": trace, "last_lp": lp}
