"""
Set Partitioning for Crew Pairing (Klabjan et al. 2001, Eq. 1)

전체 흐름 (3단계):
  [1] solve_lp_relaxation : x_j ∈ [0,1]로 LP 풀기 → dual variable 추출
  [2] column_reduction     : reduced cost ≤ 0인 pairing만 유지 (+ 안전장치)
  [3] solve_set_partitioning: 남은 pairing으로 IP 풀기 (x_j ∈ {0,1})

  min  Σ_j  c_j * x_j
  s.t. Σ_{j: i ∈ j} x_j = 1   ∀ flight i
       x_j ∈ {0, 1}

solver: PuLP + CBC (default) | Gurobi (use_gurobi=True)
"""

import pulp
from collections import defaultdict
from typing import Dict, List, Optional


def solve_lp_relaxation(
    pairings: List[Dict],
    n_flights: int,
    verbose: bool = False,
) -> Optional[Dict]:
    """
    Klabjan Step 2: Set Partitioning LP relaxation
    x_j ∈ [0,1] (continuous) → dual variable 추출 → reduced cost 계산

    reduced cost: rc_j = c_j - Σ_{i ∈ legs_j} π_i
      - rc_j < 0: 이 pairing을 쓰면 비용이 줄어듦 → IP에 포함할 가치 있음
      - rc_j ≥ 0: 최적해에 포함될 가능성 낮음 → column reduction으로 제거

    Returns: { lp_value, dual_vars, reduced_costs, status }
    None: LP 풀기 실패 시
    """
    if not pairings:
        return None

    flight_to_pairings: Dict[int, List[int]] = defaultdict(list)
    for j, p in enumerate(pairings):
        for leg in p["legs"]:
            flight_to_pairings[leg].append(j)

    covered_flights = set(flight_to_pairings.keys())
    M = len(pairings)

    prob = pulp.LpProblem("crew_pairing_lp", pulp.LpMinimize)

    # x_j ∈ [0, 1] (Binary → Continuous)
    x = [
        pulp.LpVariable(f"x_{j}", lowBound=0, upBound=1, cat="Continuous")
        for j in range(M)
    ]

    # slack: 커버 불가 flight 처리 (high penalty)
    PENALTY = 1e6
    slack = {i: pulp.LpVariable(f"s_{i}", lowBound=0) for i in covered_flights}

    prob += (
        pulp.lpSum(pairings[j]["cost"] * x[j] for j in range(M))
        + PENALTY * pulp.lpSum(slack.values())
    )

    for i in covered_flights:
        prob += (
            pulp.lpSum(x[j] for j in flight_to_pairings[i]) + slack[i] == 1,
            f"cover_{i}",
        )

    prob.solve(pulp.PULP_CBC_CMD(msg=int(verbose)))

    if prob.status != 1:  # not Optimal
        return None

    # dual variable (shadow price) 추출
    pi: Dict[int, float] = {}
    for i in covered_flights:
        pi[i] = prob.constraints[f"cover_{i}"].pi or 0.0

    # reduced cost: rc_j = c_j - Σ π_i
    reduced_costs = [
        pairings[j]["cost"] - sum(pi.get(i, 0.0) for i in pairings[j]["legs"])
        for j in range(M)
    ]

    return {
        "lp_value":      pulp.value(prob.objective),
        "dual_vars":     pi,
        "reduced_costs": reduced_costs,
        "status":        pulp.LpStatus[prob.status],
    }


def column_reduction(
    pairings: List[Dict],
    reduced_costs: List[float],
    threshold: float = 1e-6,
) -> List[Dict]:
    """
    Klabjan Step 3: reduced cost 기반 column reduction

    rc_j ≤ threshold인 pairing만 유지.
    단, 각 flight를 커버하는 pairing이 최소 1개는 남도록 보장
    (안전장치: 제거되면 coverage 0이 되는 flight 보호)

    Args:
        threshold: 기본값 1e-6 ≈ 0 (수치 오차 허용)
    """
    kept_set = {j for j, rc in enumerate(reduced_costs) if rc <= threshold}

    # 각 flight에 대해 최소 1개 pairing 보장
    flight_to_pairings: Dict[int, List[int]] = defaultdict(list)
    for j, p in enumerate(pairings):
        for leg in p["legs"]:
            flight_to_pairings[leg].append(j)

    for pairing_ids in flight_to_pairings.values():
        if not any(j in kept_set for j in pairing_ids):
            # 이 flight를 커버하는 pairing 중 reduced cost 최소인 것 추가
            best_j = min(pairing_ids, key=lambda j: reduced_costs[j])
            kept_set.add(best_j)

    return [pairings[j] for j in sorted(kept_set)]


def solve_set_partitioning(
    pairings: List[Dict],
    n_flights: int,
    time_limit: int  = 300,
    use_gurobi: bool = False,
    verbose: bool    = False,
) -> Dict:
    """
    Set Partitioning IP를 풀어 최적 pairing subset 선택

    Args:
        pairings:   pairing dict 리스트 (legs, cost 필드 필요)
        n_flights:  전체 flight 수 (flight ID: 0 ~ n_flights-1)
        time_limit: solver 제한 시간 (초)
        use_gurobi: True면 Gurobi 사용, 실패 시 CBC로 fallback
        verbose:    solver 로그 출력 여부

    Returns:
        selected, n_pairings, total_cost, coverage, status, uncoverable
    """
    if not pairings:
        return {
            "selected": [], "n_pairings": 0, "total_cost": 0.0,
            "coverage": 0.0, "status": "no_pairings", "uncoverable": n_flights,
        }

    # flight → pairing indices 역인덱스
    flight_to_pairings: Dict[int, List[int]] = defaultdict(list)
    for j, p in enumerate(pairings):
        for leg in p["legs"]:
            flight_to_pairings[leg].append(j)

    covered_flights  = set(flight_to_pairings.keys())
    uncoverable      = set(range(n_flights)) - covered_flights

    M = len(pairings)
    prob = pulp.LpProblem("crew_pairing_sp", pulp.LpMinimize)

    # 결정 변수: x_j ∈ {0, 1}
    x = [pulp.LpVariable(f"x_{j}", cat="Binary") for j in range(M)]

    # slack 변수: 커버 불가능한 flight 처리 (high penalty)
    # → infeasible 방지용, 실제로 선택되면 문제 있다는 신호
    PENALTY = 1e6
    slack = {i: pulp.LpVariable(f"s_{i}", lowBound=0) for i in covered_flights}

    # 목적함수
    prob += (
        pulp.lpSum(pairings[j]["cost"] * x[j] for j in range(M))
        + PENALTY * pulp.lpSum(slack.values())
    )

    # 커버리지 제약: 각 flight 정확히 1번 (slack으로 등호 완화)
    for i in covered_flights:
        prob += (
            pulp.lpSum(x[j] for j in flight_to_pairings[i]) + slack[i] == 1,
            f"cover_{i}",
        )

    # Solver 선택
    if use_gurobi:
        try:
            solver = pulp.GUROBI(timeLimit=time_limit, msg=int(verbose))
        except Exception:
            print("[warn] Gurobi 사용 불가 → CBC로 대체")
            solver = pulp.PULP_CBC_CMD(timeLimit=time_limit, msg=int(verbose))
    else:
        solver = pulp.PULP_CBC_CMD(timeLimit=time_limit, msg=int(verbose))

    prob.solve(solver)

    # 결과 추출
    selected = [pairings[j] for j in range(M) if (x[j].value() or 0) > 0.5]

    covered_legs = set()
    for p in selected:
        covered_legs.update(p["legs"])
    covered_count = len(covered_legs & set(range(n_flights)))

    return {
        "selected":    selected,
        "n_pairings":  len(selected),
        "total_cost":  sum(p["cost"] for p in selected),
        "coverage":    covered_count / n_flights if n_flights > 0 else 0.0,
        "status":      pulp.LpStatus[prob.status],
        "uncoverable": len(uncoverable),
    }
