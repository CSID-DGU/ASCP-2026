"""
Set Covering for Crew Pairing

전체 흐름 (3단계):
  [1] solve_lp_relaxation : x_j ∈ [0,1]로 LP 풀기 → dual variable 추출
  [2] column_reduction     : reduced cost ≤ 0인 pairing만 유지 (+ 안전장치)
  [3] solve_set_covering   : 남은 pairing으로 IP 풀기 (x_j ∈ {0,1})

  min  Σ_j  c_j * x_j  +  λ_DH * Σ_i d_i
  s.t. Σ_{j: i ∈ j} x_j  ≥  1   ∀ flight i   (Set Covering)
       d_i = Σ_{j: i ∈ j} x_j - 1             (deadhead 횟수)
       x_j ∈ {0, 1},  d_i ≥ 0

Set Covering(≥1)이므로 같은 flight를 여러 pairing이 커버 가능 → Deadhead 허용.

solver: PuLP + CBC (default) | Gurobi (use_gurobi=True)
"""

import pulp
from collections import defaultdict
from typing import Dict, List, Optional


def solve_lp_relaxation(
    pairings: List[Dict],
    lambda_dh: float = 1.0,
    verbose: bool = False,
) -> Optional[Dict]:
    """
    Set Covering LP relaxation
    x_j ∈ [0,1] (continuous) → dual variable 추출 → reduced cost 계산

    IP와 동일한 목적함수(DH 패널티 포함)로 LP를 구성해 column reduction의
    이론적 근거를 유지한다.

    reduced cost: rc_j = c_j - Σ_{i ∈ legs_j} μ_i^cov
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

    x = [
        pulp.LpVariable(f"x_{j}", lowBound=0, upBound=1, cat="Continuous")
        for j in range(M)
    ]
    d = {i: pulp.LpVariable(f"d_{i}", lowBound=0, cat="Continuous") for i in covered_flights}

    prob += (
        pulp.lpSum(pairings[j]["cost"] * x[j] for j in range(M))
        + lambda_dh * pulp.lpSum(d.values())
    )

    for i in covered_flights:
        cover_sum = pulp.lpSum(x[j] for j in flight_to_pairings[i])
        prob += (cover_sum >= 1,          f"cover_{i}")
        prob += (d[i] >= cover_sum - 1,   f"dh_{i}")

    prob.solve(pulp.PULP_CBC_CMD(msg=int(verbose)))

    if prob.status != 1:
        return None

    mu_cov: Dict[int, float] = {}
    nu_exc: Dict[int, float] = {}
    for i in covered_flights:
        mu_cov[i] = prob.constraints[f"cover_{i}"].pi or 0.0
        # dh_i 제약(d[i] >= cover_sum-1)의 dual — 이 flight가 지금 pool 기준으로 얼마나
        # 중복 커버(deadhead)되고 있는지의 그림자가격. binding일 때 lambda_dh와 같아짐.
        # cover_i dual(μ^cov, "부족분 채우기" 신호)과 반대 방향으로 RL에 피드백하는 데 사용
        # (μ^cov는 보상에 더하고, ν^exc는 빼서 "이미 넘치는 flight는 그만 채워라"는 신호를 줌).
        nu_exc[i] = prob.constraints[f"dh_{i}"].pi or 0.0

    reduced_costs = [
        pairings[j]["cost"] - sum(mu_cov.get(i, 0.0) for i in pairings[j]["legs"])
        for j in range(M)
    ]

    return {
        "lp_value":      pulp.value(prob.objective),
        "dual_vars":     mu_cov,
        "dh_dual_vars":  nu_exc,
        "reduced_costs": reduced_costs,
        "status":        pulp.LpStatus[prob.status],
    }


def column_reduction(
    pairings: List[Dict],
    reduced_costs: List[float],
    threshold: float = 1e-6,
) -> List[Dict]:
    """
    reduced cost 기반 column reduction

    rc_j ≤ threshold인 pairing만 유지.
    단, 각 flight를 커버하는 pairing이 최소 1개는 남도록 보장.

    Args:
        threshold: 기본값 1e-6 ≈ 0 (수치 오차 허용)
    """
    kept_set = {j for j, rc in enumerate(reduced_costs) if rc <= threshold}

    flight_to_pairings: Dict[int, List[int]] = defaultdict(list)
    for j, p in enumerate(pairings):
        for leg in p["legs"]:
            flight_to_pairings[leg].append(j)

    for pairing_ids in flight_to_pairings.values():
        if not any(j in kept_set for j in pairing_ids):
            best_j = min(pairing_ids, key=lambda j: reduced_costs[j])
            kept_set.add(best_j)

    return [pairings[j] for j in sorted(kept_set)]


def solve_set_covering(
    pairings: List[Dict],
    n_flights: int,
    lambda_dh: float = 1.0,
    time_limit: int  = 300,
    use_gurobi: bool = False,
    verbose: bool    = False,
) -> Dict:
    """
    Set Covering IP를 풀어 최적 pairing subset 선택

    Set Covering(≥1): 같은 flight를 여러 pairing이 커버 가능 → Deadhead 허용.
    lambda_dh로 DH 패널티를 조절한다 (0이면 DH 억제 없음).

    Args:
        pairings:   pairing dict 리스트 (legs, cost 필드 필요)
        n_flights:  전체 flight 수 (flight ID: 0 ~ n_flights-1)
        lambda_dh:  DH 패널티 가중치
        time_limit: solver 제한 시간 (초)
        use_gurobi: True면 Gurobi 사용, 실패 시 CBC로 fallback
        verbose:    solver 로그 출력 여부

    Returns:
        selected, n_pairings, total_cost, coverage, status,
        uncoverable, deadhead_count, deadhead_flights
    """
    if not pairings:
        return {
            "selected": [], "n_pairings": 0, "total_cost": 0.0,
            "coverage": 0.0, "status": "no_pairings", "uncoverable": n_flights,
            "deadhead_count": 0, "deadhead_flights": [],
        }

    flight_to_pairings: Dict[int, List[int]] = defaultdict(list)
    for j, p in enumerate(pairings):
        for leg in p["legs"]:
            flight_to_pairings[leg].append(j)

    covered_flights = set(flight_to_pairings.keys())
    uncoverable     = set(range(n_flights)) - covered_flights

    M = len(pairings)
    prob = pulp.LpProblem("crew_pairing_sc", pulp.LpMinimize)

    x = [pulp.LpVariable(f"x_{j}", cat="Binary") for j in range(M)]
    d = {i: pulp.LpVariable(f"d_{i}", lowBound=0) for i in covered_flights}

    prob += (
        pulp.lpSum(pairings[j]["cost"] * x[j] for j in range(M))
        + lambda_dh * pulp.lpSum(d.values())
    )

    for i in covered_flights:
        cover_sum = pulp.lpSum(x[j] for j in flight_to_pairings[i])
        prob += (cover_sum >= 1,          f"cover_{i}")
        prob += (d[i] >= cover_sum - 1,   f"dh_{i}")

    if use_gurobi:
        try:
            solver = pulp.GUROBI(timeLimit=time_limit, msg=int(verbose))
        except Exception:
            print("[warn] Gurobi 사용 불가 → CBC로 대체")
            solver = pulp.PULP_CBC_CMD(timeLimit=time_limit, msg=int(verbose))
    else:
        solver = pulp.PULP_CBC_CMD(timeLimit=time_limit, msg=int(verbose))

    prob.solve(solver)

    selected = [pairings[j] for j in range(M) if (x[j].value() or 0) > 0.5]

    covered_legs = set()
    for p in selected:
        covered_legs.update(p["legs"])
    covered_count = len(covered_legs & set(range(n_flights)))

    deadhead_flights = [
        i for i in covered_flights if (d[i].value() or 0) > 0.5
    ]

    total_legs = sum(len(p["legs"]) for p in selected)
    avg_legs   = total_legs / len(selected) if selected else 0.0

    return {
        "selected":            selected,
        "n_pairings":          len(selected),
        "total_cost":          sum(p["cost"] for p in selected),
        "mip_obj":             pulp.value(prob.objective),
        "coverage":            covered_count / n_flights if n_flights > 0 else 0.0,
        "status":              pulp.LpStatus[prob.status],
        "uncoverable":         len(uncoverable),
        "deadhead_count":      len(deadhead_flights),
        "deadhead_flights":    deadhead_flights,
        "avg_legs_per_pairing": avg_legs,
    }
