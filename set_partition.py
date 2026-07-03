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

    pi: Dict[int, float] = {}
    for i in covered_flights:
        pi[i] = prob.constraints[f"cover_{i}"].pi or 0.0

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
    keep_ratio: float = 2.0,
    per_flight_keep: int = 3,
) -> List[Dict]:
    M = len(pairings)
    if M == 0:
        return []

    n_legs = [p.get("n_legs", len(p["legs"])) for p in pairings]

    covered_flights: set = set()
    for p in pairings:
        covered_flights.update(p["legs"])

    target = int(max(keep_ratio * len(covered_flights), 1))
    if target >= M:
        return list(pairings)

    # 0) fallback 1-leg (is_deadhead=True, len==1)는 무조건 보존 — Set Partitioning feasibility 보장
    kept = {j for j, p in enumerate(pairings)
            if p.get("is_deadhead") and len(p["legs"]) == 1}

    # 1) 전역 상위 K: rc 오름차순, 동률이면 긴 페어링 우선
    order = sorted(range(M), key=lambda j: (reduced_costs[j], -n_legs[j]))
    kept.update(order[:target])

    # 2) flight별 커버리지 보장 (긴 페어링 우선)
    flight_to_pairings: Dict[int, List[int]] = defaultdict(list)
    for j, p in enumerate(pairings):
        for leg in p["legs"]:
            flight_to_pairings[leg].append(j)

    for js in flight_to_pairings.values():
        if sum(1 for j in js if j in kept) >= per_flight_keep:
            continue
        ranked = sorted(js, key=lambda j: (reduced_costs[j], -n_legs[j]))
        for j in ranked[:per_flight_keep]:
            kept.add(j)

    return [pairings[j] for j in sorted(kept)]


def solve_set_covering(
    pairings: List[Dict],
    n_flights: int,
    time_limit: int  = 300,
    use_gurobi: bool = False,
    verbose: bool    = False,
) -> Dict:
    """
    Set Partitioning IP를 풀어 최적 pairing subset 선택

    Set Partitioning(==1): 각 flight는 정확히 1개 pairing에만 포함
    - covering(>=1) 대비 IP가 짧은 것 여러 개로 타일링하는 통로 차단 → 긴 페어링 선호
    - feasibility는 collect_pool_full의 fallback(cost=200) 1-leg가 보장

    Args:
        pairings:   pairing dict 리스트 (legs, cost 필드 필요)
        n_flights:  전체 flight 수 (flight ID: 0 ~ n_flights-1)
        time_limit: solver 제한 시간 (초)
        use_gurobi: True면 Gurobi 사용, 실패 시 CBC로 fallback
        verbose:    solver 로그 출력 여부
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

    pre_selected_idx: set = set()
    pre_excluded_flights: set = set()   # 이미 처리된 flight (IP 제약에서 제외)

    changed = True
    while changed:
        changed = False
        for i in covered_flights - pre_excluded_flights:
            active = [j for j in flight_to_pairings[i] if j not in pre_selected_idx
                      and not any(leg in pre_excluded_flights for leg in pairings[j]["legs"]
                                  if leg != i)]
            # 유효 후보가 1개 → 강제 선택
            if len(active) == 1:
                j = active[0]
                pre_selected_idx.add(j)
                for leg in pairings[j]["legs"]:
                    pre_excluded_flights.add(leg)
                changed = True

    pre_selected = [pairings[j] for j in pre_selected_idx]
    remaining_flights = covered_flights - pre_excluded_flights
    remaining_pairing_idx = [
        j for j in range(len(pairings))
        if j not in pre_selected_idx
        and not any(leg in pre_excluded_flights for leg in pairings[j]["legs"])
    ]
    remaining_pairings = [pairings[j] for j in remaining_pairing_idx]

    print(f"  [unit-prop] 확정 {len(pre_selected)}개 pairing ({len(pre_excluded_flights)}편) "
          f"→ IP 잔여 {len(remaining_pairings)}개 pairing / {len(remaining_flights)}편", flush=True)

    # ── IP: 잔여 pairings만 풀기 ────────────────────────────────────────────────
    ip_selected = []
    ip_status   = "Optimal"

    if remaining_pairings and remaining_flights:
        r_flight_to_pairings: Dict[int, List[int]] = defaultdict(list)
        for rj, p in enumerate(remaining_pairings):
            for leg in p["legs"]:
                if leg in remaining_flights:
                    r_flight_to_pairings[leg].append(rj)

        R = len(remaining_pairings)
        prob = pulp.LpProblem("crew_pairing_sp", pulp.LpMinimize)
        x = [pulp.LpVariable(f"x_{rj}", cat="Binary") for rj in range(R)]
        prob += pulp.lpSum(remaining_pairings[rj]["cost"] * x[rj] for rj in range(R))

        for i in remaining_flights:
            if r_flight_to_pairings[i]:
                cover_sum = pulp.lpSum(x[rj] for rj in r_flight_to_pairings[i])
                prob += (cover_sum == 1, f"cover_{i}")

        if use_gurobi:
            try:
                solver = pulp.GUROBI(timeLimit=time_limit, msg=int(verbose))
            except Exception:
                print("[warn] Gurobi 사용 불가 → CBC로 대체")
                solver = pulp.PULP_CBC_CMD(timeLimit=time_limit, msg=int(verbose))
        else:
            solver = pulp.PULP_CBC_CMD(timeLimit=time_limit, msg=int(verbose))

        prob.solve(solver)
        ip_status   = pulp.LpStatus[prob.status]
        ip_selected = [remaining_pairings[rj] for rj in range(R) if (x[rj].value() or 0) > 0.5]

    selected = pre_selected + ip_selected

    # IP Not Solved / Infeasible 시 미커버 편에 fallback 1-leg 강제 배정 → coverage 100% 보장
    if ip_status not in ("Optimal",):
        covered_so_far = set()
        for p in selected:
            covered_so_far.update(p["legs"])
        fallback_map = {
            p["legs"][0]: p for p in pairings
            if p.get("is_deadhead") and len(p["legs"]) == 1
        }
        for fid in covered_flights - covered_so_far:
            if fid in fallback_map:
                selected.append(fallback_map[fid])

    covered_legs = set()
    for p in selected:
        covered_legs.update(p["legs"])
    covered_count = len(covered_legs & set(range(n_flights)))

    # fallback(cost=200, is_deadhead=True) 선택된 것 = 미커버 deadhead
    fallback_flights = [
        p["legs"][0] for p in selected
        if p.get("is_deadhead") and len(p["legs"]) == 1
    ]

    total_legs = sum(len(p["legs"]) for p in selected)
    avg_legs   = total_legs / len(selected) if selected else 0.0

    return {
        "selected":             selected,
        "n_pairings":           len(selected),
        "total_cost":           sum(p["cost"] for p in selected),
        "coverage":             covered_count / n_flights if n_flights > 0 else 0.0,
        "status":               ip_status,
        "uncoverable":          len(uncoverable),
        "deadhead_count":       len(fallback_flights),
        "deadhead_flights":     fallback_flights,
        "avg_legs_per_pairing": avg_legs,
    }
