# base_reach.py — pairing이 base로 복귀 가능한지 판정하는 backward reachability DP
#
# 배경: END_PAIRING에 "current_airport == base" hard mask만 되살리면 데드락이 난다 —
# base가 아닌 공항에서 갈 수 있는 flight도 없어지면 all-zero mask가 되고,
# rollout.py가 강제 flush해서 결국 base 미복귀 pairing이 그대로 나온다.
#
# 해결: 각 flight 도착 지점에서 base까지 "복귀에 필요한 최소 추가 경과시간의 하한"을
# 미리 계산해두고, 그 하한으로도 max_pairing_days를 넘기는 leg는 애초에 마스킹한다.
# 하한(admissible)이므로 실현 가능한 경로를 잘라내는 일이 없고, 이는 SPPRC의
# resource labeling(도달 불가 label 가지치기)과 동일한 논리다.
#
# [2026-07-29 수정] leg 예산 하한을 "duty당 max_legs"가 아니라 "duty_crossings당
# max_duty_periods"로 바꿨다. 이전 버전은 base까지 필요한 raw hop 수(H)를 현재 duty의
# state["legs"]와 더해 max_legs와 비교했는데, H는 duty/rest 제약을 무시하고 계산되므로
# 그 안에 "미래의 다른 duty"에서 소진될 hop이 섞여 있다 — END_DUTY를 거치면 leg 카운터가
# 0으로 리셋되는데 그 리셋을 모르는 채로 누적 hop을 현재 duty 예산과 비교한 것이다.
# 실제로 base 도달을 막는 진짜 자원은 "duty당 leg 수"가 아니라 "남은 duty_periods
# (overnight rest 허용 횟수)"다 — leg가 부족해지면 언제든 END_DUTY로 새 예산을 받을 수
# 있기 때문(단, duty_periods가 남아있고 나중에 출발하는 flight이 있어야 함, 후자는
# time 하한(D)이 이미 보장). 그래서 hop 대신 "필요한 추가 duty 경계 교차 횟수(C)"를
# 계산해 duty_period(지금까지 쓴 rest 횟수) + C <= max_duty_periods로 비교한다.

INF = float("inf")


def build_base_reach(flights, base_ap, constraint):
    """flight 도착 지점에서 base_ap까지의 복귀 비용 하한 2종.

    반환: {"time": {id: 최소 추가 경과시간}, "duty_crossings": {id: 최소 추가 duty 경계 교차 횟수}}

    time 하한만으로는 pruning이 거의 안 걸린다 — max_pairing_days가 5일이라 웬만한
    경로는 시간 예산 안에 들어오기 때문. 실제로 policy를 base 근처에 묶어두는 것은
    duty_crossings 하한이다(max_duty_periods가 2~4로 훨씬 빡빡함). 둘 다 하한이므로 admissible.

    하한을 보장하기 위해 연결 규칙을 가장 느슨하게 잡는다:
      - 하한: gap >= min_conn 만 요구하고 상한(max_conn)은 두지 않는다.
        큰 gap은 END_DUTY(overnight rest)로 합법화될 수 있으므로 max_conn으로
        자르면 실현 가능한 복귀 경로를 놓칠 수 있다. 다만 gap이 max_conn을 넘으면
        그 연결은 "duty 경계 교차"(rest 필요)로 집계한다 — 같은 duty 안에서는
        min_conn~max_conn 연결만 유효하기 때문.
      - duty 시간/leg 수 제약(max_duty, max_legs)은 무시한다. 모두 D를 키우는
        방향이므로 생략해야 하한이 된다. duty_crossings만 정확히 추적한다 —
        이게 실제로 END_DUTY 횟수(max_duty_periods)라는 진짜 자원과 대응되기 때문.
      - 이미 배정된 flight도 경로에 허용한다(assigned 무시). 역시 낙관적 = 하한.

    계산: dep_time 내림차순 1-pass DP. 후속편 g는 dep_g >= arr_f + min_conn > dep_f 이므로
    내림차순에서 항상 f보다 먼저 처리된다 → O(N · 평균 out-degree).

    Args:
        flights:    flight dict 리스트 (키: id, origin, dest, dep_time, arr_time)
        base_ap:    복귀 목표 base 공항 ID
        constraint: min_conn/max_conn을 읽음
    Returns:
        dict: {"time": {...}, "duty_crossings": {...}}
    """
    min_conn = constraint.get("min_conn", 0.65)
    max_conn = constraint.get("max_conn", 9.0)
    min_rest = constraint.get("min_rest", 10.0)

    by_origin = {}
    for f in flights:
        by_origin.setdefault(f["origin"], []).append(f)

    D, C = {}, {}
    for f in sorted(flights, key=lambda x: -x["dep_time"]):
        if f["dest"] == base_ap:
            D[f["id"]], C[f["id"]] = 0.0, 0
            continue

        best_t, best_c = INF, INF
        arr = f["arr_time"]
        for g in by_origin.get(f["dest"], ()):
            gap = g["dep_time"] - arr
            if gap < min_conn:
                continue
            # gap이 max_conn을 넘는데 min_rest에도 못 미치면 같은 duty로도, rest로도
            # legal하게 못 쓰는 gap이다 — 이 연결 자체를 하한 계산에서 제외해야
            # duty_crossings가 실제로 불가능한 경로를 "가능하다"고 낙관하지 않는다(2026-07-29).
            if gap > max_conn and gap < min_rest:
                continue
            d_next = D.get(g["id"])
            if d_next is None or d_next == INF:
                continue
            cand_t = gap + (g["arr_time"] - g["dep_time"]) + d_next
            if cand_t < best_t:
                best_t = cand_t
            c_next = C[g["id"]]
            cand_c = c_next + (1 if gap > max_conn else 0)
            if cand_c < best_c:
                best_c = cand_c
        D[f["id"]], C[f["id"]] = best_t, best_c

    return {"time": D, "duty_crossings": C}


def can_reach_base(reach, flight, pairing_start_time, max_pairing_days,
                   duty_period=None, max_duty_periods=None):
    """flight를 선택해도 base 복귀 여지가 남는가 (하한 기준).

    reach가 None이면(계산 안 됨) 항상 True — 기존 동작 유지.

    duty_period/max_duty_periods가 주어지면 rest 횟수 예산도 검사한다:
        (지금까지 쓴 duty_period) + (base까지 필요한 최소 추가 duty 경계 교차 수) <= max_duty_periods
    duty당 leg 수(max_legs)는 여기서 검사하지 않는다 — END_DUTY로 언제든 새 예산을
    받을 수 있어서 실제 병목이 아니다(기존 environment.py의 매 스텝 max_legs 체크가
    현재 duty 안에서는 이미 정확하게 걸러준다).
    """
    if reach is None:
        return True
    fid = flight["id"]
    d = reach["time"].get(fid, INF)
    if d == INF:
        return False
    if (flight["arr_time"] + d - pairing_start_time) / 24.0 > max_pairing_days:
        return False
    if duty_period is not None and max_duty_periods is not None:
        c = reach["duty_crossings"].get(fid, INF)
        if duty_period + c > max_duty_periods:
            return False
    return True
