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

INF = float("inf")


def build_base_reach(flights, base_ap, constraint):
    """flight 도착 지점에서 base_ap까지의 복귀 비용 하한 2종.

    반환: {"time": {id: 최소 추가 경과시간}, "hops": {id: 최소 추가 leg 수}}

    time 하한만으로는 pruning이 거의 안 걸린다 — max_pairing_days가 5일이라 웬만한
    경로는 시간 예산 안에 들어오기 때문. 실제로 policy를 base 근처에 묶어두는 것은
    hops 하한이다 (max_legs=8이라 훨씬 빡빡함). 둘 다 하한이므로 admissible.

    하한을 보장하기 위해 연결 규칙을 가장 느슨하게 잡는다:
      - 하한: gap >= min_conn 만 요구하고 상한(max_conn)은 두지 않는다.
        큰 gap은 END_DUTY(overnight rest)로 합법화될 수 있으므로 max_conn으로
        자르면 실현 가능한 복귀 경로를 놓칠 수 있다.
      - duty 시간/leg 수 제약도 무시한다. 모두 D를 키우는 방향이므로 생략해야 하한이 된다.
      - 이미 배정된 flight도 경로에 허용한다(assigned 무시). 역시 낙관적 = 하한.

    계산: dep_time 내림차순 1-pass DP. 후속편 g는 dep_g >= arr_f + min_conn > dep_f 이므로
    내림차순에서 항상 f보다 먼저 처리된다 → O(N · 평균 out-degree).

    Args:
        flights:    flight dict 리스트 (키: id, origin, dest, dep_time, arr_time)
        base_ap:    복귀 목표 base 공항 ID
        constraint: min_conn을 읽음
    Returns:
        dict: flight id -> 복귀 소요시간 하한 (또는 INF)
    """
    min_conn = constraint.get("min_conn", 0.65)

    by_origin = {}
    for f in flights:
        by_origin.setdefault(f["origin"], []).append(f)

    D, H = {}, {}
    for f in sorted(flights, key=lambda x: -x["dep_time"]):
        if f["dest"] == base_ap:
            D[f["id"]], H[f["id"]] = 0.0, 0
            continue

        best_t, best_h = INF, INF
        arr = f["arr_time"]
        for g in by_origin.get(f["dest"], ()):
            gap = g["dep_time"] - arr
            if gap < min_conn:
                continue
            d_next = D.get(g["id"])
            if d_next is None or d_next == INF:
                continue
            cand_t = gap + (g["arr_time"] - g["dep_time"]) + d_next
            if cand_t < best_t:
                best_t = cand_t
            cand_h = 1 + H[g["id"]]
            if cand_h < best_h:
                best_h = cand_h
        D[f["id"]], H[f["id"]] = best_t, best_h

    return {"time": D, "hops": H}


def can_reach_base(reach, flight, pairing_start_time, max_pairing_days,
                   legs_after=None, max_legs=None):
    """flight를 선택해도 base 복귀 여지가 남는가 (하한 기준).

    reach가 None이면(계산 안 됨) 항상 True — 기존 동작 유지.

    legs_after/max_legs가 주어지면 leg 예산도 검사한다:
        (이 flight까지 쓴 leg 수) + (base까지 최소 추가 leg 수) <= max_legs
    """
    if reach is None:
        return True
    fid = flight["id"]
    d = reach["time"].get(fid, INF)
    if d == INF:
        return False
    if (flight["arr_time"] + d - pairing_start_time) / 24.0 > max_pairing_days:
        return False
    if legs_after is not None and max_legs is not None:
        if legs_after + reach["hops"].get(fid, INF) > max_legs:
            return False
    return True
