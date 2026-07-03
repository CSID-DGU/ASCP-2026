import numpy as np
import config

# flight dict 키: "origin", "dest", "dep_time", "arr_time", "id" 


def get_max_duty(legs_count, custom_max_duty=None):
    """constraint["max_duty"]를 duty 한도로 사용. None이면 FAA_DUTY_TABLE 기본값 적용.

    legs_count: 해당 duty에 추가될 leg 수 (현재 legs + 1)
    custom_max_duty: constraint["max_duty"] 값. None이면 DEFAULT_CONSTRAINTS 사용

    [수정] 기존: min(faa_limit, custom_max_duty) — FAA legs 기반 한도와 constraint를 동시에 적용.
           문제: Stage 3에서 max_duty=14h 샘플링 시 legs=3이면 실제 마스킹이 min(12,14)=12h로
                 적용되어 FiLM 입력(14h 기준)과 environment 마스킹 불일치 → 학습 신호 오염.
           수정: custom_max_duty를 그대로 사용. constraint 값이 유일한 기준.
                 FAA 규정 준수는 constraint 설정 단계(config.py)에서 보장.
    """
    if custom_max_duty is not None:
        return custom_max_duty
    return config.FAA_DUTY_TABLE.get(min(legs_count, 6), 10.0)


def get_mask(state, flights, assigned, constraint=None, stage=3):
    """현재 state에서 선택 가능한 action 마스크 반환

    반환: 크기 [N + 2]의 list (0: 불가, 1: 가능)
         index 0..N-1 = flight, N = END_DUTY, N+1 = END_PAIRING
    """
    c = constraint if constraint else config.DEFAULT_CONSTRAINTS
    stage_rule = config.CURRICULUM_CONFIG.get(stage, config.CURRICULUM_CONFIG[3])

    N = len(flights)
    mask = np.zeros(N + 2, dtype=np.int32)

    pairing_start    = state.get("pairing_start", False)
    is_resting       = state.get("is_resting", False)
    rest_end         = state.get("rest_end_time", 0.0)
    duty_period      = state.get("duty_period", 0)
    duty_start_time  = state.get("duty_start_time", state["current_time"])
    pairing_start_time = state.get("pairing_start_time", state["current_time"])

    # pairing 첫 leg: base-origin 미배정 편이 남아있으면 base 출발 강제
    # base-origin 소진 시 origin 제한 해제 → deadhead loop 방지
    # HB1/HB2 비대칭 허용: base_ids가 있으면 그 중 아무 base 출발이든 인정 (log/0703/base.md)
    # base_remaining은 후보 flight f에 의존하지 않으므로(loop-invariant) 루프 밖에서 1회만 계산 —
    # 루프 안에서 매번 재계산하면 O(N^2)이 되어 저연결 base(HB2)에서 episode당 수십 초까지
    # 느려짐 (log/0704 turkish 스모크 테스트에서 발견)
    base_ap = c.get("base_airport", config.DEFAULT_CONSTRAINTS["base_airport"])
    base_id_set = set(c.get("base_ids") or [base_ap])
    if pairing_start:
        base_remaining = any(
            not assigned[fl["id"]] and fl["origin"] in base_id_set
            for fl in flights
        )

    for i, f in enumerate(flights):
        if assigned[f["id"]]:
            continue

        valid = True

        # 1. 공항 연결 검사
        if pairing_start:
            if base_remaining and f["origin"] not in base_id_set:
                valid = False
        elif f["origin"] != state["current_airport"]:
            valid = False

        # 2. 시간 연결 검사
        if is_resting:
            # rest 미종료 시 탑승 불가
            if f["dep_time"] < rest_end:
                valid = False
        else:
            if not pairing_start:
                gap = f["dep_time"] - state["current_time"]
                if gap < c.get("min_conn", config.DEFAULT_CONSTRAINTS["min_conn"]) or \
                   gap > c.get("max_conn", config.DEFAULT_CONSTRAINTS["max_conn"]):
                    valid = False

        # 3. Duty 시간 제약 (FAA Part 117)
        # duty window = duty 시작 ~ 해당 flight 도착까지 전체 경과 시간
        # pairing 첫 편이거나 rest 직후 새 duty 시작이면 기준점을 해당 편 출발로 리셋
        legs_after = state.get("legs", 0) + 1
        effective_max_duty = get_max_duty(legs_after, c.get("max_duty"))
        current_duty_start = f["dep_time"] if (pairing_start or is_resting) else duty_start_time
        total_duty_window = f["arr_time"] - current_duty_start
        if total_duty_window > effective_max_duty:
            valid = False
        if legs_after > c.get("max_legs", config.DEFAULT_CONSTRAINTS["max_legs"]):
            valid = False

        # 4. Pairing 기간 제약
        elapsed_days = (f["arr_time"] - pairing_start_time) / 24.0
        if elapsed_days > c.get("max_pairing_days", config.DEFAULT_CONSTRAINTS["max_pairing_days"]):
            valid = False

        if valid:
            mask[i] = 1

    # END_DUTY (mask[-2] = mask[N])
    can_end_duty = (
        stage_rule["allow_end_duty"]
        and state.get("legs", 0) > 0
        and not is_resting
        and not pairing_start
        and duty_period < c.get("max_duty_periods", config.DEFAULT_CONSTRAINTS["max_duty_periods"])  # overnight 횟수 기준
    )
    if can_end_duty:
        mask[config.END_DUTY] = 1

    # END_PAIRING (mask[-1] = mask[N+1])
    # min_pairing_legs: 항공사별 설정 (Delta/Alaska/JetBlue=3, Turkish=2)
    pairing_elapsed_days = (state["current_time"] - pairing_start_time) / 24.0
    min_pairing_legs = c.get("min_pairing_legs", 2)
    can_end_pairing = (
        state.get("total_legs", 0) >= min_pairing_legs
        and pairing_elapsed_days <= c.get("max_pairing_days", config.DEFAULT_CONSTRAINTS["max_pairing_days"])
    )
    # base 미복귀 시 BASE_PENALTY는 step()에서 reward로 처리 (hard mask 제거 → soft penalty)
    if can_end_pairing:
        mask[config.END_PAIRING] = 1

    return mask.tolist()


def step(state, action, flights, assigned, constraint=None):
    """action을 받아 next_state, reward, done 반환

    action 범위:
      0 .. N-1  : flight 선택
      N         : END_DUTY
      N+1       : END_PAIRING
    done=True 조건: END_PAIRING 선택 시 모든 flight가 커버된 경우
    """
    c = constraint if constraint else config.DEFAULT_CONSTRAINTS
    N = len(flights)

    # END_DUTY → rest 진입, pairing 계속
    if action == N:
        min_rest = c.get("min_rest", config.DEFAULT_CONSTRAINTS["min_rest"])
        next_state = {
            **state,
            "duty_time":       0.0,
            "duty_start_time": state["current_time"] + min_rest,
            "legs":            0,
            "is_resting":      True,
            "rest_end_time":   state["current_time"] + min_rest,
            "duty_period":     state.get("duty_period", 0) + 1,
            "pairing_start":   False,
        }
        # v8: overnight rest 직접 장려 — multi-day pairing 유도
        # overnight 10h는 dead_time에서 제외되므로 legs↑ dead_time 증가 없이 avg_legs 개선 가능
        # v16: duty당 leg 수가 MIN_LEGS_FOR_DUTY_BONUS 미만이면 보너스를 비례 축소 —
        # END_DUTY가 무위험 고정보상이라 duty를 짧게 끝내고 반복 수령하는 유인을 제거
        duty_legs = state.get("legs", 0)
        min_legs = config.MIN_LEGS_FOR_DUTY_BONUS
        scale = min(1.0, duty_legs / min_legs) if min_legs > 0 else 1.0
        return next_state, config.END_DUTY_BONUS * scale, False

    # END_PAIRING → pairing 비용 부과 후 새 pairing 시작 (또는 에피소드 종료)
    if action == N + 1:
        p_cost = c.get("pairing_cost", config.DEFAULT_CONSTRAINTS["pairing_cost"])
        base_penalty = c.get("base_penalty", config.DEFAULT_CONSTRAINTS["base_penalty"])
        # constraint["base_airport"] 에피소드별 주입
        base = c.get("base_airport", config.DEFAULT_CONSTRAINTS["base_airport"])
        # HB1/HB2 비대칭 허용: base_ids가 있으면 그 중 아무 base 복귀든 무패널티 (log/0703/base.md)
        base_id_set = set(c.get("base_ids") or [base])

        total_legs = state.get("total_legs", 0)
        reward = -p_cost + total_legs * config.LEG_PER_PAIRING_BONUS

        if total_legs < config.MIN_LEGS_FOR_PAIRING:
            reward += config.MIN_LEGS_PENALTY
        if state["current_airport"] not in base_id_set:
            reward -= base_penalty

        unassigned = [f for f in flights if not assigned[f["id"]]]
        if not unassigned:
            # 모든 flight 커버 완료 → 에피소드 종료
            return state, reward, True
        # 미배정 flight 남아있음 → 새 pairing 시작
        # 방금 도착한 위치가 base_ids 중 하나면 그 위치에서, 아니면(base 미복귀) 가장 가까운
        # base로 재배치해 새 pairing 시작 — base_ids 미지정 시 기존과 동일(단일 base로 텔레포트)
        restart_base = state["current_airport"] if state["current_airport"] in base_id_set else base
        base_unassigned = [f for f in unassigned if f["origin"] == restart_base]
        next_time = min(f["dep_time"] for f in base_unassigned) if base_unassigned else min(f["dep_time"] for f in unassigned)
        next_state = {
            **state,
            "current_airport":    restart_base,
            "current_time":       next_time,
            "duty_time":          0.0,
            "duty_start_time":    next_time,
            "legs":               0,
            "total_legs":         0,    # 새 pairing 시작 시 리셋
            "duty_period":        0,
            "is_resting":         False,
            "rest_end_time":      None,
            "pairing_start":      True,
            "pairing_start_time": next_time,
        }
        return next_state, reward, False

    # Flight 선택
    f = flights[action]
    assigned[f["id"]] = True
    flight_time = f["arr_time"] - f["dep_time"]

    # pairing 첫 편이면 pairing_start_time 리셋, rest 직후 새 duty면 duty_start_time 리셋
    p_start_time = f["dep_time"] if state.get("pairing_start", False) else state["pairing_start_time"]
    d_start_time = f["dep_time"] if (state.get("pairing_start", False) or state.get("is_resting", False)) \
                   else state["duty_start_time"]

    next_state = {
        "current_airport":    f["dest"],
        "current_time":       f["arr_time"],
        "duty_time":          (0.0 if state.get("is_resting", False) else state["duty_time"]) + flight_time,
        "duty_start_time":    d_start_time,
        "legs":               state.get("legs", 0) + 1,
        "total_legs":         state.get("total_legs", 0) + 1,  # pairing 전체 누적 (END_DUTY에서 리셋 안 함)
        "remaining":          state["remaining"] - 1,
        "pairing_start":      False,
        "duty_period":        state.get("duty_period", 0),
        "pairing_start_time": p_start_time,
        "is_resting":         False,
        "rest_end_time":      None,
    }

    # dead time reward: duty 중간 flight 간 대기 시간 패널티 + 연결 보너스
    # pairing 첫 편이거나 rest 직후 첫 편은 gap 패널티 없음
    # LEG_PER_PAIRING_BONUS: 모든 flight 선택 시 즉시 지급 (END_PAIRING 지연 지급 제거)
    # → credit assignment 문제 해결: agent가 multi-leg 가치를 즉각 인식
    if not state.get("pairing_start", False) and not state.get("is_resting", False):
        reward = -(f["dep_time"] - state["current_time"]) + config.LEG_CONN_BONUS + config.LEG_PER_PAIRING_BONUS
    else:
        reward = config.LEG_PER_PAIRING_BONUS  # 첫 편 / rest 직후: gap 패널티 없이 bonus만

    return next_state, reward, False


def final_reward(assigned, custom_penalty=None):
    """에피소드 종료 시 미배정 flight에 대한 패널티 반환"""
    penalty = custom_penalty if custom_penalty is not None else config.DEFAULT_CONSTRAINTS["uncovered_penalty"]
    remaining = sum(1 for v in assigned.values() if not v)
    return -penalty * remaining
