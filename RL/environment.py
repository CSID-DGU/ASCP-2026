END_DUTY = -2     # 현재 duty 종료 → rest period 진입, pairing 계속
END_PAIRING = -1  # pairing 전체 종료


def get_max_duty(legs_after, constraint):
    """FAA Part 117 — duty에 추가될 leg 수 기준 max flight duty period (hours)"""
    table = {1: 13.0, 2: 13.0, 3: 12.0, 4: 11.5, 5: 11.0, 6: 10.5}
    return table.get(min(legs_after, 6), 10.0)


def get_mask(state, flights, assigned, constraint):
    """
    반환: [flight_0, ..., flight_N-1, END_DUTY, END_PAIRING]
    인덱스: 0..N-1 = flight, N = END_DUTY, N+1 = END_PAIRING
    """
    mask = []
    pairing_start = state.get("pairing_start", False)
    is_resting = state.get("is_resting", False)
    rest_end = state.get("rest_end_time")
    duty_period = state.get("duty_period", 0)
    max_duty_periods = constraint.get("max_duty_periods", 4)
    max_pairing_days = constraint.get("max_pairing_days", 5)
    pairing_start_time = state.get("pairing_start_time", state["current_time"])

    for f in flights:
        if assigned[f["id"]]:
            mask.append(0)
            continue

        flight_time = f["arr_time"] - f["dep_time"]
        gap = f["dep_time"] - state["current_time"]

        valid = True

        # rest 중이면 rest_end_time 이후 출발 flight만 허용
        if is_resting:
            if rest_end is not None and f["dep_time"] < rest_end:
                valid = False

        legs_after = state.get("legs", 0) + 1
        effective_max_duty = get_max_duty(legs_after, constraint)

        if pairing_start:
            if state.get("duty_time", 0.0) + flight_time > effective_max_duty:
                valid = False
            if legs_after > constraint.get("max_legs", 999):
                valid = False
        else:
            if not is_resting:
                if f["origin"] != state["current_airport"]:
                    valid = False
                if gap < constraint["min_conn"] or gap > constraint["max_conn"]:
                    valid = False
                if state["duty_time"] + flight_time > effective_max_duty:
                    valid = False
                if legs_after > constraint.get("max_legs", 999):
                    valid = False
            else:
                # rest 종료 후 새 duty 시작: 공항 연결 체크, connection 제약은 없음
                if f["origin"] != state["current_airport"]:
                    valid = False

        # pairing 기간 초과 방지
        elapsed_days = (f["dep_time"] - pairing_start_time) / 24.0
        if elapsed_days > max_pairing_days:
            valid = False

        mask.append(1 if valid else 0)

    # END_DUTY: 현재 duty에 leg가 있고, 추가 duty period 여유가 있고, rest 중 아닐 때
    can_end_duty = (
        state.get("legs", 0) > 0
        and not is_resting
        and not pairing_start
        and duty_period < max_duty_periods - 1
    )
    mask.append(1 if can_end_duty else 0)

    # END_PAIRING: base에 있고 leg가 있고 pairing 기간 내
    pairing_elapsed_days = (state["current_time"] - pairing_start_time) / 24.0
    can_end_pairing = (
        state["current_airport"] == constraint["base_airport"]
        and state.get("legs", 0) > 0
        and pairing_elapsed_days <= max_pairing_days
    )
    mask.append(1 if can_end_pairing else 0)

    return mask


def step(state, action, flights, assigned, constraint):
    f = flights[action]
    assigned[f["id"]] = True

    flight_time = f["arr_time"] - f["dep_time"]

    next_state = {
        "current_airport":    f["dest"],
        "current_time":       f["arr_time"],
        "duty_time":          state["duty_time"] + flight_time,
        "duty_start_time":    state["duty_start_time"],
        "legs":               state.get("legs", 0) + 1,
        "remaining":          state["remaining"] - 1,
        "pairing_start":      False,
        # multi-day 필드 전파
        "duty_period":        state.get("duty_period", 0),
        "pairing_start_time": state.get("pairing_start_time", state["current_time"]),
        "is_resting":         False,
        "rest_end_time":      None,
    }

    # dead time: duty 내 flight 간 연결 대기 시간 (pairing 첫 flight나 rest 직후는 제외)
    if not state.get("pairing_start", False) and not state.get("is_resting", False):
        reward = -(f["dep_time"] - state["current_time"])
    else:
        reward = 0.0

    return next_state, reward, False


def step_end_duty(state, constraint):
    """현재 duty 종료 → rest period 진입, pairing은 계속"""
    min_rest = constraint.get("min_rest", 9.5)
    return {
        **state,
        "duty_time":       0.0,
        "duty_start_time": state["current_time"] + min_rest,
        "legs":            0,
        "is_resting":      True,
        "rest_end_time":   state["current_time"] + min_rest,
        "duty_period":     state.get("duty_period", 0) + 1,
        "pairing_start":   False,
    }


def final_reward(assigned, uncovered_penalty=10.0):
    remaining = sum(1 for v in assigned.values() if not v)
    return -uncovered_penalty * remaining
