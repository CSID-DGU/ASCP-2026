import numpy as np
import torch
import config
from base_reach import can_reach_any_base

INF = float("inf")

# flight dict keys: "origin", "dest", "dep_time", "arr_time", "id"


def get_max_duty(legs_count, custom_max_duty=None):
    """Use constraint["max_duty"] as the duty limit. Falls back to FAA_DUTY_TABLE if None.

    legs_count: number of legs this duty will have after adding this one (current legs + 1)
    custom_max_duty: the constraint["max_duty"] value. Uses DEFAULT_CONSTRAINTS if None.

    custom_max_duty, when given, is used directly as the duty limit -- the constraint value is
    the single source of truth here, so masking always matches the value used for the FiLM
    input. FAA-compliant limits are guaranteed upstream, at constraint-configuration time
    (config.py).
    """
    if custom_max_duty is not None:
        return custom_max_duty
    return config.FAA_DUTY_TABLE.get(min(legs_count, 6), 10.0)


def get_mask(state, flights, assigned, constraint=None, stage=3):
    """Return the action mask of choices available from the current state.

    Returns: a list of size [N + 2] (0: not allowed, 1: allowed)
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

    # 첫 flight는 episode base에서 시작하되 HB1/HB2 중 어느 home base로든 복귀 가능함.
    base_ap = c["base_airport"]
    base_id_set = set(c.get("base_ids") or [base_ap])
    base_reaches = c.get("_base_reaches")
    if base_reaches is None and c.get("_base_reach") is not None:
        base_reaches = {base_ap: c["_base_reach"]}
    if not base_reaches:
        raise ValueError("Turkish CPP constraint에는 _base_reaches가 필요합니다.")
    max_pd           = c.get("max_pairing_days", config.DEFAULT_CONSTRAINTS["max_pairing_days"])
    max_duty_periods = c.get("max_duty_periods", config.DEFAULT_CONSTRAINTS["max_duty_periods"])
    for i, f in enumerate(flights):
        if assigned[f["id"]]:
            continue

        valid = True

        # 1. Airport connectivity check
        if pairing_start:
            if f["origin"] != base_ap:
                valid = False
        elif f["origin"] != state["current_airport"]:
            valid = False

        # 2. Time connectivity check
        if is_resting:
            # Cannot board while rest is still in progress
            if f["dep_time"] < rest_end:
                valid = False
        else:
            if not pairing_start:
                gap = f["dep_time"] - state["current_time"]
                if gap < c.get("min_conn", config.DEFAULT_CONSTRAINTS["min_conn"]) or \
                   gap > c.get("max_conn", config.DEFAULT_CONSTRAINTS["max_conn"]):
                    valid = False

        # 3. Duty time constraint (FAA Part 117)
        # duty window = total elapsed time from duty start to this flight's arrival
        # If this is the first flight of a pairing, or a new duty right after rest,
        # reset the reference point to this flight's departure
        legs_after = state.get("legs", 0) + 1
        effective_max_duty = get_max_duty(legs_after, c.get("max_duty"))
        current_duty_start = f["dep_time"] if (pairing_start or is_resting) else duty_start_time
        total_duty_window = f["arr_time"] - current_duty_start
        if total_duty_window > effective_max_duty:
            valid = False
        if legs_after > c.get("max_legs", config.DEFAULT_CONSTRAINTS["max_legs"]):
            valid = False

        # 4. Pairing duration constraint
        elapsed_days = (f["arr_time"] - pairing_start_time) / 24.0
        if elapsed_days > c.get("max_pairing_days", config.DEFAULT_CONSTRAINTS["max_pairing_days"]):
            valid = False

        # 5. Base 복귀 가능성
        if valid:
            ps_time = f["dep_time"] if pairing_start else pairing_start_time
            if not can_reach_any_base(
                base_reaches, f, ps_time, max_pd,
                duty_period=duty_period, max_duty_periods=max_duty_periods,
            ):
                valid = False

        if valid:
            mask[i] = 1

    # END_DUTY (mask[-2] = mask[N])
    can_end_duty = (
        stage_rule["allow_end_duty"]
        and state.get("legs", 0) > 0
        and not is_resting
        and not pairing_start
        and duty_period < c.get("max_duty_periods", config.DEFAULT_CONSTRAINTS["max_duty_periods"])  # based on overnight count
    )
    if can_end_duty:
        mask[config.END_DUTY] = 1

    # END_PAIRING (mask[-1] = mask[N+1])
    pairing_elapsed_days = (state["current_time"] - pairing_start_time) / 24.0
    can_end_pairing = (
        pairing_elapsed_days <= c.get("max_pairing_days", config.DEFAULT_CONSTRAINTS["max_pairing_days"])
    )
    # Turkish pairing은 HB1/HB2 중 어느 home base에서도 종료 가능함.
    if state["current_airport"] not in base_id_set:
        can_end_pairing = False
    if can_end_pairing:
        mask[config.END_PAIRING] = 1

    return mask.tolist()


def get_mask_batch(states, flights, assigneds, constraint=None, stage=3):
    """get_mask()의 배치 버전 (Turkish) -- RL/environment.py::get_mask_batch()와
    구조는 동일하고, HB1/HB2 교차 복귀 부분만 다름:
    - base-reach 체크(5번)를 `base_reaches`(base_id별 reach table)에 대해 OR로 계산
      (delta의 can_reach_base 대신 can_reach_any_base와 동일한 의미)
    - EndPairing은 current_airport가 base_id_set 중 하나이기만 하면 됨(단일 base_ap
      아님) -- 시작(pairing_start)은 delta와 동일하게 여전히 base_ap 하나로 고정

    states/assigneds는 길이 B 리스트, flights/constraint는 전부 공유. 반환값은
    [get_mask(s, flights, a, constraint, stage) for s, a in zip(states, assigneds)]와
    내용이 완전히 같아야 함(전수 비교는 tests/test_environment_turkish_batch.py 참고).
    """
    c = constraint if constraint else config.DEFAULT_CONSTRAINTS
    stage_rule = config.CURRICULUM_CONFIG.get(stage, config.CURRICULUM_CONFIG[3])

    B = len(states)
    N = len(flights)
    if B == 0:
        return []
    flight_ids = [f["id"] for f in flights]

    base_ap = c["base_airport"]
    base_id_set = set(c.get("base_ids") or [base_ap])
    base_reaches = c.get("_base_reaches")
    if base_reaches is None and c.get("_base_reach") is not None:
        base_reaches = {base_ap: c["_base_reach"]}
    if not base_reaches:
        raise ValueError("Turkish CPP constraint에는 _base_reaches가 필요합니다.")
    max_pd           = c.get("max_pairing_days", config.DEFAULT_CONSTRAINTS["max_pairing_days"])
    max_duty_periods = c.get("max_duty_periods", config.DEFAULT_CONSTRAINTS["max_duty_periods"])
    max_legs         = c.get("max_legs", config.DEFAULT_CONSTRAINTS["max_legs"])
    min_conn         = c.get("min_conn", config.DEFAULT_CONSTRAINTS["min_conn"])
    max_conn         = c.get("max_conn", config.DEFAULT_CONSTRAINTS["max_conn"])
    custom_max_duty  = c.get("max_duty")

    # ── flight별 공유 텐서 (N,) ──────────────────────────────────────────
    origin_t = torch.tensor([f["origin"] for f in flights], dtype=torch.long)
    dep_t    = torch.tensor([f["dep_time"] for f in flights], dtype=torch.float64)
    arr_t    = torch.tensor([f["arr_time"] for f in flights], dtype=torch.float64)

    # ── episode별 텐서 (B,) ──────────────────────────────────────────────
    pairing_start_b = torch.tensor([bool(s.get("pairing_start", False)) for s in states], dtype=torch.bool)
    is_resting_b    = torch.tensor([bool(s.get("is_resting", False)) for s in states], dtype=torch.bool)
    rest_end_b      = torch.tensor([s.get("rest_end_time", 0.0) or 0.0 for s in states], dtype=torch.float64)
    duty_period_b   = torch.tensor([s.get("duty_period", 0) for s in states], dtype=torch.float64)
    current_time_b  = torch.tensor([s["current_time"] for s in states], dtype=torch.float64)
    current_airport_b = torch.tensor([s["current_airport"] for s in states], dtype=torch.long)
    duty_start_time_b = torch.tensor(
        [s.get("duty_start_time", s["current_time"]) for s in states], dtype=torch.float64
    )
    pairing_start_time_b = torch.tensor(
        [s.get("pairing_start_time", s["current_time"]) for s in states], dtype=torch.float64
    )
    legs_b       = torch.tensor([s.get("legs", 0) for s in states], dtype=torch.float64)

    assigned_b = torch.tensor(
        [[bool(a.get(fid, False)) for fid in flight_ids] for a in assigneds]
    )  # (B, N)

    # ── 1. 공항 연속성 체크 (시작은 여전히 단일 base_ap) ──────────────────
    cond1 = torch.where(
        pairing_start_b.unsqueeze(1),
        origin_t.unsqueeze(0) == base_ap,
        origin_t.unsqueeze(0) == current_airport_b.unsqueeze(1),
    )  # (B, N)

    # ── 2. connection 시간 체크 ──────────────────────────────────────────
    gap = dep_t.unsqueeze(0) - current_time_b.unsqueeze(1)  # (B, N)
    conn_ok = (gap >= min_conn) & (gap <= max_conn)
    rest_ok = dep_t.unsqueeze(0) >= rest_end_b.unsqueeze(1)
    cond2 = torch.where(
        is_resting_b.unsqueeze(1),
        rest_ok,
        torch.where(pairing_start_b.unsqueeze(1), torch.ones_like(conn_ok), conn_ok),
    )

    # ── 3. duty 시간 제약 ────────────────────────────────────────────────
    legs_after_b = legs_b + 1  # (B,)
    if custom_max_duty is not None:
        effective_max_duty_b = torch.full((B,), float(custom_max_duty), dtype=torch.float64)
    else:
        capped = torch.clamp(legs_after_b, max=6)
        effective_max_duty_b = torch.tensor(
            [config.FAA_DUTY_TABLE.get(int(v.item()), 10.0) for v in capped], dtype=torch.float64
        )
    new_duty_b = pairing_start_b | is_resting_b  # (B,)
    current_duty_start = torch.where(
        new_duty_b.unsqueeze(1),
        dep_t.unsqueeze(0).expand(B, N),
        duty_start_time_b.unsqueeze(1).expand(B, N),
    )
    total_duty_window = arr_t.unsqueeze(0) - current_duty_start  # (B, N)
    cond3a = total_duty_window <= effective_max_duty_b.unsqueeze(1)
    cond3b = (legs_after_b <= max_legs).unsqueeze(1).expand(B, N)

    # ── 4. pairing 기간 제약 ─────────────────────────────────────────────
    elapsed_days = (arr_t.unsqueeze(0) - pairing_start_time_b.unsqueeze(1)) / 24.0  # (B, N)
    cond4 = elapsed_days <= max_pd

    # ── 5. base 복귀 가능성 체크 (HB1/HB2 중 하나라도 복귀 가능하면 OK, OR) ──
    ps_time = torch.where(
        pairing_start_b.unsqueeze(1),
        dep_t.unsqueeze(0).expand(B, N),
        pairing_start_time_b.unsqueeze(1).expand(B, N),
    )
    cond5 = torch.tensor([
        [can_reach_any_base(
            base_reaches, f,
            f["dep_time"] if states[b].get("pairing_start", False)
            else states[b].get("pairing_start_time", states[b]["current_time"]),
            max_pd, duty_period=states[b].get("duty_period", 0),
            max_duty_periods=max_duty_periods,
        ) for f in flights]
        for b in range(B)
    ], dtype=torch.bool)

    valid = cond1 & cond2 & cond3a & cond3b & cond4 & cond5
    mask_flights = valid & (~assigned_b)  # (B, N)

    # ── EndDuty ───────────────────────────────────────────────────────────
    can_end_duty = (
        bool(stage_rule["allow_end_duty"])
        & (legs_b > 0)
        & (~is_resting_b)
        & (~pairing_start_b)
        & (duty_period_b < max_duty_periods)
    )  # (B,)

    # ── EndPairing (HB1/HB2 중 어디서든 종료 가능) ────────────────────────
    pairing_elapsed_days = (current_time_b - pairing_start_time_b) / 24.0
    in_base_set = torch.isin(current_airport_b, torch.tensor(sorted(base_id_set), dtype=torch.long))
    can_end_pairing = (
        (pairing_elapsed_days <= max_pd)
        & in_base_set
    )  # (B,)

    mask = torch.cat(
        [mask_flights, can_end_duty.unsqueeze(1), can_end_pairing.unsqueeze(1)], dim=1
    ).to(torch.int32)  # (B, N+2)
    return mask.tolist()


def step(state, action, flights, assigned, constraint=None):
    """Take an action and return next_state, reward, done.

    action range:
      0 .. N-1  : select a flight
      N         : END_DUTY
      N+1       : END_PAIRING
    done=True when: END_PAIRING is chosen and all flights are covered
    """
    c = constraint if constraint else config.DEFAULT_CONSTRAINTS
    N = len(flights)
    if action < 0 or action >= N + 2:
        raise IndexError("action이 허용 범위를 벗어났습니다.")

    # END_DUTY -> enter rest, pairing continues
    if action == N:
        if not get_mask(state, flights, assigned, c)[config.END_DUTY]:
            raise ValueError("현재 상태에서는 END_DUTY를 선택할 수 없습니다.")
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
        # v8: directly rewards overnight rest -- encourages multi-day pairings.
        # Overnight 10h is excluded from dead_time, so avg_legs can improve without increasing
        # dead_time as legs go up.
        # v16: scale the bonus proportionally when a duty has fewer than MIN_LEGS_FOR_DUTY_BONUS
        # legs -- removes the incentive to end duties short and collect the risk-free fixed
        # END_DUTY reward repeatedly.
        duty_legs = state.get("legs", 0)
        min_legs = config.MIN_LEGS_FOR_DUTY_BONUS
        scale = min(1.0, duty_legs / min_legs) if min_legs > 0 else 1.0
        return next_state, config.END_DUTY_BONUS * scale, False

    # END_PAIRING -> charge the pairing cost, then start a new pairing (or end the episode)
    if action == N + 1:
        if not get_mask(state, flights, assigned, c)[config.END_PAIRING]:
            raise ValueError("CPP 제약을 만족하지 않은 pairing은 종료할 수 없습니다.")
        p_cost = c.get("pairing_cost", config.DEFAULT_CONSTRAINTS["pairing_cost"])
        # constraint["base_airport"] is injected per episode
        base = c["base_airport"]
        base_id_set = set(c.get("base_ids") or [base])

        total_legs = state.get("total_legs", 0)
        # LEG_PER_PAIRING_BONUS는 이미 flight 선택마다 즉시 지급됨(line ~405) -- 여기서
        # total_legs배로 다시 지급하면 이중 지급이 됨 (RL/environment.py와 동일 버그).
        reward = -p_cost

        if total_legs < config.MIN_LEGS_FOR_PAIRING:
            reward += config.MIN_LEGS_PENALTY

        unassigned = [f for f in flights if not assigned[f["id"]]]
        if not unassigned:
            # All flights covered -> end the episode
            return state, reward, True
        # Unassigned flights remain -> start a new pairing
        # 도착한 Turkish home base에서 다음 pairing을 시작함.
        restart_base = state["current_airport"] if state["current_airport"] in base_id_set else base
        base_unassigned = [f for f in unassigned if f["origin"] == restart_base]
        # restart_base에서 출발 가능한 미배정 flight가 없으면, 이 정책으로는 더 이상
        # 합법적인 pairing을 시작할 수 없음(모든 pairing은 base에서 시작해야 하므로
        # base 아닌 flight로 새 pairing을 억지로 시작시키면 그 자체가 CPP 위반).
        # RL/environment.py(delta)와 동일한 이유로 episode를 바로 종료함 -- 남은
        # flight는 uncovered로 두고 후처리(salvage/rescue 등)가 커버함.
        if not base_unassigned:
            return state, reward, True
        next_time = min(f["dep_time"] for f in base_unassigned)
        next_state = {
            **state,
            "current_airport":    restart_base,
            "current_time":       next_time,
            "duty_time":          0.0,
            "duty_start_time":    next_time,
            "legs":               0,
            "total_legs":         0,    # reset when a new pairing starts
            "duty_period":        0,
            "is_resting":         False,
            "rest_end_time":      None,
            "pairing_start":      True,
            "pairing_start_time": next_time,
        }
        return next_state, reward, False

    # flight action도 직접 호출 시 hard mask legality를 다시 확인함.
    if not get_mask(state, flights, assigned, c)[action]:
        raise ValueError("CPP 제약을 위반한 flight는 선택할 수 없습니다.")
    f = flights[action]
    assigned[f["id"]] = True
    flight_time = f["arr_time"] - f["dep_time"]

    # Reset pairing_start_time on a pairing's first flight; reset duty_start_time on a new
    # duty right after rest
    p_start_time = f["dep_time"] if state.get("pairing_start", False) else state["pairing_start_time"]
    d_start_time = f["dep_time"] if (state.get("pairing_start", False) or state.get("is_resting", False)) \
                   else state["duty_start_time"]

    next_state = {
        "current_airport":    f["dest"],
        "current_time":       f["arr_time"],
        "duty_time":          (0.0 if state.get("is_resting", False) else state["duty_time"]) + flight_time,
        "duty_start_time":    d_start_time,
        "legs":               state.get("legs", 0) + 1,
        "total_legs":         state.get("total_legs", 0) + 1,  # accumulates over the whole pairing (not reset on END_DUTY)
        "remaining":          state["remaining"] - 1,
        "pairing_start":      False,
        "duty_period":        state.get("duty_period", 0),
        "pairing_start_time": p_start_time,
        "is_resting":         False,
        "rest_end_time":      None,
    }

    # dead time reward: penalty for wait time between flights within a duty, plus a connection bonus
    # No gap penalty for a pairing's first flight or the first flight right after rest
    # LEG_PER_PAIRING_BONUS: paid immediately on every flight selection (no deferred payment at
    # END_PAIRING) -> solves the credit assignment problem: the agent recognizes multi-leg value
    # immediately
    if not state.get("pairing_start", False) and not state.get("is_resting", False):
        reward = -(f["dep_time"] - state["current_time"]) + config.LEG_CONN_BONUS + config.LEG_PER_PAIRING_BONUS
    else:
        reward = config.LEG_PER_PAIRING_BONUS  # first flight / right after rest: bonus only, no gap penalty

    return next_state, reward, False


def step_batch(states, actions, flights, assigneds, constraint=None):
    """step()의 얇은 배치 wrapper (Turkish) -- RL/environment.py::step_batch()와
    동일한 이유로 텐서 벡터화 대신 기존 검증된 step()을 그대로 B번 호출한다.

    states/actions/assigneds: 길이 B 리스트, flights/constraint는 전부 공유.
    반환: (next_states, rewards, dones) 각각 길이 B 리스트.
    """
    next_states, rewards, dones = [], [], []
    for state, action, assigned in zip(states, actions, assigneds):
        next_state, reward, done = step(state, action, flights, assigned, constraint)
        next_states.append(next_state)
        rewards.append(reward)
        dones.append(done)
    return next_states, rewards, dones


def final_reward(assigned, custom_penalty=None):
    """Return the penalty for unassigned flights at episode end."""
    penalty = custom_penalty if custom_penalty is not None else config.DEFAULT_CONSTRAINTS["uncovered_penalty"]
    remaining = sum(1 for v in assigned.values() if not v)
    return -penalty * remaining
