import numpy as np
import torch
import config
from base_reach import can_reach_base

INF = float("inf")

# flight dict keys: "origin", "dest", "dep_time", "arr_time", "id"


def get_max_duty(legs_count, custom_max_duty=None):
    """Duty-time limit: use constraint["max_duty"] if given, else the FAA_DUTY_TABLE default.

    legs_count: number of legs this duty would have after adding the candidate (current legs + 1)
    custom_max_duty: constraint["max_duty"] value. None falls back to DEFAULT_CONSTRAINTS.

    The rule-profile value (custom_max_duty) is used as-is, with no FAA-table
    min() applied on top -- it is the sole source of truth for masking, so it
    stays consistent with the same value used as FiLM input.
    """
    if custom_max_duty is not None:
        return custom_max_duty
    return config.FAA_DUTY_TABLE.get(min(legs_count, 6), 10.0)


def get_mask(state, flights, assigned, constraint=None, stage=3):
    """Dynamic feasibility mask: return the selectable-action mask A_feas_t(c) for the current state.

    Paper Sec. "Constraint-Conditioned Pairing Generator": the dynamic
    feasibility mechanism builds A_feas_t(c) subset of A_t by removing
    actions that violate airport continuity, connection-time limits,
    duty-time limits, max legs per duty, rest requirements, overnight
    limits, or pairing-duration constraints, plus backward-reachability
    pruning to the base. This mask becomes the additive mask b_mask_t(j; c)
    of Eq. (6), applied in model/decoder.py.

    Returns: a list of size [N + 2] (0: infeasible, 1: feasible)
             index 0..N-1 = flight legs, N = EndDuty, N+1 = EndPairing
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

    base_ap = c["base_airport"]

    # CPP pairing이 base 복귀 가능성을 잃는 action을 항상 제거함.
    base_reach = c.get("_base_reach")
    if base_reach is None:
        # CPP 실행에는 base 복귀 가능성 자료가 필수이며 누락은 구성 오류로 처리함.
        raise ValueError("CPP constraint에는 _base_reach가 필요합니다.")
    max_pd           = c.get("max_pairing_days", config.DEFAULT_CONSTRAINTS["max_pairing_days"])
    max_duty_periods = c.get("max_duty_periods", config.DEFAULT_CONSTRAINTS["max_duty_periods"])

    # base_remaining(계산만 되고 mask에 반영 안 되던 지역변수) 제거함 -- 원래 주석은
    # "base 출발 flight가 소진되면 origin 제약을 풀어준다"는 의도였지만, 그 자체가
    # CPP 규칙(pairing은 반드시 base에서 시작) 위반이라 맞는 해법이 아니었음. 진짜
    # 원인은 step()의 EndPairing 처리로 보임(base 출발 flight가 없어도 에피소드를
    # 안 끝내고 계속 진행 -> 데드락 가능) -- step() 쪽 실제 수정은 이 파일 담당자 판단 필요.

    for i, f in enumerate(flights):
        if assigned[f["id"]]:
            continue

        valid = True

        # 1. Airport-continuity check
        if pairing_start:
            if f["origin"] != base_ap:
                valid = False
        elif f["origin"] != state["current_airport"]:
            valid = False

        # 2. Connection-time check
        if is_resting:
            # Cannot board before rest ends
            if f["dep_time"] < rest_end:
                valid = False
        else:
            if not pairing_start:
                gap = f["dep_time"] - state["current_time"]
                if gap < c.get("min_conn", config.DEFAULT_CONSTRAINTS["min_conn"]) or \
                   gap > c.get("max_conn", config.DEFAULT_CONSTRAINTS["max_conn"]):
                    valid = False

        # 3. Duty-time constraint (FAA Part 117)
        # duty window = elapsed time from duty start to this flight's arrival.
        # If this is the pairing's first leg or the first leg of a new duty
        # after rest, reset the reference point to this flight's departure.
        legs_after = state.get("legs", 0) + 1
        effective_max_duty = get_max_duty(legs_after, c.get("max_duty"))
        current_duty_start = f["dep_time"] if (pairing_start or is_resting) else duty_start_time
        total_duty_window = f["arr_time"] - current_duty_start
        if total_duty_window > effective_max_duty:
            valid = False
        if legs_after > c.get("max_legs", config.DEFAULT_CONSTRAINTS["max_legs"]):
            valid = False

        # 4. Pairing-duration constraint (P_max)
        elapsed_days = (f["arr_time"] - pairing_start_time) / 24.0
        if elapsed_days > max_pd:
            valid = False

        # 5. Base 복귀 가능성
        # Checked via duty_period/max_duty_periods, not max_legs -- EndDuty can
        # always grant a fresh leg budget, so per-duty leg count is not the
        # binding resource for reaching the base; remaining overnight/rest
        # opportunities are.
        if valid:
            ps_time = f["dep_time"] if pairing_start else pairing_start_time
            if not can_reach_base(
                base_reach, f, ps_time, max_pd,
                duty_period=duty_period, max_duty_periods=max_duty_periods,
            ):
                valid = False

        if valid:
            mask[i] = 1

    # EndDuty (mask[-2] = mask[N])
    can_end_duty = (
        stage_rule["allow_end_duty"]
        and state.get("legs", 0) > 0
        and not is_resting
        and not pairing_start
        and duty_period < c.get("max_duty_periods", config.DEFAULT_CONSTRAINTS["max_duty_periods"])  # overnight count limit
    )
    if can_end_duty:
        mask[config.END_DUTY] = 1

    # EndPairing (mask[-1] = mask[N+1])
    # min_pairing_legs: airline-specific (Delta/Alaska/JetBlue=3, Turkish=2)
    pairing_elapsed_days = (state["current_time"] - pairing_start_time) / 24.0
    min_pairing_legs = c.get("min_pairing_legs", 2)
    can_end_pairing = (
        state.get("total_legs", 0) >= min_pairing_legs
        and pairing_elapsed_days <= c.get("max_pairing_days", config.DEFAULT_CONSTRAINTS["max_pairing_days"])
    )
    # CPP pairing은 base에 도착한 상태에서만 종료 가능함.
    if state["current_airport"] != base_ap:
        can_end_pairing = False
    if can_end_pairing:
        mask[config.END_PAIRING] = 1

    return mask.tolist()


def get_mask_batch(states, flights, assigneds, constraint=None, stage=3):
    """get_mask()의 배치 버전 -- states/assigneds는 길이 B 리스트(episode마다 하나씩),
    flights/constraint는 전부 공유(rollout_batch()가 원래 하나의 constraint/_base_reach를
    모든 episode에 똑같이 넘기는 것과 동일한 전제). 반환값: 길이 N+2인 mask가 B개
    담긴 리스트 -- [get_mask(s, flights, a, constraint, stage) for s, a in
    zip(states, assigneds)]와 내용이 완전히 같아야 함(전수 비교는
    tests/test_environment_batch.py 참고)
    """
    c = constraint if constraint else config.DEFAULT_CONSTRAINTS
    stage_rule = config.CURRICULUM_CONFIG.get(stage, config.CURRICULUM_CONFIG[3])

    B = len(states)
    N = len(flights)
    if B == 0:
        return []
    flight_ids = [f["id"] for f in flights]

    base_ap = c["base_airport"]
    base_reach = c.get("_base_reach")
    if base_reach is None:
        raise ValueError("CPP constraint에는 _base_reach가 필요합니다.")
    max_pd           = c.get("max_pairing_days", config.DEFAULT_CONSTRAINTS["max_pairing_days"])
    max_duty_periods = c.get("max_duty_periods", config.DEFAULT_CONSTRAINTS["max_duty_periods"])
    max_legs         = c.get("max_legs", config.DEFAULT_CONSTRAINTS["max_legs"])
    min_conn         = c.get("min_conn", config.DEFAULT_CONSTRAINTS["min_conn"])
    max_conn         = c.get("max_conn", config.DEFAULT_CONSTRAINTS["max_conn"])
    min_pairing_legs = c.get("min_pairing_legs", 2)
    custom_max_duty  = c.get("max_duty")

    # ── flight별 공유 텐서 (N,) ──────────────────────────────────────────
    origin_t = torch.tensor([f["origin"] for f in flights], dtype=torch.long)
    dep_t    = torch.tensor([f["dep_time"] for f in flights], dtype=torch.float64)
    arr_t    = torch.tensor([f["arr_time"] for f in flights], dtype=torch.float64)
    reach_time_t = torch.tensor(
        [base_reach["time"].get(fid, INF) for fid in flight_ids], dtype=torch.float64
    )
    reach_duty_t = torch.tensor(
        [base_reach["duty_crossings"].get(fid, INF) for fid in flight_ids], dtype=torch.float64
    )

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
    total_legs_b = torch.tensor([s.get("total_legs", 0) for s in states], dtype=torch.float64)

    assigned_b = torch.tensor(
        [[bool(a.get(fid, False)) for fid in flight_ids] for a in assigneds]
    )  # (B, N)

    # ── 1. 공항 연속성 체크 ──────────────────────────────────────────────
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

    # ── 5. base 복귀 가능성 체크 ─────────────────────────────────────────
    ps_time = torch.where(
        pairing_start_b.unsqueeze(1),
        dep_t.unsqueeze(0).expand(B, N),
        pairing_start_time_b.unsqueeze(1).expand(B, N),
    )
    cond5 = torch.tensor([
        [can_reach_base(
            base_reach, f,
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
    # stage_rule["allow_end_duty"]는 plain bool -- tensor와 Python `and`를 섞으면
    # False일 때 tensor가 아니라 bare False가 반환돼서 아래 cat()이 깨지므로 `&`만 사용.
    can_end_duty = (
        bool(stage_rule["allow_end_duty"])
        & (legs_b > 0)
        & (~is_resting_b)
        & (~pairing_start_b)
        & (duty_period_b < max_duty_periods)
    )  # (B,)

    # ── EndPairing ────────────────────────────────────────────────────────
    pairing_elapsed_days = (current_time_b - pairing_start_time_b) / 24.0
    can_end_pairing = (
        (total_legs_b >= min_pairing_legs)
        & (pairing_elapsed_days <= max_pd)
        & (current_airport_b == base_ap)
    )  # (B,)

    mask = torch.cat(
        [mask_flights, can_end_duty.unsqueeze(1), can_end_pairing.unsqueeze(1)], dim=1
    ).to(torch.int32)  # (B, N+2)
    return mask.tolist()


def step(state, action, flights, assigned, constraint=None):
    """Apply an action and return (next_state, reward, done).

    The returned reward is the local reward r^loc_t of Eq. (10) (Phase I
    curriculum reward: rewards coverage and efficient connections while
    penalizing wait time, overly short pairings, and unnecessary return
    movements). Phase II adds the net-dual term w_dual(e)*delta_i on top of
    this value (see experiments/train.py::run_episode_with_dual).

    action range:
      0 .. N-1  : select flight leg
      N         : EndDuty
      N+1       : EndPairing
    done=True when EndPairing is selected and all flights are covered.
    """
    c = constraint if constraint else config.DEFAULT_CONSTRAINTS
    N = len(flights)
    if action < 0 or action >= N + 2:
        raise IndexError("action이 허용 범위를 벗어났습니다.")

    # EndDuty -> enter rest, pairing continues
    if action == N:
        if not get_mask(state, flights, assigned, c)[config.END_DUTY]:
            raise ValueError("현재 상태에서는 EndDuty를 선택할 수 없습니다.")
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
        # Directly reward overnight rest to encourage multi-day pairings.
        # Overnight time is excluded from dead_time, so this raises avg_legs
        # without increasing dead_time. The bonus is scaled down when the duty
        # has fewer than MIN_LEGS_FOR_DUTY_BONUS legs, removing the incentive
        # to end duties early just to collect the flat EndDuty bonus repeatedly.
        duty_legs = state.get("legs", 0)
        min_legs = config.MIN_LEGS_FOR_DUTY_BONUS
        scale = min(1.0, duty_legs / min_legs) if min_legs > 0 else 1.0
        return next_state, config.END_DUTY_BONUS * scale, False

    # EndPairing -> charge pairing cost, then start a new pairing (or end the episode)
    if action == N + 1:
        if not get_mask(state, flights, assigned, c)[config.END_PAIRING]:
            raise ValueError("CPP 제약을 만족하지 않은 pairing은 종료할 수 없습니다.")
        p_cost = c.get("pairing_cost", config.DEFAULT_CONSTRAINTS["pairing_cost"])
        # constraint["base_airport"] is injected per episode
        base = c["base_airport"]

        total_legs = state.get("total_legs", 0)
        # LEG_PER_PAIRING_BONUS는 이미 leg 선택마다 즉시 지급됨(line ~429) -- 여기서
        # total_legs배로 다시 지급하면 이중 지급이 됨(같은 pairing 안에서 leg당 2번씩
        # 보너스를 받는 셈).
        reward = -p_cost

        if total_legs < config.MIN_LEGS_FOR_PAIRING:
            reward += config.MIN_LEGS_PENALTY

        unassigned = [f for f in flights if not assigned[f["id"]]]
        if not unassigned:
            # All flights covered -> end episode
            return state, reward, True
        # base에서 출발 가능한 미배정 flight가 하나도 없으면, 이 정책으로는 더 이상
        # 합법적인 pairing을 시작할 수 없음(모든 pairing은 base에서 시작해야 하므로
        # base 아닌 flight로 새 pairing을 억지로 시작시키면 CPP 위반)
        # 이전엔 base 아닌 flight의 dep_time으로 pairing_start=True인 state를
        # 만들었는데, current_airport=base라고 주장하면서 실제로는 base에서
        # 출발 못 하는 flight를 기준으로 삼는 모순된 state였음 -- get_mask()에서
        # 모든 action이 막혀서(전부 0) 사실상 데드락 상태(호출부의 별도
        # 방어 코드 덕분에 크래시는 안 났지만 불필요한 헛스텝이 발생)
        # 남은 flight들은 uncovered로 두고(추후 salvage/rescue 등 후처리가 커버) 그냥
        # episode를 끝내기 -- unassigned 전부 소진됐을 때와 동일하게 처리
        base_unassigned = [f for f in unassigned if f["origin"] == base]
        if not base_unassigned:
            return state, reward, True
        next_time = min(f["dep_time"] for f in base_unassigned)
        next_state = {
            **state,
            "current_airport":    base,
            "current_time":       next_time,
            "duty_time":          0.0,
            "duty_start_time":    next_time,
            "legs":               0,
            "total_legs":         0,    # reset at the start of a new pairing
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

    # Reset pairing_start_time on the pairing's first leg; reset duty_start_time
    # on the first leg of a new duty right after rest
    p_start_time = f["dep_time"] if state.get("pairing_start", False) else state["pairing_start_time"]
    d_start_time = f["dep_time"] if (state.get("pairing_start", False) or state.get("is_resting", False)) \
                   else state["duty_start_time"]

    next_state = {
        "current_airport":    f["dest"],
        "current_time":       f["arr_time"],
        "duty_time":          (0.0 if state.get("is_resting", False) else state["duty_time"]) + flight_time,
        "duty_start_time":    d_start_time,
        "legs":               state.get("legs", 0) + 1,
        "total_legs":         state.get("total_legs", 0) + 1,  # cumulative over the whole pairing (not reset by EndDuty)
        "remaining":          state["remaining"] - 1,
        "pairing_start":      False,
        "duty_period":        state.get("duty_period", 0),
        "pairing_start_time": p_start_time,
        "is_resting":         False,
        "rest_end_time":      None,
    }

    # Dead-time reward: penalizes wait time between flights within a duty and
    # rewards efficient connections (part of the local reward r^loc_t, Eq. 10).
    # No gap penalty on the pairing's first leg or the first leg after rest.
    # LEG_PER_PAIRING_BONUS is paid immediately on every leg selection (not
    # deferred to EndPairing) so the credit-assignment signal for multi-leg
    # value is immediate.
    if not state.get("pairing_start", False) and not state.get("is_resting", False):
        reward = -(f["dep_time"] - state["current_time"]) + config.LEG_CONN_BONUS + config.LEG_PER_PAIRING_BONUS
    else:
        reward = config.LEG_PER_PAIRING_BONUS  # first leg / right after rest: bonus only, no gap penalty

    return next_state, reward, False


def step_batch(states, actions, flights, assigneds, constraint=None):
    """step()의 얇은 배치 wrapper -- Phase 3, experiment/rollout-batch-vectorization.

    step() 자체는 state 필드 몇 개를 갱신하는 가벼운 Python 로직이라 텐서
    벡터화로 얻는 성능 이득이 없음(진짜 병목인 신경망 forward pass는 이미
    state_to_vec_batch()/get_mask_batch()에서 배치화됨) -- 그래서 로직을
    다시 구현하지 않고 기존에 검증된 step()을 그대로 B번 호출한다.

    states/actions/assigneds: 길이 B 리스트(episode마다 하나씩), flights/constraint는
    전부 공유. assigneds의 각 dict는 step()이 그 자리에서 직접 수정함(step()과 동일).

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
    """Return the end-of-episode penalty for flights left unassigned."""
    penalty = custom_penalty if custom_penalty is not None else config.DEFAULT_CONSTRAINTS["uncovered_penalty"]
    remaining = sum(1 for v in assigned.values() if not v)
    return -penalty * remaining
