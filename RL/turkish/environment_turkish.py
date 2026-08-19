import numpy as np
import config
from base_reach import can_reach_base

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

    # First leg of a pairing: force departure from a base if unassigned base-origin flights
    # remain. Once base-origin flights are exhausted, lift the origin restriction to prevent
    # deadhead loops.
    # HB1/HB2 asymmetry: if base_ids is given, departure from any base in it is accepted.
    # base_remaining does not depend on the candidate flight f (loop-invariant), so it is
    # computed once outside the loop -- recomputing it inside the loop would be O(N^2), which
    # slows episodes on low-connectivity bases (HB2) to tens of seconds each (found in the
    # log/0704 turkish smoke test).
    base_ap = c["base_airport"]
    base_id_set = set(c.get("base_ids") or [base_ap])
    # Turkish CPP도 episode의 출발 base로 복귀 가능한 action만 허용함.
    base_reach = c.get("_base_reach")
    if base_reach is None:
        # CPP 실행에는 base 복귀 가능성 자료가 필수이며 누락은 구성 오류로 처리함.
        raise ValueError("CPP constraint에는 _base_reach가 필요합니다.")
    max_pd           = c.get("max_pairing_days", config.DEFAULT_CONSTRAINTS["max_pairing_days"])
    max_duty_periods = c.get("max_duty_periods", config.DEFAULT_CONSTRAINTS["max_duty_periods"])
    if pairing_start:
        base_remaining = any(
            not assigned[fl["id"]] and fl["origin"] in base_id_set
            for fl in flights
        )

    for i, f in enumerate(flights):
        if assigned[f["id"]]:
            continue

        valid = True

        # 1. Airport connectivity check
        if pairing_start:
            if f["origin"] != base_ap:
                valid = False
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
            if not can_reach_base(
                base_reach, f, ps_time, max_pd,
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
    # min_pairing_legs: set per airline (Delta/Alaska/JetBlue=3, Turkish=2)
    pairing_elapsed_days = (state["current_time"] - pairing_start_time) / 24.0
    min_pairing_legs = c.get("min_pairing_legs", 2)
    can_end_pairing = (
        state.get("total_legs", 0) >= min_pairing_legs
        and pairing_elapsed_days <= c.get("max_pairing_days", config.DEFAULT_CONSTRAINTS["max_pairing_days"])
    )
    # CPP pairing은 episode의 출발 base에서만 종료 가능함.
    if state["current_airport"] != base_ap:
        can_end_pairing = False
    if can_end_pairing:
        mask[config.END_PAIRING] = 1

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

        total_legs = state.get("total_legs", 0)
        reward = -p_cost + total_legs * config.LEG_PER_PAIRING_BONUS

        if total_legs < config.MIN_LEGS_FOR_PAIRING:
            reward += config.MIN_LEGS_PENALTY

        unassigned = [f for f in flights if not assigned[f["id"]]]
        if not unassigned:
            # All flights covered -> end the episode
            return state, reward, True
        # Unassigned flights remain -> start a new pairing
        # 다음 pairing도 현재 episode에 지정된 동일 base에서 시작함.
        restart_base = base
        base_unassigned = [f for f in unassigned if f["origin"] == base]
        next_time = min(f["dep_time"] for f in base_unassigned) if base_unassigned else min(f["dep_time"] for f in unassigned)
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

    # Select a flight
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


def final_reward(assigned, custom_penalty=None):
    """Return the penalty for unassigned flights at episode end."""
    penalty = custom_penalty if custom_penalty is not None else config.DEFAULT_CONSTRAINTS["uncovered_penalty"]
    remaining = sum(1 for v in assigned.values() if not v)
    return -penalty * remaining
