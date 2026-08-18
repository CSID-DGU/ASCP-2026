import numpy as np
import config
from base_reach import can_reach_base

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

    # First leg of a pairing: if unassigned base-origin legs remain, force
    # departure from the base; once base-origin legs are exhausted, lift the
    # origin restriction to avoid a deadhead loop.
    # base_remaining does not depend on the candidate flight f (loop-invariant),
    # so it is computed once outside the loop rather than recomputed per
    # candidate (which would be O(N^2)).
    base_ap = c.get("base_airport", config.DEFAULT_CONSTRAINTS["base_airport"])

    # Hard base-return enforcement (decode-time feasibility masking, Eq. 6).
    # require_base_return=True (1) masks any leg that would make base return
    # infeasible and (2) forbids EndPairing away from the base -- so every
    # completed pairing is structurally guaranteed to start and end at the
    # base, matching the backward-reachability-from-base mechanism described
    # in the paper. _base_reach is precomputed once per (flights, base) by
    # the caller in rollout.py and passed through the constraint dict.
    require_return   = c.get("require_base_return", False)
    base_reach       = c.get("_base_reach") if require_return else None
    if require_return and base_reach is None:
        # strict 모드가 복귀 가능성 검사를 빠뜨린 채 완화되지 않도록 즉시 중단함.
        raise ValueError("require_base_return=True이면 _base_reach가 필요합니다.")
    max_pd           = c.get("max_pairing_days", config.DEFAULT_CONSTRAINTS["max_pairing_days"])
    max_duty_periods = c.get("max_duty_periods", config.DEFAULT_CONSTRAINTS["max_duty_periods"])

    if pairing_start:
        base_remaining = any(
            not assigned[fl["id"]] and fl["origin"] == base_ap
            for fl in flights
        )

    for i, f in enumerate(flights):
        if assigned[f["id"]]:
            continue

        valid = True

        # 1. Airport-continuity check
        if pairing_start:
            if c.get("strict_base_start", False) and f["origin"] != base_ap:
                valid = False
            elif base_remaining and f["origin"] != base_ap:
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

        # 5. Base-reachability (only when require_base_return is set)
        # Checked via duty_period/max_duty_periods, not max_legs -- EndDuty can
        # always grant a fresh leg budget, so per-duty leg count is not the
        # binding resource for reaching the base; remaining overnight/rest
        # opportunities are.
        if valid and base_reach is not None:
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
    # Failing to return to base is otherwise a soft penalty applied as reward
    # in step(), not a hard mask. When require_base_return is set, EndPairing
    # away from the base is masked out (hard constraint).
    if require_return and state["current_airport"] != base_ap:
        can_end_pairing = False
    if can_end_pairing:
        mask[config.END_PAIRING] = 1

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

    # EndDuty -> enter rest, pairing continues
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
        p_cost = c.get("pairing_cost", config.DEFAULT_CONSTRAINTS["pairing_cost"])
        base_penalty = c.get("base_penalty", config.DEFAULT_CONSTRAINTS["base_penalty"])
        # constraint["base_airport"] is injected per episode
        base = c.get("base_airport", config.DEFAULT_CONSTRAINTS["base_airport"])

        total_legs = state.get("total_legs", 0)
        reward = -p_cost + total_legs * config.LEG_PER_PAIRING_BONUS

        if total_legs < config.MIN_LEGS_FOR_PAIRING:
            reward += config.MIN_LEGS_PENALTY
        if state["current_airport"] != base:
            reward -= base_penalty

        unassigned = [f for f in flights if not assigned[f["id"]]]
        if not unassigned:
            # All flights covered -> end episode
            return state, reward, True
        # Flights remain unassigned -> start a new pairing (anchored on the
        # earliest base-departing leg)
        base_unassigned = [f for f in unassigned if f["origin"] == base]
        next_time = min(f["dep_time"] for f in base_unassigned) if base_unassigned else min(f["dep_time"] for f in unassigned)
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

    # Select a flight leg
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


def final_reward(assigned, custom_penalty=None):
    """Return the end-of-episode penalty for flights left unassigned."""
    penalty = custom_penalty if custom_penalty is not None else config.DEFAULT_CONSTRAINTS["uncovered_penalty"]
    remaining = sum(1 for v in assigned.values() if not v)
    return -penalty * remaining
