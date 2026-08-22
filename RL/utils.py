# utils.py -- shared utilities for experiments/train.py and evaluation/evaluate_ip.py
#
# Shared conversions:
#   - constraint_to_tensor: constraint dict -> FiLM input tensor c~ (Eq. 1, 5)
#   - flights_to_tensors: list of flight dicts -> model input tensors
#   - state_to_vec: environment state dict -> decoder query input tensor
#   - flight_gap_bias: per-candidate connection gap g_t(j) used in the
#     pointer logit's w_gap * g_t(j) term (Eq. 7)

import torch
import config
from constraints import FILM_CONSTRAINT_KEYS

# 2x2 FiLM 인과성 실험(C/D/C'/D', log/0717/FiLM_방향결정_및_계획.md §6)용 —
# True면 state_to_vec()이 constraint_vec 자리를 0으로 채워 디코더의 constraint
# 직접 concat 경로를 원천 차단한다(C'/D' 조건). set_skip_decoder_constraint()로
# 학습/평가 스크립트 시작 시 한 번 설정하면 이 모듈을 쓰는 모든 호출부
# (experiments/train.py, RL/rollout.py, evaluation/evaluate_ip.py)에 즉시 반영된다.
_SKIP_DECODER_CONSTRAINT = False


def set_skip_decoder_constraint(flag: bool):
    global _SKIP_DECODER_CONSTRAINT
    _SKIP_DECODER_CONSTRAINT = flag


def constraint_to_tensor(constraint, device=None):
    """Convert a constraint dict to the FiLM input tensor c~ of shape (constraint_dim,) (Eq. 5).

    Normalized by CONSTRAINT_NORMS into [0, 1].
    """
    t = torch.tensor(
        [constraint[k] / config.CONSTRAINT_NORMS[k] for k in FILM_CONSTRAINT_KEYS],
        dtype=torch.float32,
    )
    if device is not None:
        t = t.to(device)
    return t


def flights_to_tensors(flights, max_time, device=None):
    """Convert a list of flight dicts into (origins, dests, dep_times, arr_times, fly_times) tensors.

    max_time: time-normalization denominator. window_days*24 during training,
    ckpt["max_time"] during evaluation.
    """
    origins  = torch.tensor([f["origin"]   for f in flights], dtype=torch.long)
    dests    = torch.tensor([f["dest"]     for f in flights], dtype=torch.long)
    dep_raw  = torch.tensor([f["dep_time"] for f in flights], dtype=torch.float32)
    arr_raw  = torch.tensor([f["arr_time"] for f in flights], dtype=torch.float32)
    dep_norm = dep_raw / max_time
    arr_norm = arr_raw / max_time
    fly_norm = (arr_raw - dep_raw) / max_time
    if device is not None:
        origins  = origins.to(device)
        dests    = dests.to(device)
        dep_norm = dep_norm.to(device)
        arr_norm = arr_norm.to(device)
        fly_norm = fly_norm.to(device)
    return origins, dests, dep_norm, arr_norm, fly_norm


def state_to_vec(state, encoder, constraint, device=None, include_total_legs=True):
    """Convert an environment state dict into the decoder query input tensor state_vec (79,).

    state_vec = current_airport_emb(32) + base_airport_emb(32) + scalars(8) + constraint_vec(7)

    8 scalars:
      time_of_day, day_norm, duty_elapsed/NORM, legs/NORM,
      duty_period/NORM, is_resting, rest_remaining, total_legs/NORM

    constraint_vec: normalized FILM_CONSTRAINT_KEYS values (Eq. 5 c~) --
    concatenated so the decoder query q_t can see the rule profile directly,
    in addition to conditioning the encoder via FiLM.

    Fixed denominators (CONSTRAINT_NORMS) are used rather than dividing by the
    constraint's own value, because doing the latter would encode constraint
    information into state_vec itself and weaken the FiLM gradient.
    CONSTRAINT_NORMS is fixed at the training-range maximum, so FiLM remains
    the sole pathway through which rule-profile information reaches the model.
    """
    dev = device or torch.device("cpu")
    current_emb = encoder.airport_emb(torch.tensor(state["current_airport"]).to(dev))
    base_emb    = encoder.airport_emb(torch.tensor(constraint["base_airport"]).to(dev))

    time_of_day      = (state["current_time"] % 24.0) / 24.0
    day_norm         = (state["current_time"] // 24.0) / config.CONSTRAINT_NORMS["max_pairing_days"]
    duty_period_norm = state.get("duty_period", 0) / config.CONSTRAINT_NORMS["max_duty_periods"]

    if state.get("is_resting", False) or state.get("pairing_start", False):
        duty_elapsed = 0.0
    else:
        duty_elapsed = max(0.0, state["current_time"] - state.get("duty_start_time", state["current_time"]))

    if state.get("is_resting", False) and state.get("rest_end_time") is not None:
        rest_remaining = max(0.0, state["rest_end_time"] - state["current_time"]) / config.CONSTRAINT_NORMS["min_rest"]
    else:
        rest_remaining = 0.0

    scalars = [
        time_of_day,
        day_norm,
        duty_elapsed / config.CONSTRAINT_NORMS["max_duty"],
        state.get("legs", 0) / config.CONSTRAINT_NORMS["max_legs"],
        duty_period_norm,
        1.0 if state.get("is_resting", False) else 0.0,
        rest_remaining,
    ]
    if include_total_legs:
        scalars.append(state.get("total_legs", 0) / config.CONSTRAINT_NORMS["max_legs"])

    if _SKIP_DECODER_CONSTRAINT:
        c_vec = torch.zeros(len(FILM_CONSTRAINT_KEYS), dtype=torch.float32).to(dev)
    else:
        c_vec = constraint_to_tensor(constraint, device=dev)

    return torch.cat([
        current_emb,
        base_emb,
        torch.tensor(scalars, dtype=torch.float32).to(dev),
        c_vec,
    ])


def state_to_vec_batch(states, encoder, constraint, device=None, include_total_legs=True):
    """Batched version of state_to_vec() -- states: list of state dicts (all sharing
    the same encoder/constraint, as rollout_batch() does). Returns: (B, 79).

    airport_emb lookups are batched (the actual nn.Module call, done once instead of
    B times); the per-state scalar bookkeeping stays a Python loop since it's plain
    arithmetic on dict fields, not something worth vectorizing on its own.
    Phase 1 of experiment/rollout-batch-vectorization -- see
    journal/experiment-plan/rollout-batch-vectorization.md.
    """
    dev = device or torch.device("cpu")
    B = len(states)
    current_airports = torch.tensor(
        [state["current_airport"] for state in states], dtype=torch.long, device=dev
    )
    current_emb = encoder.airport_emb(current_airports)  # (B, 32)
    base_emb = encoder.airport_emb(
        torch.tensor(constraint["base_airport"], device=dev)
    ).unsqueeze(0).expand(B, -1)  # (B, 32)

    rows = []
    for state in states:
        time_of_day      = (state["current_time"] % 24.0) / 24.0
        day_norm         = (state["current_time"] // 24.0) / config.CONSTRAINT_NORMS["max_pairing_days"]
        duty_period_norm = state.get("duty_period", 0) / config.CONSTRAINT_NORMS["max_duty_periods"]

        if state.get("is_resting", False) or state.get("pairing_start", False):
            duty_elapsed = 0.0
        else:
            duty_elapsed = max(0.0, state["current_time"] - state.get("duty_start_time", state["current_time"]))

        if state.get("is_resting", False) and state.get("rest_end_time") is not None:
            rest_remaining = max(0.0, state["rest_end_time"] - state["current_time"]) / config.CONSTRAINT_NORMS["min_rest"]
        else:
            rest_remaining = 0.0

        row = [
            time_of_day,
            day_norm,
            duty_elapsed / config.CONSTRAINT_NORMS["max_duty"],
            state.get("legs", 0) / config.CONSTRAINT_NORMS["max_legs"],
            duty_period_norm,
            1.0 if state.get("is_resting", False) else 0.0,
            rest_remaining,
        ]
        if include_total_legs:
            row.append(state.get("total_legs", 0) / config.CONSTRAINT_NORMS["max_legs"])
        rows.append(row)
    scalars = torch.tensor(rows, dtype=torch.float32, device=dev)  # (B, 7 or 8)

    if _SKIP_DECODER_CONSTRAINT:
        c_vec = torch.zeros(len(FILM_CONSTRAINT_KEYS), dtype=torch.float32, device=dev)
    else:
        c_vec = constraint_to_tensor(constraint, device=dev)
    c_vec = c_vec.unsqueeze(0).expand(B, -1)

    return torch.cat([current_emb, base_emb, scalars, c_vec], dim=1)


def flight_gap_bias(state, flights, constraint, device=None):
    """Compute the decoder gap_bias input g_t(j) of Eq. (7): shape (N+2,), last 2 entries (EndDuty/EndPairing) always 0.

    Only filled with the actual normalized gap at within-duty connection
    points (i.e. not pairing_start/is_resting); zero everywhere else, so
    gap_weight has no effect at those steps.
    """
    dev = device or torch.device("cpu")
    n = len(flights)
    if state.get("pairing_start", False) or state.get("is_resting", False):
        return torch.zeros(n + 2, dtype=torch.float32, device=dev)
    cap = constraint.get("max_conn", config.DEFAULT_CONSTRAINTS["max_conn"])
    current_time = state["current_time"]
    gaps = [min(max(f["dep_time"] - current_time, 0.0), cap) / cap for f in flights]
    return torch.tensor(gaps + [0.0, 0.0], dtype=torch.float32, device=dev)


def flight_gap_bias_batch(states, flights, constraint, device=None):
    """Batched version of flight_gap_bias() -- states: list of state dicts. Returns: (B, N+2).

    Genuinely vectorized (not a per-state Python loop): gaps for all (state, flight)
    pairs are computed as one (B, N) broadcast, then rows for resting/pairing_start
    states are zeroed via a boolean mask -- matches flight_gap_bias()'s per-state
    early-return exactly. Phase 1 of experiment/rollout-batch-vectorization.
    """
    dev = device or torch.device("cpu")
    n = len(flights)
    if len(states) == 0:
        return torch.zeros(0, n + 2, dtype=torch.float32, device=dev)
    cap = constraint.get("max_conn", config.DEFAULT_CONSTRAINTS["max_conn"])
    dep_times = torch.tensor([f["dep_time"] for f in flights], dtype=torch.float32, device=dev)  # (N,)
    current_times = torch.tensor(
        [state["current_time"] for state in states], dtype=torch.float32, device=dev
    )  # (B,)
    gaps = (dep_times.unsqueeze(0) - current_times.unsqueeze(1)).clamp(min=0.0, max=cap) / cap  # (B, N)

    skip = torch.tensor(
        [bool(state.get("pairing_start", False) or state.get("is_resting", False)) for state in states],
        dtype=torch.bool, device=dev,
    )  # (B,)
    gaps = gaps.masked_fill(skip.unsqueeze(1), 0.0)

    tail = torch.zeros(len(states), 2, dtype=torch.float32, device=dev)
    return torch.cat([gaps, tail], dim=1)
