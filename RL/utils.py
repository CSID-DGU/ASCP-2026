# utils.py — experiments/train.py와 evaluation/evaluate_ip.py 공용 유틸
#
# 중복 제거 대상:
#   - constraint_to_tensor: constraint dict → FiLM 입력 tensor
#   - flights_to_tensors: flight dict 리스트 → 모델 입력 tensor
#   - state_to_vec: environment state dict → decoder 입력 tensor

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
    """constraint dict → FiLM 입력 tensor (constraint_dim,)

    CONSTRAINT_NORMS로 정규화 → [0, 1] 범위.
    """
    t = torch.tensor(
        [constraint[k] / config.CONSTRAINT_NORMS[k] for k in FILM_CONSTRAINT_KEYS],
        dtype=torch.float32,
    )
    if device is not None:
        t = t.to(device)
    return t


def flights_to_tensors(flights, max_time, device=None):
    """flight dict 리스트 → (origins, dests, dep_times, arr_times, fly_times) tensor.

    max_time: 시간 정규화 분모. train에서는 window_days*24, eval에서는 ckpt["max_time"].
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
    """environment state dict → decoder 입력 tensor (79,)

    state_vec = current_airport_emb(32) + base_airport_emb(32) + scalars(8) + constraint_vec(7)

    scalars 8개:
      time_of_day, day_norm, duty_elapsed/NORM, legs/NORM,
      duty_period/NORM, is_resting, rest_remaining, total_legs/NORM

    constraint_vec: FILM_CONSTRAINT_KEYS 정규화값 — decoder가 constraint를 직접 볼 수 있게 함.

    고정 분모(CONSTRAINT_NORMS) 사용 이유:
      constraint 값으로 직접 나누면 constraint 정보가 state_vec에 인코딩되어 FiLM gradient가 약해짐.
      CONSTRAINT_NORMS는 훈련 범위 최대값으로 고정 → FiLM이 constraint 정보의 유일한 경로가 됨.
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


def flight_gap_bias(state, flights, constraint, device=None):
    """decoder gap_bias 입력 — (N+2,), 마지막 2개(END_DUTY/END_PAIRING)는 항상 0.

    duty-내부 연결 시점(pairing_start/is_resting 아닐 때)에만 실제 gap을 채우고,
    그 외에는 전부 0(gap_weight가 안 걸림).
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
    """flight_gap_bias의 배치 버전 — states: state dict 리스트. 반환: (B, N+2)"""
    dev = device or torch.device("cpu")
    return torch.stack([flight_gap_bias(s, flights, constraint, device=dev) for s in states])
