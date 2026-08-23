"""
diagnose_anticipatory_constraint_awareness.py — 하드마스크는 그대로 유지한 채(=
feasibility 100% 보장), 마스킹으로 액션이 사라지기 "직전"의 원본(마스킹 전) 확률
분포를 봐서 모델이 곧 규정 위반이 될 flight을 스스로 회피하는지(anticipatory
avoidance)를 측정한다.

배경(log/0717/FiLM_방향결정_및_계획.md §2 옵션 C):
  film-chanju(하드마스크 제거 + soft penalty)는 재학습 결과 규정 위반율
  60~70%로 사실상 실패했다(별도 진단 diagnose_softmask_violations.py로 확인).
  soft penalty 재설계+재학습(옵션 B)은 비용이 크고 성공 보장이 없어 보류하고,
  대신 재학습 없이 기존 체크포인트로 "모델이 규정을 실제로 인지하고 있는가"를
  검증하는 이 저비용 실험을 채택했다.

방법:
  1. 실제 rollout은 항상 원본(하드마스크) get_mask/step으로 진행 — feasibility는
     100% 유지된다(film-chanju처럼 실제로 위반된 액션을 선택하게 두지 않음).
  2. 매 flight 선택 스텝에서, "물리적으로만 유효한" 마스크(physical-only mask —
     항공사 규정성 제약은 걸지 않고 공항 연결·시간 순서·rest 상태만 체크,
     film-chanju가 하드마스크에서 뺐던 항목들과 동일한 기준)를 추가로 계산해서
     디코더에 흘려보내 "규정이 없다면 모델이 어떤 확률을 줬을지"를 읽는다.
  3. "물리적으로는 갈 수 있지만 규정상 하드마스크에 걸려 실제로는 막힌 flight"
     집합에 대해, 모델이 배정한 확률 질량(mass ratio)이 그 flight들의 개수 비율
     (count ratio, "차별 없이 균등하게 골랐다면 받았을 몫")보다 낮은지를 본다.
     count_ratio - mass_ratio = "회피 격차"(avoidance gap) — 클수록 모델이 곧
     무효화될 선택지를 스스로 피하고 있다는 뜻(=규정을 선제적으로 인지).
  4. 정상 체크포인트 vs FiLM 무력화(diagnose_film_inference_ablation.py와 동일한
     identity 우회) 두 모드에서 회피 격차를 비교 — FiLM 무력화 시 격차가 뚜렷이
     줄어들면 FiLM의 인과 기여를 뒷받침하는 추가 증거.

Usage:
    cd /home/hyrn/ASCP-2026
    source ascp/bin/activate
    python -u diagnose/diagnose_anticipatory_constraint_awareness.py checkpoints/pws5cjlz/stage3_best.pt \
        --device cpu --n-rollouts 6
"""
import os
import sys
import argparse

import numpy as np
import torch
from torch.distributions import Categorical

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "RL"))

import config
import environment as env  # 하드마스크 원본 — 실제 rollout 진행에 사용
from environment import get_max_duty
from loader import validate_airport_map, bases_to_ids, load_flights_rolling, sample_connected_subnet
from constraints import get_delta_constraints, get_alaska_constraints, get_jetblue_constraints, FILM_CONSTRAINT_KEYS
from utils import constraint_to_tensor, flights_to_tensors, state_to_vec, flight_gap_bias
from model import FlightEncoder, PointerDecoder

_GET_CONSTRAINT = {
    "delta": get_delta_constraints,
    "alaska": get_alaska_constraints,
    "jetblue": get_jetblue_constraints,
}


def get_mask_physical_only(state, flights, assigned, constraint=None, stage=3):
    """film-chanju(68cf35c)가 채택했던 '물리적 제약만' 마스크 — 규정성 제약
    (max_legs/max_duty/min_conn·max_conn/max_pairing_days/max_duty_periods)은
    걸지 않는다. 실제 환경 진행에는 쓰지 않고, 디코더의 '규정이 없다면 어떤
    확률을 줬을지' 원본 분포를 읽어내는 용도로만 사용."""
    c = constraint if constraint else config.DEFAULT_CONSTRAINTS
    stage_rule = config.CURRICULUM_CONFIG.get(stage, config.CURRICULUM_CONFIG[3])
    N = len(flights)
    mask = np.zeros(N + 2, dtype=np.int32)

    pairing_start = state.get("pairing_start", False)
    is_resting    = state.get("is_resting", False)
    rest_end      = state.get("rest_end_time", 0.0)
    base_ap = c.get("base_airport", config.DEFAULT_CONSTRAINTS["base_airport"])

    base_remaining = False
    if pairing_start:
        base_remaining = any(not assigned[fl["id"]] and fl["origin"] == base_ap for fl in flights)

    for i, f in enumerate(flights):
        if assigned[f["id"]]:
            continue
        valid = True
        if pairing_start:
            if base_remaining and f["origin"] != base_ap:
                valid = False
        elif f["origin"] != state["current_airport"]:
            valid = False

        if is_resting:
            if f["dep_time"] < rest_end:
                valid = False
        elif not pairing_start:
            if f["dep_time"] < state["current_time"]:
                valid = False

        if valid:
            mask[i] = 1

    can_end_duty = (
        stage_rule["allow_end_duty"] and state.get("legs", 0) > 0
        and not is_resting and not pairing_start
    )
    if can_end_duty:
        mask[config.END_DUTY] = 1

    min_pairing_legs = c.get("min_pairing_legs", 2)
    if state.get("total_legs", 0) >= min_pairing_legs:
        mask[config.END_PAIRING] = 1

    return mask.tolist()


def bypass_film(encoder):
    identity = lambda flight_vecs, constraint: flight_vecs
    encoder.film_before.forward = identity
    encoder.film_after.forward = identity


def load_model(checkpoint, device):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
    n_airports = ckpt.get("n_airports", ckpt["encoder"]["airport_emb.weight"].shape[0])
    encoder = FlightEncoder(n_airports=n_airports, constraint_dim=len(FILM_CONSTRAINT_KEYS)).to(device)
    airport_emb_dim = encoder.airport_emb.embedding_dim
    ckpt_state_dim = ckpt["decoder"]["state_mlp.0.weight"].shape[1]
    n_scalars = ckpt_state_dim - airport_emb_dim * 2 - len(FILM_CONSTRAINT_KEYS)
    decoder = PointerDecoder(constraint_dim=len(FILM_CONSTRAINT_KEYS), airport_emb_dim=airport_emb_dim, n_scalars=n_scalars).to(device)
    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])
    encoder.eval(); decoder.eval()
    return encoder, decoder, n_airports


def rollout_and_measure(flights, constraint, encoder, decoder, encoded, greedy, device):
    """실제 진행은 하드마스크(env.get_mask/env.step)로 — feasibility 100% 유지.
    매 flight 선택 스텝마다 physical-only 마스크로 원본 분포를 추가로 읽어
    회피 격차 표본을 수집한다."""
    assigned = {f["id"]: False for f in flights}
    N = len(flights)

    gaps = []  # (count_ratio, mass_ratio) 표본

    episode_base = constraint.get("base_airport", 0)
    base_flights = [f for f in flights if f["origin"] == episode_base]
    first = sorted(base_flights or flights, key=lambda f: f["dep_time"])[0]
    assigned[first["id"]] = True
    state = {
        "current_airport": first["dest"], "current_time": first["arr_time"],
        "duty_time": first["arr_time"] - first["dep_time"], "duty_start_time": first["dep_time"],
        "legs": 1, "total_legs": 1, "remaining": sum(1 for v in assigned.values() if not v),
        "pairing_start": False, "duty_period": 0, "pairing_start_time": first["dep_time"],
        "is_resting": False, "rest_end_time": None, "base_airport": episode_base,
    }

    incl_total = decoder.state_mlp[0].weight.shape[1] > 78

    for _ in range(N * 6):
        mask_list = env.get_mask(state, flights, assigned, constraint)
        if sum(mask_list[:-2]) == 0 and mask_list[-2] == 0 and mask_list[-1] == 0:
            unassigned = [f for f in flights if not assigned[f["id"]]]
            if not unassigned:
                break
            base_flights = [f for f in unassigned if f["origin"] == episode_base]
            nxt = sorted(base_flights or unassigned, key=lambda f: f["dep_time"])[0]
            assigned[nxt["id"]] = True
            state = {
                "current_airport": nxt["dest"], "current_time": nxt["arr_time"],
                "duty_time": nxt["arr_time"] - nxt["dep_time"], "duty_start_time": nxt["dep_time"],
                "legs": 1, "total_legs": 1, "remaining": sum(1 for v in assigned.values() if not v),
                "pairing_start": False, "duty_period": 0, "pairing_start_time": nxt["dep_time"],
                "is_resting": False, "rest_end_time": None, "base_airport": episode_base,
            }
            continue

        # --- 진단: physical-only 마스크로 원본 분포 읽기 (환경 진행에는 영향 없음) ---
        phys_mask_list = get_mask_physical_only(state, flights, assigned, constraint)
        invalid_flight_idx = [i for i in range(N)
                               if phys_mask_list[i] == 1 and mask_list[i] == 0]
        valid_flight_idx = [i for i in range(N) if phys_mask_list[i] == 1]
        if len(valid_flight_idx) > 0 and len(invalid_flight_idx) > 0:
            phys_mask_t = torch.tensor(phys_mask_list, dtype=torch.float32).to(device)
            svec = state_to_vec(state, encoder, constraint, device=device, include_total_legs=incl_total)
            gbias = flight_gap_bias(state, flights, constraint, device=device)
            with torch.no_grad():
                probs_soft = decoder(encoded, svec, phys_mask_t, gap_bias=gbias)
            flight_mass_total = probs_soft[valid_flight_idx].sum().item()
            invalid_mass = probs_soft[invalid_flight_idx].sum().item()
            if flight_mass_total > 1e-8:
                mass_ratio = invalid_mass / flight_mass_total
                count_ratio = len(invalid_flight_idx) / len(valid_flight_idx)
                gaps.append((count_ratio, mass_ratio))

        # --- 실제 진행: 하드마스크로 액션 선택 (feasibility 보장) ---
        mask = torch.tensor(mask_list, dtype=torch.float32).to(device)
        svec = state_to_vec(state, encoder, constraint, device=device, include_total_legs=incl_total)
        gbias = flight_gap_bias(state, flights, constraint, device=device)
        with torch.no_grad():
            probs = decoder(encoded, svec, mask, gap_bias=gbias)
        action = probs.argmax().item() if greedy else Categorical(probs).sample().item()

        if action == N:  # END_DUTY
            state, _, _ = env.step(state, action, flights, assigned, constraint)
            continue
        if action == N + 1:  # END_PAIRING
            unassigned = [f for f in flights if not assigned[f["id"]]]
            if not unassigned:
                break
            base_flights = [f for f in unassigned if f["origin"] == episode_base]
            nxt = sorted(base_flights or unassigned, key=lambda f: f["dep_time"])[0]
            assigned[nxt["id"]] = True
            state = {
                "current_airport": nxt["dest"], "current_time": nxt["arr_time"],
                "duty_time": nxt["arr_time"] - nxt["dep_time"], "duty_start_time": nxt["dep_time"],
                "legs": 1, "total_legs": 1, "remaining": sum(1 for v in assigned.values() if not v),
                "pairing_start": False, "duty_period": 0, "pairing_start_time": nxt["dep_time"],
                "is_resting": False, "rest_end_time": None, "base_airport": episode_base,
            }
            continue

        f = flights[action]
        assigned[f["id"]] = True
        state, _, done = env.step(state, action, flights, assigned, constraint)
        if done:
            break

    return gaps


def evaluate_mode(label, encoder, decoder, airline, base_ids, window_flights, constraint,
                   subset_size, n_rollouts, device, max_time):
    all_gaps = []
    for r in range(n_rollouts):
        subset = sample_connected_subnet(window_flights, base_ids[0], subset_size) or \
                 sorted(window_flights, key=lambda f: f["dep_time"])[:subset_size]
        for i, f in enumerate(subset):
            f["id"] = i
        origins, dests, dep_norm, arr_norm, fly_norm = flights_to_tensors(subset, max_time, device=device)
        c_tensor = constraint_to_tensor(constraint, device=device)
        with torch.no_grad():
            encoded = encoder(origins, dests, dep_norm, arr_norm, fly_norm, c_tensor)
        greedy = (r == n_rollouts - 1)
        gaps = rollout_and_measure(subset, constraint, encoder, decoder, encoded, greedy, device)
        all_gaps.extend(gaps)

    if not all_gaps:
        print(f"  [{label}] {airline}: 유효 표본 없음 (규정상 구별되는 상황이 발생 안 함)")
        return None
    count_ratios = np.array([g[0] for g in all_gaps])
    mass_ratios  = np.array([g[1] for g in all_gaps])
    avoidance = count_ratios - mass_ratios
    print(f"  [{label}] {airline}: n={len(all_gaps)}  "
          f"count_ratio(평균)={count_ratios.mean():.3f}  mass_ratio(평균)={mass_ratios.mean():.3f}  "
          f"회피격차(평균)={avoidance.mean():+.3f}  회피격차>0 비율={100*np.mean(avoidance>0):.1f}%")
    return avoidance.mean()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--window-days", type=int, default=5)
    ap.add_argument("--subset-size", type=int, default=config.EPISODE_MAX_FLIGHTS)
    ap.add_argument("--n-rollouts", type=int, default=6)
    args = ap.parse_args()

    device = torch.device(args.device)
    max_time = args.window_days * 24.0

    map_ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    map_n_airports = map_ckpt.get("n_airports", map_ckpt["encoder"]["airport_emb.weight"].shape[0])
    airport_map = validate_airport_map(map_ckpt.get("airport_map"), map_n_airports)

    summary = {}
    for mode, label in [("normal", "정상"), ("bypassed", "FiLM 무력화")]:
        print(f"\n{'='*70}\n{label}\n{'='*70}")
        encoder, decoder, n_airports = load_model(args.checkpoint, device)
        if mode == "bypassed":
            bypass_film(encoder)
        summary[mode] = {}
        for airline in ["delta", "alaska", "jetblue"]:
            base_ids = bases_to_ids(list(config.AIRLINE_BASES[airline]), airport_map)
            constraint = _GET_CONSTRAINT[airline](base_ids[0])
            window_flights = load_flights_rolling(
                config.AIRLINE_DATA[airline], window_days=args.window_days, offset_days=0,
                airport_map=airport_map,
            )
            for i, f in enumerate(window_flights):
                f["id"] = i
            avg_gap = evaluate_mode(label, encoder, decoder, airline, base_ids, window_flights,
                                     constraint, args.subset_size, args.n_rollouts, device, max_time)
            summary[mode][airline] = avg_gap

    print(f"\n\n{'='*70}\n최종 비교 (회피격차 평균, 클수록 규정 선제 회피)\n{'='*70}")
    print(f"{'항공사':<10}{'정상':>12}{'FiLM 무력화':>14}{'차이':>10}")
    for airline in ["delta", "alaska", "jetblue"]:
        n = summary["normal"][airline]
        b = summary["bypassed"][airline]
        if n is not None and b is not None:
            print(f"{airline:<10}{n:>12.4f}{b:>14.4f}{n-b:>10.4f}")


if __name__ == "__main__":
    main()
