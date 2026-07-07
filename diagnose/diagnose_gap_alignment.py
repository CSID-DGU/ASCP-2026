"""
diagnose_gap_alignment.py — intra-duty gap 진단 (재학습 없이 기존 체크포인트로)

log/0706/FTC_gap_진단_실행계획.md Step1(후보 gap 분포) + Step2(score-gap 상관) 실행 스크립트.

Step1: duty-내부 연결 선택 시점마다 mask 통과 후보들의 gap 중 "선택된 gap == 최솟값"인
       비율을 측정 — 낮으면 "더 짧은 후보가 있었는데 다른 걸 골랐다"는 뜻.
Step2: 같은 시점에서 decoder raw score(logit)와 실제 gap의 Spearman 상관을 측정 —
       약하면 모델이 gap을 잘 구분 못한다는 뜻(FTC_근본원인_모델설계_분석.md 가설 B).
"""
import os
import sys
import argparse

import torch
from torch.distributions import Categorical
from scipy.stats import spearmanr

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "RL"))

from model import FlightEncoder, PointerDecoder
from loader import build_airport_map, bases_to_ids, load_flights_rolling
from constraints import get_delta_constraints, FILM_CONSTRAINT_KEYS
from utils import flights_to_tensors, constraint_to_tensor, state_to_vec
import environment as env
import config


def load_model(checkpoint, device):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
    n_airports = ckpt.get("n_airports", ckpt["encoder"]["airport_emb.weight"].shape[0])
    encoder = FlightEncoder(n_airports=n_airports, constraint_dim=len(FILM_CONSTRAINT_KEYS)).to(device)
    airport_emb_dim = encoder.airport_emb.embedding_dim
    ckpt_state_dim = ckpt["decoder"]["state_mlp.0.weight"].shape[1]
    n_scalars = ckpt_state_dim - airport_emb_dim * 2 - len(FILM_CONSTRAINT_KEYS)
    decoder = PointerDecoder(constraint_dim=len(FILM_CONSTRAINT_KEYS),
                              airport_emb_dim=airport_emb_dim, n_scalars=n_scalars).to(device)
    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])
    encoder.eval()
    decoder.eval()
    return encoder, decoder


def new_pairing_state(f, episode_base, assigned):
    return {
        "current_airport": f["dest"], "current_time": f["arr_time"],
        "duty_time": f["arr_time"] - f["dep_time"], "duty_start_time": f["dep_time"],
        "legs": 1, "remaining": sum(1 for v in assigned.values() if not v),
        "pairing_start": False, "duty_period": 0,
        "pairing_start_time": f["dep_time"], "is_resting": False, "rest_end_time": None,
        "base_airport": episode_base,
    }


def rollout_with_diagnostics(flights, constraint, encoder, decoder, encoded, device, greedy=False):
    """duty-내부 연결 선택마다 (선택 gap, 후보 gap 목록, 후보 score 목록)을 기록."""
    assigned = {f["id"]: False for f in flights}
    records = []

    episode_base = constraint.get("base_airport", 0)
    unassigned = [f for f in flights if not assigned[f["id"]]]
    base_flights = [f for f in unassigned if f["origin"] == episode_base]
    first = sorted(base_flights or unassigned, key=lambda f: f["dep_time"])[0]
    assigned[first["id"]] = True
    state = new_pairing_state(first, episode_base, assigned)
    state["pairing_start"] = False  # 첫 편은 pairing_start가 아니라 첫 편 자체(gap 패널티 없음 케이스)

    _incl_total = decoder.state_mlp[0].weight.shape[1] > 78
    max_steps = len(flights) * 4
    steps = 0
    while steps < max_steps:
        steps += 1
        mask_list = env.get_mask(state, flights, assigned, constraint)
        mask = torch.tensor(mask_list, dtype=torch.float32).to(device)

        if sum(mask_list[:-2]) == 0 and mask_list[-2] == 0 and mask_list[-1] == 0:
            unassigned = [f for f in flights if not assigned[f["id"]]]
            if not unassigned:
                break
            nxt = sorted([f for f in unassigned if f["origin"] == episode_base] or unassigned,
                         key=lambda f: f["dep_time"])[0]
            assigned[nxt["id"]] = True
            state = new_pairing_state(nxt, episode_base, assigned)
            continue

        state_vec = state_to_vec(state, encoder, constraint, device=device, include_total_legs=_incl_total)
        logits = decoder(encoded, state_vec, mask, return_logits=True)
        probs = torch.softmax(logits, dim=-1)

        # 진단 대상: pairing 첫 편도 아니고 rest 직후도 아닌, "같은 duty 내부 연결" 시점만
        is_intra_context = not state.get("pairing_start", False) and not state.get("is_resting", False)
        pending = None
        if is_intra_context:
            valid_idx = [i for i, m in enumerate(mask_list[:-2]) if m == 1]
            if len(valid_idx) >= 2:
                gaps = [flights[i]["dep_time"] - state["current_time"] for i in valid_idx]
                scores = [logits[i].item() for i in valid_idx]
                pending = {"gaps": gaps, "scores": scores}

        if greedy:
            action = probs.argmax().item()
        else:
            action = Categorical(probs).sample().item()

        if action == len(flights):  # END_DUTY
            state, _, _ = env.step(state, action, flights, assigned, constraint)
            continue
        if action == len(flights) + 1:  # END_PAIRING
            unassigned = [f for f in flights if not assigned[f["id"]]]
            if not unassigned:
                break
            nxt = sorted([f for f in unassigned if f["origin"] == episode_base] or unassigned,
                         key=lambda f: f["dep_time"])[0]
            assigned[nxt["id"]] = True
            state = new_pairing_state(nxt, episode_base, assigned)
            continue

        f = flights[action]
        if pending is not None:
            pending["chosen_gap"] = f["dep_time"] - state["current_time"]
            records.append(pending)

        state, _, done = env.step(state, action, flights, assigned, constraint)
        if done:
            break

    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-rollouts", type=int, default=30)
    parser.add_argument("--subset-size", type=int, default=1200)
    args = parser.parse_args()

    device = torch.device(args.device)
    encoder, decoder = load_model(args.checkpoint, device)

    data_path = config.AIRLINE_DATA["delta"]
    airport_map = build_airport_map(data_path)
    base_ids = bases_to_ids(list(config.AIRLINE_BASES["delta"]), airport_map)
    base = base_ids[0]
    constraint = get_delta_constraints(base)

    flights = load_flights_rolling(
        data_path, window_days=5, offset_days=0, airport_map=airport_map,
        base_airport=base, n_max=args.subset_size, use_utc=True,
    )
    max_time = 5 * 24.0
    origins, dests, dep_times, arr_times, fly_times = flights_to_tensors(flights, max_time, device=device)
    c_tensor = constraint_to_tensor(constraint, device=device)

    all_records = []
    with torch.no_grad():
        encoded = encoder(origins, dests, dep_times, arr_times, fly_times, c_tensor)
        for _ in range(args.n_rollouts):
            all_records.extend(
                rollout_with_diagnostics(flights, constraint, encoder, decoder, encoded, device, greedy=False)
            )

    n = len(all_records)
    print(f"checkpoint: {args.checkpoint}")
    print(f"진단 대상 duty-내부 연결 선택 시점: {n}건 ({args.n_rollouts} rollouts)")
    if n == 0:
        print("기록 없음 — subset/제약 조건 확인 필요")
        return

    # Step 1: 선택 gap == 후보 중 최솟값 비율
    is_min_choice = [abs(r["chosen_gap"] - min(r["gaps"])) < 1e-6 for r in all_records]
    min_gaps = [min(r["gaps"]) for r in all_records]
    chosen_gaps = [r["chosen_gap"] for r in all_records]
    print()
    print("=== Step 1: 후보 gap 분포 (가설 A: 데이터 구조) ===")
    print(f"  선택 gap == 후보 최솟값 비율: {sum(is_min_choice)/n*100:.1f}%")
    print(f"  후보 최솟값 평균:   {sum(min_gaps)/n:.2f}h  (중앙값 {sorted(min_gaps)[n//2]:.2f}h)")
    print(f"  실제 선택 gap 평균: {sum(chosen_gaps)/n:.2f}h  (중앙값 {sorted(chosen_gaps)[n//2]:.2f}h)")

    # Step 2: score-gap Spearman 상관 (decision point별로 계산 후 평균)
    corrs = []
    for r in all_records:
        if len(set(r["gaps"])) < 2:  # 후보 gap이 다 같으면 상관 정의 불가
            continue
        rho, _ = spearmanr(r["gaps"], r["scores"])
        if rho == rho:  # NaN 체크
            corrs.append(rho)
    print()
    print("=== Step 2: score-gap 상관 (가설 B: 모델 설계) ===")
    print(f"  상관 계산 가능한 decision point: {len(corrs)}/{n}건")
    if corrs:
        avg_rho = sum(corrs) / len(corrs)
        print(f"  평균 Spearman rho (gap vs score): {avg_rho:.3f}")
        print(f"  (음수면 gap 클수록 score 낮음 = 모델이 gap을 인식하고 있다는 뜻)")
        strong_neg = sum(1 for c in corrs if c <= -0.5)
        print(f"  rho <= -0.5(강한 음의 상관)인 decision point 비율: {strong_neg/len(corrs)*100:.1f}%")


if __name__ == "__main__":
    main()
