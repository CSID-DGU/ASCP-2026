"""
diagnose_softmask_violations.py — film-chanju(hard mask 제거 + soft penalty,
커밋 68cf35c) 학습 체크포인트가 실제로 항공사 규정(max_legs/max_duty/
min_conn·max_conn/max_pairing_days/max_duty_periods)을 얼마나 위반하는지 직접
측정한다.

배경(log/0717/FiLM_방향결정_및_계획.md §1):
  기존 evaluate_ip.py/rollout.py는 pairing pool을 만들 때 위반 여부를 전혀
  체크하지 않는다(마스크가 없는 지금은 위반된 pairing도 그대로 pool에 들어가
  IP가 선택할 수 있음). 여기서는 rollout을 직접 실행하며 매 flight 선택/
  END_DUTY 시점마다 "만약 하드 마스크가 있었다면 막혔을 상황인가"를 계산해
  위반 건수·심각도를 센다.

  이 스크립트는 이 저장소(하드마스크 유지 버전)의 environment.py를 그대로
  써도 동작한다 — 위반 판정 로직은 film-chanju의 _flight_penalties/
  _end_duty_penalty와 동일한 기준을 이 스크립트 안에 그대로 재현해뒀을 뿐,
  film-chanju의 soft-penalty 코드 자체에 의존하지 않는다. film-chanju
  체크포인트(다른 브랜치에서 학습됨)를 이 저장소 코드로 평가하려면
  --checkpoint에 해당 .pt 경로만 넘기면 된다.

Usage:
    cd /home/hyrn/ASCP-2026
    source ascp/bin/activate
    python -u diagnose/diagnose_softmask_violations.py \
        --checkpoint <film-chanju로 학습한 phase2_best.pt 경로> \
        --airline delta --subset-size 600 --n-rollouts 6

    # 대조군(하드마스크로 학습된 체크포인트, 위반율 0%가 나와야 정상):
    python -u diagnose/diagnose_softmask_violations.py \
        --checkpoint checkpoints/pws5cjlz/phase2_best.pt --airline delta
"""
import sys, os, argparse

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "RL"))

import torch
from torch.distributions import Categorical

import config
import environment as env
from environment import get_max_duty
from loader import build_airport_map, bases_to_ids, load_flights_rolling, sample_connected_subnet
from constraints import get_delta_constraints, get_alaska_constraints, get_jetblue_constraints, FILM_CONSTRAINT_KEYS
from utils import constraint_to_tensor, flights_to_tensors, state_to_vec, flight_gap_bias
from model import FlightEncoder, PointerDecoder

_GET_CONSTRAINT = {
    "delta": get_delta_constraints,
    "alaska": get_alaska_constraints,
    "jetblue": get_jetblue_constraints,
}

DEVICE = torch.device("cpu")


def load_model(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    n_airports = ckpt.get("n_airports", ckpt["encoder"]["airport_emb.weight"].shape[0])
    encoder = FlightEncoder(n_airports=n_airports, constraint_dim=len(FILM_CONSTRAINT_KEYS)).to(DEVICE)
    airport_emb_dim = encoder.airport_emb.embedding_dim
    ckpt_state_dim = ckpt["decoder"]["state_mlp.0.weight"].shape[1]
    n_scalars = ckpt_state_dim - airport_emb_dim * 2 - len(FILM_CONSTRAINT_KEYS)
    decoder = PointerDecoder(constraint_dim=len(FILM_CONSTRAINT_KEYS), airport_emb_dim=airport_emb_dim, n_scalars=n_scalars).to(DEVICE)
    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])
    encoder.eval()
    decoder.eval()
    return encoder, decoder


def rollout_and_check(flights, constraint, encoder, decoder, encoded, greedy, max_time):
    """rollout 1회 실행하며 매 스텝 위반 여부 체크. 위반 카운트/초과분 dict 반환."""
    assigned = {f["id"]: False for f in flights}
    N = len(flights)

    viol = dict(max_legs=0, min_conn=0, max_conn=0, max_duty=0, max_pairing_days=0, max_duty_periods=0)
    excess = dict(max_legs=[], max_conn=[], max_duty=[])
    n_flight_steps = 0
    n_end_duty = 0
    n_pairings = 0

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

    max_steps = N * 6
    for _ in range(max_steps):
        mask_list = env.get_mask(state, flights, assigned, constraint)
        if sum(mask_list[:-2]) == 0 and mask_list[-2] == 0 and mask_list[-1] == 0:
            unassigned = [f for f in flights if not assigned[f["id"]]]
            if not unassigned:
                break
            n_pairings += 1
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

        mask = torch.tensor(mask_list, dtype=torch.float32).to(DEVICE)
        incl_total = decoder.state_mlp[0].weight.shape[1] > 78
        svec = state_to_vec(state, encoder, constraint, device=DEVICE, include_total_legs=incl_total)
        gbias = flight_gap_bias(state, flights, constraint, device=DEVICE)
        probs = decoder(encoded, svec, mask, gap_bias=gbias)
        action = probs.argmax().item() if greedy else Categorical(probs).sample().item()

        if action == N:  # END_DUTY
            n_end_duty += 1
            if state.get("duty_period", 0) >= constraint.get("max_duty_periods", config.DEFAULT_CONSTRAINTS["max_duty_periods"]):
                viol["max_duty_periods"] += 1
            state, _, _ = env.step(state, action, flights, assigned, constraint)
            continue

        if action == N + 1:  # END_PAIRING
            n_pairings += 1
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

        # flight 선택
        f = flights[action]
        n_flight_steps += 1

        if not state.get("pairing_start", False) and not state.get("is_resting", False):
            gap = f["dep_time"] - state["current_time"]
            if gap < constraint.get("min_conn", config.DEFAULT_CONSTRAINTS["min_conn"]):
                viol["min_conn"] += 1
            elif gap > constraint.get("max_conn", config.DEFAULT_CONSTRAINTS["max_conn"]):
                viol["max_conn"] += 1
                excess["max_conn"].append(gap - constraint.get("max_conn", config.DEFAULT_CONSTRAINTS["max_conn"]))

        next_legs = state.get("legs", 0) + 1
        if next_legs > constraint.get("max_legs", config.DEFAULT_CONSTRAINTS["max_legs"]):
            viol["max_legs"] += 1
            excess["max_legs"].append(next_legs - constraint.get("max_legs", config.DEFAULT_CONSTRAINTS["max_legs"]))

        d_start_time = f["dep_time"] if (state.get("pairing_start", False) or state.get("is_resting", False)) else state["duty_start_time"]
        eff_max_duty = get_max_duty(next_legs, constraint.get("max_duty"))
        if f["arr_time"] - d_start_time > eff_max_duty:
            viol["max_duty"] += 1
            excess["max_duty"].append(f["arr_time"] - d_start_time - eff_max_duty)

        p_start_time = f["dep_time"] if state.get("pairing_start", False) else state["pairing_start_time"]
        pairing_elapsed = (f["arr_time"] - p_start_time) / 24.0
        if pairing_elapsed > constraint.get("max_pairing_days", config.DEFAULT_CONSTRAINTS["max_pairing_days"]):
            viol["max_pairing_days"] += 1

        assigned[f["id"]] = True
        state, _, done = env.step(state, action, flights, assigned, constraint)
        if done:
            n_pairings += 1
            break

    return viol, excess, n_flight_steps, n_end_duty, n_pairings


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--airline", required=True, choices=["delta", "alaska", "jetblue"])
    ap.add_argument("--subset-size", type=int, default=600)
    ap.add_argument("--window-days", type=int, default=5)
    ap.add_argument("--n-rollouts", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    import random
    random.seed(args.seed)

    data_path = config.AIRLINE_DATA[args.airline]
    encoder, decoder = load_model(args.checkpoint)
    n_airports = encoder.airport_emb.num_embeddings

    if n_airports > 100:
        map_paths = [v for k, v in config.AIRLINE_DATA.items() if k != "turkish"]
    else:
        map_paths = data_path
    airport_map = build_airport_map(map_paths)
    base_ids = bases_to_ids(list(config.AIRLINE_BASES[args.airline]), airport_map)
    constraint = _GET_CONSTRAINT[args.airline](base_ids[0])

    window_flights = load_flights_rolling(data_path, window_days=args.window_days, offset_days=0, airport_map=airport_map, use_utc=True)
    for i, f in enumerate(window_flights):
        f["id"] = i
    subset = sample_connected_subnet(window_flights, base_ids[0], args.subset_size)
    if not subset:
        subset = sorted(window_flights, key=lambda f: f["dep_time"])[:args.subset_size]
    for i, f in enumerate(subset):
        f["id"] = i

    max_time = args.window_days * 24.0
    origins, dests, dep_norm, arr_norm, fly_norm = flights_to_tensors(subset, max_time, device=DEVICE)
    c_tensor = constraint_to_tensor(constraint, device=DEVICE)
    with torch.no_grad():
        encoded = encoder(origins, dests, dep_norm, arr_norm, fly_norm, c_tensor)

    total_viol = dict(max_legs=0, min_conn=0, max_conn=0, max_duty=0, max_pairing_days=0, max_duty_periods=0)
    total_excess = dict(max_legs=[], max_conn=[], max_duty=[])
    total_flight_steps = 0
    total_end_duty = 0
    total_pairings = 0

    with torch.no_grad():
        for r in range(args.n_rollouts):
            greedy = (r == args.n_rollouts - 1)
            viol, exc, nfs, ned, npg = rollout_and_check(subset, constraint, encoder, decoder, encoded, greedy, max_time)
            for k in total_viol:
                total_viol[k] += viol[k]
            for k in total_excess:
                total_excess[k].extend(exc[k])
            total_flight_steps += nfs
            total_end_duty += ned
            total_pairings += npg

    print(f"\n=== {args.airline} | {os.path.basename(args.checkpoint)} | subset={len(subset)}편 | rollouts={args.n_rollouts} ===")
    print(f"constraint: max_legs={constraint['max_legs']} max_duty={constraint['max_duty']} "
          f"min_conn={constraint['min_conn']} max_conn={constraint['max_conn']} "
          f"max_duty_periods={constraint['max_duty_periods']} max_pairing_days={constraint['max_pairing_days']}")
    print(f"총 flight 선택 스텝: {total_flight_steps}, 총 END_DUTY: {total_end_duty}, 총 pairing: {total_pairings}")
    for k, v in total_viol.items():
        denom = total_end_duty if k == "max_duty_periods" else total_flight_steps
        pct = 100.0 * v / denom if denom else 0.0
        exc = total_excess.get(k)
        exc_str = f", 초과분 평균={sum(exc)/len(exc):.2f} 최대={max(exc):.2f}" if exc else ""
        print(f"  위반[{k}]: {v} / {denom} ({pct:.3f}%){exc_str}")


if __name__ == "__main__":
    main()
