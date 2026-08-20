"""
diagnose_film_crossairline.py -- swap in the full delta/alaska/jetblue constraint sets
on the same delta flight data to check whether pairing behavior actually changes.

diagnose_film_overnight.py는 delta 안에서 max_duty_periods 성분 하나만 흔들었는데,
이건 FiLM의 존재 목적(항공사 간 일반화)을 검증 못 한다 — 성분 하나가 아니라 항공사
전체 constraint 세트를 통째로 바꿔서 비교해야 함. multi-airline 체크포인트(168 공항
통합 임베딩)에만 의미 있음 — delta 단독 체크포인트에 alaska/jetblue를 넣으면 airport
ID가 우연히 겹칠 뿐이라 무의미(evaluation/evaluate_ip.py의 n_airports>145 분기 참고).
"""
import os
import sys
import argparse

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "RL"))

from model import FlightEncoder, PointerDecoder
from loader import build_airport_map, bases_to_ids, load_flights_rolling
from constraints import (
    get_delta_constraints, get_alaska_constraints, get_jetblue_constraints,
    get_turkish_constraints, FILM_CONSTRAINT_KEYS,
)
from utils import flights_to_tensors, constraint_to_tensor
import config

sys.path.insert(0, os.path.join(REPO_ROOT, "experiments"))
import train as train_mod
from train import run_episode

AIRLINES = {
    "delta":   get_delta_constraints,
    "alaska":  get_alaska_constraints,
    "jetblue": get_jetblue_constraints,
    "turkish": get_turkish_constraints,  # for reference values only -- turkish flight/embedding is not used (experiment 2)
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-rollouts", type=int, default=10)
    parser.add_argument("--greedy", action="store_true", help="use greedy (deterministic) rollout instead of stochastic -- for removing noise")
    args = parser.parse_args()

    device = torch.device(args.device)
    train_mod.DEVICE = device
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    n_airports = ckpt.get("n_airports", ckpt["encoder"]["airport_emb.weight"].shape[0])
    if n_airports <= 145:
        print(f"warning: n_airports={n_airports} -- this looks like a single-airline "
              f"checkpoint rather than a multi-airline unified embedding (168). "
              f"alaska/jetblue constraint results may be meaningless.")

    # Rebuild the same unified airport map used during multi-airline training (delta+alaska+jetblue)
    map_paths = [v for k, v in config.AIRLINE_DATA.items() if k != "turkish"]
    airport_map = build_airport_map(map_paths)
    base_ids = bases_to_ids(list(config.AIRLINE_BASES["delta"]), airport_map)
    base = base_ids[0]

    encoder = FlightEncoder(n_airports=n_airports, constraint_dim=len(FILM_CONSTRAINT_KEYS)).to(device)
    airport_emb_dim = encoder.airport_emb.embedding_dim
    ckpt_state_dim = ckpt["decoder"]["state_mlp.0.weight"].shape[1]
    n_scalars = ckpt_state_dim - airport_emb_dim * 2 - len(FILM_CONSTRAINT_KEYS)
    decoder = PointerDecoder(constraint_dim=len(FILM_CONSTRAINT_KEYS),
                              airport_emb_dim=airport_emb_dim, n_scalars=n_scalars).to(device)
    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])
    encoder.eval(); decoder.eval()

    # Flight data is fixed to delta -- the key point is comparing while only swapping constraints per airline
    flights = load_flights_rolling(
        config.AIRLINE_DATA["delta"], window_days=5, offset_days=0, airport_map=airport_map,
        base_airport=base, n_max=config.EPISODE_MAX_FLIGHTS,
    )
    origins, dests, dep_times, arr_times, fly_times = flights_to_tensors(flights, 5 * 24.0, device=device)

    print(f"checkpoint: {args.checkpoint}  (n_airports={n_airports})")
    print(f"flight data: delta fixed, {len(flights)} flights -- only constraints swapped per airline")
    print(f"{'airline':<10}{'pairings':>10}{'deadheads':>11}{'coverage':>10}{'avg_overnight':>15}{'avg_legs':>10}")
    with torch.no_grad():
        for airline, get_fn in AIRLINES.items():
            val_c = get_fn(base)
            val_enc = encoder(origins, dests, dep_times, arr_times, fly_times,
                               constraint_to_tensor(val_c, device=device))
            p_list, dh_list, cov_list, on_list, legs_list = [], [], [], [], []
            n_reps = 1 if args.greedy else args.n_rollouts
            for _ in range(n_reps):
                _, _, _, m = run_episode(flights, val_c, encoder, decoder, val_enc, greedy=args.greedy)
                p_list.append(m["n_pairings"])
                dh_list.append(m["n_deadheads"])
                cov_list.append(m["coverage_pct"])
                on_list.append(m.get("avg_overnight", 0))
                legs_list.append(m.get("avg_legs", 0))
            n = n_reps
            print(f"{airline:<10}{sum(p_list)/n:>10.1f}{sum(dh_list)/n:>11.1f}"
                  f"{sum(cov_list)/n:>9.1f}%{sum(on_list)/n:>15.3f}{sum(legs_list)/n:>10.3f}")

    print()
    print("constraint values for reference:")
    for airline, get_fn in AIRLINES.items():
        c = get_fn(base)
        print(f"  {airline}: " + ", ".join(f"{k}={c[k]}" for k in FILM_CONSTRAINT_KEYS))


if __name__ == "__main__":
    main()
