"""
diagnose_film_overnight.py — FiLM이 max_duty_periods에 반응해 overnight을 실제로 조절하는지 확인.

experiments/train.py의 _film_validation은 max_duty_periods 스윕에서 avg_overnight을
로깅하지 않는다 (max_legs 스윕에서만 로깅). 이 스크립트는 기존 checkpoint로 같은 실험을
재현하되 avg_overnight까지 함께 출력한다.
"""
import sys
import argparse

import torch

sys.path.insert(0, "RL")

from model import FlightEncoder, PointerDecoder
from loader import build_airport_map, bases_to_ids, load_flights_rolling
from constraints import get_delta_constraints, FILM_CONSTRAINT_KEYS
from utils import flights_to_tensors, constraint_to_tensor
import config

sys.path.insert(0, "experiments")
import train as train_mod
from train import run_episode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-rollouts", type=int, default=10)
    args = parser.parse_args()

    device = torch.device(args.device)
    train_mod.DEVICE = device
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    n_airports = ckpt.get("n_airports", ckpt["encoder"]["airport_emb.weight"].shape[0])

    data_path = config.AIRLINE_DATA["delta"]
    airport_map = build_airport_map(data_path)
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

    flights = load_flights_rolling(
        data_path, window_days=5, offset_days=0, airport_map=airport_map,
        base_airport=base, n_max=config.EPISODE_MAX_FLIGHTS,
    )
    origins, dests, dep_times, arr_times, fly_times = flights_to_tensors(flights, 5 * 24.0, device=device)

    print(f"checkpoint: {args.checkpoint}")
    print(f"  [max_duty_periods 변화] (합격 기준: dp=1→4 pairings ≥30% 감소, avg_overnight도 함께 증가해야 함)")
    with torch.no_grad():
        for dp in [1, 2, 3, 4]:
            val_c = {**get_delta_constraints(base), "max_duty_periods": dp, "max_pairing_days": 5}
            val_enc = encoder(origins, dests, dep_times, arr_times, fly_times,
                               constraint_to_tensor(val_c, device=device))
            p_list, dh_list, cov_list, on_list, legs_list = [], [], [], [], []
            for _ in range(args.n_rollouts):
                _, _, _, m = run_episode(flights, val_c, encoder, decoder, val_enc, greedy=False)
                p_list.append(m["n_pairings"])
                dh_list.append(m["n_deadheads"])
                cov_list.append(m["coverage_pct"])
                on_list.append(m.get("avg_overnight", 0))
                legs_list.append(m.get("avg_legs", 0))
            n = args.n_rollouts
            print(f"    max_duty_periods={dp} → "
                  f"pairings(avg{n})={sum(p_list)/n:.1f}  "
                  f"deadheads={sum(dh_list)/n:.1f}  "
                  f"coverage={sum(cov_list)/n:.1f}%  "
                  f"avg_overnight={sum(on_list)/n:.3f}  "
                  f"avg_legs={sum(legs_list)/n:.3f}")


if __name__ == "__main__":
    main()
