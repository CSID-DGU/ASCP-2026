"""
diagnose_film_crossairline.py — 같은 delta flight 데이터에 delta/alaska/jetblue의
constraint를 그대로 바꿔 넣어서 pairing 행동이 실제로 달라지는지 확인.

diagnose_film_overnight.py는 delta 안에서 max_duty_periods 성분 하나만 흔들었는데,
이건 FiLM의 존재 목적(항공사 간 일반화)을 검증 못 한다 — 성분 하나가 아니라 항공사
전체 constraint 세트를 통째로 바꿔서 비교해야 함. multi-airline 체크포인트(168 공항
통합 임베딩)에만 의미 있음 — delta 단독 체크포인트에 alaska/jetblue를 넣으면 airport
ID가 우연히 겹칠 뿐이라 무의미(evaluate_ip.py의 n_airports>145 분기 참고).
"""
import sys
import argparse

import torch

sys.path.insert(0, "RL")

from model import FlightEncoder, PointerDecoder
from loader import build_airport_map, bases_to_ids, load_flights_rolling
from constraints import (
    get_delta_constraints, get_alaska_constraints, get_jetblue_constraints,
    get_turkish_constraints, FILM_CONSTRAINT_KEYS,
)
from utils import flights_to_tensors, constraint_to_tensor
import config

sys.path.insert(0, "experiments")
import train as train_mod
from train import run_episode

AIRLINES = {
    "delta":   get_delta_constraints,
    "alaska":  get_alaska_constraints,
    "jetblue": get_jetblue_constraints,
    "turkish": get_turkish_constraints,  # 값만 참고 — turkish flight/embedding은 안 씀 (실험 2)
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-rollouts", type=int, default=10)
    parser.add_argument("--greedy", action="store_true", help="stochastic 대신 greedy(결정론) rollout — 노이즈 제거용")
    args = parser.parse_args()

    device = torch.device(args.device)
    train_mod.DEVICE = device
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    n_airports = ckpt.get("n_airports", ckpt["encoder"]["airport_emb.weight"].shape[0])
    if n_airports <= 145:
        print(f"경고: n_airports={n_airports} — multi-airline 통합 임베딩(168)이 아닌 "
              f"단일 항공사 체크포인트로 보임. alaska/jetblue constraint 결과는 무의미할 수 있음.")

    # multi-airline 학습 시 쓴 것과 동일한 통합 airport map 재구성 (delta+alaska+jetblue)
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

    # flight 데이터는 delta 고정 — constraint만 항공사별로 바꿔가며 비교하는 게 핵심
    flights = load_flights_rolling(
        config.AIRLINE_DATA["delta"], window_days=5, offset_days=0, airport_map=airport_map,
        base_airport=base, n_max=config.EPISODE_MAX_FLIGHTS,
    )
    origins, dests, dep_times, arr_times, fly_times = flights_to_tensors(flights, 5 * 24.0, device=device)

    print(f"checkpoint: {args.checkpoint}  (n_airports={n_airports})")
    print(f"flight 데이터: delta 고정 {len(flights)}편 — constraint만 항공사별로 교체")
    print(f"{'항공사':<10}{'pairings':>10}{'deadheads':>11}{'coverage':>10}{'avg_overnight':>15}{'avg_legs':>10}")
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
    print("constraint 값 참고:")
    for airline, get_fn in AIRLINES.items():
        c = get_fn(base)
        print(f"  {airline}: " + ", ".join(f"{k}={c[k]}" for k in FILM_CONSTRAINT_KEYS))


if __name__ == "__main__":
    main()
