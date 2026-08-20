"""
diagnose_film_crossairline_ip.py — diagnose_film_crossairline.py(③ greedy rollout)를
IP 평가 레벨로 확장. 같은 delta flight 윈도우를 고정하고 constraint만 delta/alaska/jetblue로
교체해가며 evaluation/evaluate_ip.py와 동일한 pool 수집 + Set Covering IP를 돌려, legs/deadhead/ManDays
등에서 나오는 차이가 순수하게 FiLM의 constraint 조건화 때문인지 확인한다(항공사별 flight
network 차이라는 confound 제거).

evaluation/evaluate_ip.py의 collect_pool_full/solve_set_covering을 그대로 재사용하되, 월 전체가 아니라
윈도우 1개(offset_days=0, window_days=5)만 고정해서 세 항공사 constraint에 반복 적용한다.
"""
import os
import sys
import argparse

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "RL"))

from loader import load_flights_rolling, build_airport_map, bases_to_ids
from constraints import (
    get_delta_constraints, get_alaska_constraints, get_jetblue_constraints,
    FILM_CONSTRAINT_KEYS,
)
from model import FlightEncoder, PointerDecoder
from evaluation.set_partition import solve_set_covering
from rollout import set_environment
import config

from evaluation import evaluate_ip
from evaluation.evaluate_ip import collect_pool_full, sample_connected_subnet_std

AIRLINES = {
    "delta":   get_delta_constraints,
    "alaska":  get_alaska_constraints,
    "jetblue": get_jetblue_constraints,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--window-days", type=int, default=5)
    parser.add_argument("--offset-days", type=int, default=0)
    parser.add_argument("--subset-size", type=int, default=config.EPISODE_MAX_FLIGHTS)
    parser.add_argument("--n-rollouts-per-chunk", type=int, default=5)
    parser.add_argument("--ip-time-limit", type=int, default=1800)
    parser.add_argument("--lambda-dh", type=float, default=1.0)
    parser.add_argument("--use-utc", action="store_true")
    parser.add_argument("--require-base-return", action="store_true",
                        help="decode-time hard mask 활성화 — rollout 중 base 복귀가 불가능해지는 "
                             "leg를 마스킹하고, base 아닌 곳에서 END_PAIRING을 금지한다.")
    args = parser.parse_args()

    device = torch.device(args.device)
    evaluate_ip.DEVICE = device
    set_environment("delta")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    n_airports = ckpt.get("n_airports", ckpt["encoder"]["airport_emb.weight"].shape[0])
    if n_airports <= 145:
        print(f"경고: n_airports={n_airports} — multi-airline 통합 임베딩(168)이 아닐 수 있음.")

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

    # flight 윈도우는 delta 고정 — constraint만 항공사별로 교체
    print(f"delta flight 윈도우 로드 중 (offset_days={args.offset_days}, window_days={args.window_days})...", flush=True)
    window_flights = load_flights_rolling(
        config.AIRLINE_DATA["delta"], window_days=args.window_days, offset_days=args.offset_days,
        airport_map=airport_map, base_airport=base, n_max=None, use_utc=args.use_utc,
    )
    for f in window_flights:
        f["global_id"] = f["id"]
    n_total = len(window_flights)
    print(f"고정 flight 수: {n_total}편 (base={base})", flush=True)

    results = {}
    for airline, get_fn in AIRLINES.items():
        constraint = get_fn(base)
        print(f"\n{'='*60}\nconstraint={airline} (flight 데이터는 delta 고정)\n{'='*60}", flush=True)

        # collect_pool_full은 window_flights를 in-place로 건드리므로(f['id'] 재부여) 매 airline마다 복사
        windows = [[dict(f) for f in window_flights]]

        with torch.no_grad():
            pool, covered = collect_pool_full(
                windows, base_ids, constraint, encoder, decoder,
                n_rollouts_per_chunk=args.n_rollouts_per_chunk,
                subset_size=args.subset_size,
                connected_sampler=sample_connected_subnet_std,
                airline=airline,
                require_base_return=args.require_base_return,
            )

        result = solve_set_covering(pool, n_flights=n_total, time_limit=args.ip_time_limit, lambda_dh=args.lambda_dh)
        sel = result["selected"]
        import math
        fly_total    = sum(p["fly"] for p in sel) if sel else 0.0
        dead_total   = sum(p.get("dead_time", p["cost"]) for p in sel) if sel else 0.0
        legs_total   = sum(p.get("n_legs", len(p["legs"])) for p in sel) if sel else 0
        duties_total = sum(p.get("n_duties", 1) for p in sel) if sel else 0
        man_days     = sum(math.ceil(p["elapsed"] / 24.0) for p in sel) if sel else 0
        intra_gap_total    = sum(p.get("intra_duty_gap", 0.0) for p in sel) if sel else 0.0
        inter_excess_total = sum(p.get("inter_duty_excess", 0.0) for p in sel) if sel else 0.0
        ftc = intra_gap_total / fly_total * 100 if fly_total > 0 else 0.0

        results[airline] = dict(
            n_pairings=result["n_pairings"], man_days=man_days,
            coverage=result["coverage"] * 100, deadhead=result["deadhead_count"],
            fly=fly_total, dead=dead_total, intra_gap=intra_gap_total,
            inter_excess=inter_excess_total, ftc=ftc,
            avg_legs=legs_total / len(sel) if sel else 0.0,
            avg_duties=duties_total / len(sel) if sel else 0.0,
            status=result["status"],
        )
        print(f"  pairing 수: {result['n_pairings']}  ManDays: {man_days}  "
              f"deadhead: {result['deadhead_count']}  dead_time: {dead_total:.2f}h  FTC: {ftc:.2f}%  "
              f"avg_legs: {results[airline]['avg_legs']:.2f}  status: {result['status']}", flush=True)

    print(f"\n\n{'='*60}\n요약 (delta {n_total}편 고정, constraint만 교체)\n{'='*60}")
    header = f"{'항공사':<10}{'pairings':>10}{'ManDays':>10}{'deadhead':>10}{'dead_time':>12}{'FTC':>9}{'avg_legs':>10}{'avg_duties':>11}"
    print(header)
    for airline, r in results.items():
        print(f"{airline:<10}{r['n_pairings']:>10}{r['man_days']:>10}{r['deadhead']:>10}"
              f"{r['intra_gap']:>11.2f}h{r['ftc']:>8.2f}%{r['avg_legs']:>10.2f}{r['avg_duties']:>11.2f}")


if __name__ == "__main__":
    main()
