"""
diagnose_A_table3_wrongconstraint.py — diagnose_film_crossairline_ip.py(Table 3,
C 체크포인트 전용)를 단일 항공사 체크포인트(A, 예: z2db089m)에도 쓸 수 있게 고친 버전.

기존 스크립트는 airport_map을 항상 delta+alaska+jetblue 통합 맵으로 만드는데
(멀티에어라인 체크포인트 n_airports=168 전제), 단일 항공사 체크포인트(n_airports=145,
Delta 공항만 학습)에 그 통합 맵을 쓰면 공항 인덱스가 학습 시점과 어긋나서 조용히
잘못된 임베딩을 참조하게 된다(evaluation/evaluate_ip.py의 모델 로드 경로가 이미 이 문제를 n_airports
분기로 처리하고 있음 — 이 스크립트는 그 분기 로직만 가져와 적용).

flight window는 항상 delta 고정(Table 3 설계 자체가 그렇다)이므로, A가 학습한
Delta 공항 범위를 벗어나는 OOV 공항 문제는 발생하지 않는다 — airport_map만
체크포인트에 맞게 델타 단독으로 만들면 된다.

기존 diagnose_film_crossairline_ip.py는 무수정, 이 스크립트는 그 로직을 그대로
복사해 airport_map 생성 부분만 고쳤다.
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
    args = parser.parse_args()

    device = torch.device(args.device)
    evaluate_ip.DEVICE = device
    set_environment("delta")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    n_airports = ckpt.get("n_airports", ckpt["encoder"]["airport_emb.weight"].shape[0])

    # ── 핵심 수정: 체크포인트가 실제로 학습한 공항 맵과 일치시킨다 ──────────
    # multi-airline(n_airports>145)이면 delta+alaska+jetblue 통합 맵(원본과 동일),
    # 단일 항공사(n_airports<=145, 이 프로젝트에서는 사실상 delta 단독)면 delta
    # CSV 하나로만 맵을 만든다 — evaluation/evaluate_ip.py의 airport-map 분기와 동일한 처리.
    if n_airports > 145:
        print(f"n_airports={n_airports} → multi-airline 통합 임베딩으로 판단, 통합 공항맵 사용")
        map_paths = [v for k, v in config.AIRLINE_DATA.items() if k != "turkish"]
    else:
        print(f"n_airports={n_airports} → 단일 항공사(delta) 임베딩으로 판단, delta 단독 공항맵 사용")
        map_paths = config.AIRLINE_DATA["delta"]
    airport_map = build_airport_map(map_paths)
    if len(airport_map) != n_airports:
        print(f"경고: 재구성한 airport_map 크기({len(airport_map)})가 체크포인트 "
              f"n_airports({n_airports})와 다름 — 학습 당시와 다른 데이터로 맵을 만들었을 "
              f"가능성이 있어 임베딩이 여전히 어긋날 수 있음. 결과 해석 시 주의.")
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

    # flight 윈도우는 delta 고정 — constraint만 항공사별로 교체 (A는 학습 때 본
    # 공항 범위를 벗어나지 않으므로 OOV 문제 없음)
    print(f"delta flight 윈도우 로드 중 (offset_days={args.offset_days}, window_days={args.window_days})...", flush=True)
    window_flights = load_flights_rolling(
        config.AIRLINE_DATA["delta"], window_days=args.window_days, offset_days=args.offset_days,
        airport_map=airport_map, base_airport=base, n_max=None,
    )
    for f in window_flights:
        f["global_id"] = f["id"]
    n_total = len(window_flights)
    print(f"고정 flight 수: {n_total}편 (base={base})", flush=True)

    results = {}
    for airline, get_fn in AIRLINES.items():
        constraint = get_fn(base)
        print(f"\n{'='*60}\nconstraint={airline} (flight 데이터는 delta 고정)\n{'='*60}", flush=True)

        windows = [[dict(f) for f in window_flights]]

        with torch.no_grad():
            pool, covered = collect_pool_full(
                windows, base_ids, constraint, encoder, decoder,
                n_rollouts_per_chunk=args.n_rollouts_per_chunk,
                subset_size=args.subset_size,
                connected_sampler=sample_connected_subnet_std,
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
              f"deadhead: {result['deadhead_count']}  FTC: {ftc:.2f}%  "
              f"avg_legs: {results[airline]['avg_legs']:.2f}  status: {result['status']}", flush=True)

    print(f"\n\n{'='*60}\n요약 (delta {n_total}편 고정, constraint만 교체)\n{'='*60}")
    header = f"{'항공사':<10}{'pairings':>10}{'ManDays':>10}{'deadhead':>10}{'FTC':>9}{'avg_legs':>10}{'avg_duties':>11}"
    print(header)
    for airline, r in results.items():
        print(f"{airline:<10}{r['n_pairings']:>10}{r['man_days']:>10}{r['deadhead']:>10}"
              f"{r['ftc']:>8.2f}%{r['avg_legs']:>10.2f}{r['avg_duties']:>11.2f}")


if __name__ == "__main__":
    main()
