"""
diagnose_film_and_decoder_ablation.py — C(pws5cjlz)의 두 constraint 인식 경로
(①FiLM: 인코더, ②디코더 직접 concat)를 재학습 없이 추론 시점에 독립적으로
켜고 끄면서 4가지 조합(정상 / FiLM만 무력화 / 디코더만 무력화 / 둘 다 무력화)을
같은 실행 안에서 전부 비교한다.

배경: `diagnose_film_inference_ablation.py`(FiLM만 무력화, log/0712 실행 →
예측-일치 8/9→7/9)와 `diagnose_decoder_constraint_ablation.py`(디코더 경로만
무력화, log/0713 실행 → 9/9→9/9)를 **서로 다른 실행**에서 따로 돌렸더니, 각
실행의 "정상 C" 기준선 자체가 rollout stochasticity 때문에 달랐다(8/9 vs 9/9).
그래서 "경로 하나만 꺼도 거의 안 줄어든다"는 관찰은 맞지만, 정말 두 경로를
**동시에** 꺼야 크게 떨어지는지는 같은 실행 안에서 4개 조건을 나란히 비교해야
깨끗하게 알 수 있다 — 이 스크립트가 그 비교를 한 번에 한다.

무력화 방법(둘 다 기존 스크립트와 동일, 무수정 재사용):
  - FiLM 무력화: encoder.film_before/film_after.forward를 identity로 교체.
  - 디코더 경로 무력화: train.py/rollout.py 모듈 전역의 state_to_vec을
    "constraint 인자 대신 항상 delta 고정값을 쓰는" 래퍼로 교체.
두 무력화는 서로 독립적인 지점(인코더 vs 디코더)이라 자유롭게 조합 가능하다.

Usage:
    cd /home/hyrn/ASCP-2026
    source ascp/bin/activate
    python -u diagnose/diagnose_film_and_decoder_ablation.py checkpoints/pws5cjlz/stage3_best.pt \
        --device cuda:0 --use-utc
"""
import os
import sys
import math
import argparse

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "RL"))
sys.path.insert(0, os.path.join(REPO_ROOT, "experiments"))

from model import FlightEncoder, PointerDecoder
from loader import build_airport_map, bases_to_ids, load_flights_rolling
from constraints import (
    get_delta_constraints, get_alaska_constraints, get_jetblue_constraints,
    get_turkish_constraints, FILM_CONSTRAINT_KEYS,
)
from utils import flights_to_tensors, constraint_to_tensor, state_to_vec as _state_to_vec_orig
from evaluation.set_partition import solve_set_covering
from rollout import set_environment

import config

import train as train_mod
import rollout as rollout_mod
from train import run_episode

from evaluation import evaluate_ip
from evaluation.evaluate_ip import collect_pool_full, sample_connected_subnet_std

GREEDY_AIRLINES = {
    "delta": get_delta_constraints, "alaska": get_alaska_constraints,
    "jetblue": get_jetblue_constraints, "turkish": get_turkish_constraints,
}
TABLE3_AIRLINES = {
    "delta": get_delta_constraints, "alaska": get_alaska_constraints,
    "jetblue": get_jetblue_constraints,
}


# ── 무력화 유틸(두 기존 diagnose 스크립트와 동일 로직, 무수정 재사용) ──────────

def bypass_film(encoder):
    identity = lambda flight_vecs, constraint: flight_vecs
    encoder.film_before.forward = identity
    encoder.film_after.forward = identity


def make_blind_state_to_vec(fixed_constraint):
    def blind(state, encoder, constraint, device=None, include_total_legs=True):
        return _state_to_vec_orig(state, encoder, fixed_constraint,
                                   device=device, include_total_legs=include_total_legs)
    return blind


def bypass_decoder_constraint(fixed_constraint):
    blind = make_blind_state_to_vec(fixed_constraint)
    train_mod.state_to_vec = blind
    rollout_mod.state_to_vec = blind


def restore_state_to_vec():
    train_mod.state_to_vec = _state_to_vec_orig
    rollout_mod.state_to_vec = _state_to_vec_orig


def run_greedy_stage(encoder, decoder, airport_map, base, device, label):
    print(f"\n{'='*70}\n③ greedy 단계 ({label}) — delta 600편 고정, constraint만 교체\n{'='*70}")
    flights = load_flights_rolling(
        config.AIRLINE_DATA["delta"], window_days=5, offset_days=0, airport_map=airport_map,
        base_airport=base, n_max=config.EPISODE_MAX_FLIGHTS,
    )
    origins, dests, dep_times, arr_times, fly_times = flights_to_tensors(flights, 5 * 24.0, device=device)
    train_mod.DEVICE = device
    results = {}
    print(f"{'항공사':<10}{'pairings':>10}{'deadheads':>11}{'avg_overnight':>15}{'avg_legs':>10}")
    with torch.no_grad():
        for airline, get_fn in GREEDY_AIRLINES.items():
            val_c = get_fn(base)
            val_enc = encoder(origins, dests, dep_times, arr_times, fly_times,
                               constraint_to_tensor(val_c, device=device))
            _, _, _, m = run_episode(flights, val_c, encoder, decoder, val_enc, greedy=True)
            results[airline] = dict(pairings=m["n_pairings"], deadheads=m["n_deadheads"],
                                     avg_overnight=m.get("avg_overnight", 0), avg_legs=m.get("avg_legs", 0))
            print(f"{airline:<10}{m['n_pairings']:>10}{m['n_deadheads']:>11}"
                  f"{m.get('avg_overnight', 0):>15.3f}{m.get('avg_legs', 0):>10.3f}")
    return results


def run_table3_stage(encoder, decoder, airport_map, base_ids, base, device, args, label):
    print(f"\n{'='*70}\n④' Table3 단계 ({label}) — delta 편 고정, constraint만 교체\n{'='*70}")
    window_flights = load_flights_rolling(
        config.AIRLINE_DATA["delta"], window_days=args.window_days, offset_days=0,
        airport_map=airport_map, base_airport=base, n_max=None, use_utc=args.use_utc,
    )
    for f in window_flights:
        f["global_id"] = f["id"]
    n_total = len(window_flights)
    print(f"고정 flight 수: {n_total}편 (base={base})")

    results = {}
    for airline, get_fn in TABLE3_AIRLINES.items():
        constraint = get_fn(base)
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
        fly_total  = sum(p["fly"] for p in sel) if sel else 0.0
        legs_total = sum(p.get("n_legs", len(p["legs"])) for p in sel) if sel else 0
        duties_total = sum(p.get("n_duties", 1) for p in sel) if sel else 0
        man_days   = sum(math.ceil(p["elapsed"] / 24.0) for p in sel) if sel else 0
        intra_gap_total = sum(p.get("intra_duty_gap", 0.0) for p in sel) if sel else 0.0
        ftc = intra_gap_total / fly_total * 100 if fly_total > 0 else 0.0
        results[airline] = dict(
            n_pairings=result["n_pairings"], man_days=man_days, deadhead=result["deadhead_count"],
            ftc=ftc, avg_legs=legs_total / len(sel) if sel else 0.0,
            avg_duties=duties_total / len(sel) if sel else 0.0, status=result["status"],
        )
        print(f"  constraint={airline:<8} pairing={result['n_pairings']:>6} ManDays={man_days:>6} "
              f"deadhead={result['deadhead_count']:>6} FTC={ftc:>6.2f}% "
              f"avg_legs={results[airline]['avg_legs']:.2f} avg_duties={results[airline]['avg_duties']:.2f} "
              f"status={result['status']}")
    return results


def check_predictions(greedy_res, table3_res, label):
    print(f"\n{'='*70}\n예측-일치 체크 ({label})\n{'='*70}")
    checks = []
    g = greedy_res
    checks.append(("greedy jetblue pairings 최저", g["jetblue"]["pairings"] == min(v["pairings"] for v in g.values())))
    checks.append(("greedy jetblue avg_legs 최고", g["jetblue"]["avg_legs"] == max(v["avg_legs"] for v in g.values())))
    checks.append(("greedy jetblue avg_overnight 최고", g["jetblue"]["avg_overnight"] == max(v["avg_overnight"] for v in g.values())))
    checks.append(("greedy turkish pairings 최다", g["turkish"]["pairings"] == max(v["pairings"] for v in g.values())))
    checks.append(("greedy turkish avg_legs 최저", g["turkish"]["avg_legs"] == min(v["avg_legs"] for v in g.values())))
    t = table3_res
    checks.append(("table3 jetblue pairings 최저", t["jetblue"]["n_pairings"] == min(v["n_pairings"] for v in t.values())))
    checks.append(("table3 jetblue avg_legs 최고", t["jetblue"]["avg_legs"] == max(v["avg_legs"] for v in t.values())))
    checks.append(("table3 jetblue avg_duties 최고", t["jetblue"]["avg_duties"] == max(v["avg_duties"] for v in t.values())))
    checks.append(("table3 jetblue FTC 최저", t["jetblue"]["ftc"] == min(v["ftc"] for v in t.values())))
    n_pass = sum(1 for _, ok in checks if ok)
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"  합계: {n_pass}/{len(checks)} 통과")
    return n_pass, len(checks)


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
    encoder.eval(); decoder.eval()
    return encoder, decoder, n_airports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--window-days", type=int, default=5)
    parser.add_argument("--subset-size", type=int, default=config.EPISODE_MAX_FLIGHTS)
    parser.add_argument("--n-rollouts-per-chunk", type=int, default=5)
    parser.add_argument("--ip-time-limit", type=int, default=1800)
    parser.add_argument("--lambda-dh", type=float, default=1.0)
    parser.add_argument("--use-utc", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    evaluate_ip.DEVICE = device
    set_environment("delta")

    map_paths = [v for k, v in config.AIRLINE_DATA.items() if k != "turkish"]
    airport_map = build_airport_map(map_paths)
    base_ids = bases_to_ids(list(config.AIRLINE_BASES["delta"]), airport_map)
    base = base_ids[0]

    fixed_constraint = get_delta_constraints(base)

    modes = [
        ("normal",        "정상 C(FiLM+디코더 둘 다 켜짐)",        False, False),
        ("film_only",     "FiLM만 무력화(디코더 경로는 켜짐)",       True,  False),
        ("decoder_only",  "디코더 경로만 무력화(FiLM은 켜짐)",       False, True),
        ("both",          "FiLM+디코더 둘 다 무력화",               True,  True),
    ]

    summary = {}
    for mode, label, film_off, decoder_off in modes:
        encoder, decoder, n_airports = load_model(args.checkpoint, device)
        if n_airports <= 145:
            print(f"경고: n_airports={n_airports} — multi-airline 체크포인트가 아닐 수 있음.")

        restore_state_to_vec()
        if film_off:
            bypass_film(encoder)
        if decoder_off:
            bypass_decoder_constraint(fixed_constraint)

        greedy_res = run_greedy_stage(encoder, decoder, airport_map, base, device, label)
        table3_res = run_table3_stage(encoder, decoder, airport_map, base_ids, base, device, args, label)
        n_pass, n_total = check_predictions(greedy_res, table3_res, label)
        summary[mode] = (n_pass, n_total, greedy_res, table3_res)

    restore_state_to_vec()

    print(f"\n\n{'='*70}\n최종 비교 — 4개 조건 나란히\n{'='*70}")
    for mode, label, _, _ in modes:
        n_pass, n_total, _, _ = summary[mode]
        print(f"  {label:<32} 예측-일치 {n_pass}/{n_total}")


if __name__ == "__main__":
    main()
