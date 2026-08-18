"""
eval_reselect_variants.py — RL 모델(rollout)은 전혀 건드리지 않고, IP 선택 단계의
cost 함수/λ_dh만 바꿔서 pairing 압축력(legs/pairing, deadhead, Tahir 기준 비용)이
개선되는지 실험한다.

배경(log/0709/tahir_비교_계획.md §7-1):
  Tahir 자신의 pairing_cost()로 우리 solution을 재채점하면 +171% 나쁘다(§8). 원인
  가설은 Tahir가 duty당 4시간 최소보장 pay(T_p) 때문에 자연히 pairing을 길게 묶는
  유인이 있는데 우리 IP 목적함수(dead_time 기반)엔 그런 유인이 없다는 것. 이 스크립트는
  같은 rollout pool(1번만 수집, RL 모델 무수정)을 여러 cost/λ_dh 조합으로 다시 선택해서
  어떤 조합이 Tahir 기준 비용을 가장 줄이는지 비교한다.

변형(variant) 5개:
  A. baseline(원래 dead_time 기반 cost, λ_dh=10) — 지금까지 쓰던 것
  B. λ_dh=50(원래 cost)
  C. λ_dh=100(원래 cost)
  D. Tahir T_p 근사 cost(λ_dh=10) — max(elapsed/4, n_duties*4h, fly)로 cost 대체
  E. Tahir T_p 근사 cost(λ_dh=50)

Usage:
    cd /home/hyrn/ASCP-2026
    source ascp/bin/activate
    python -u eval_tahir_reselect_variants.py --checkpoint checkpoints/z2db089m/model_latest.pt
"""

import sys
import os
import argparse
import copy

_THIS_DIR   = os.path.dirname(os.path.realpath(__file__))
_REPO_ROOT  = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))
_RL_DIR     = os.path.join(_REPO_ROOT, "RL")
_TAHIR_DIR  = os.path.join(_REPO_ROOT, "Tahir")
for p in (_REPO_ROOT, _RL_DIR, _TAHIR_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch

from eval_same_subset import (
    parse_raw_rows, run_tahir_and_get_covered_keys,
    DEFAULT_CSV, DEFAULT_TAHIR_CSV, DEFAULT_DATE_START, DEFAULT_DATE_END,
)
from solver.constraints import pairing_cost


def tp_approx_cost(p):
    """Tahir Eq.2(T_p)의 근사치 — per-duty fly time 분해 정보가 없어 균등분배로 근사.
    max(pairing span/4, duty 수*4h(최소보장), 총 비행시간) — duty를 잘게 쪼갤수록
    n_duties*4h 항이 커져서 자연히 penalize된다."""
    elapsed = p.get("elapsed", 0.0)
    n_duties = p.get("n_duties", 1)
    fly = p.get("fly", 0.0)
    return max(elapsed / 4.0, n_duties * 4.0, fly)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/z2db089m/model_latest.pt")
    parser.add_argument("--csv",        default=DEFAULT_CSV)
    parser.add_argument("--tahir-csv",  default=DEFAULT_TAHIR_CSV)
    parser.add_argument("--date-start", default=DEFAULT_DATE_START)
    parser.add_argument("--date-end",   default=DEFAULT_DATE_END)
    parser.add_argument("--max-legs",   type=int, default=4000)
    parser.add_argument("--ip-time-limit", type=int, default=300)
    parser.add_argument("--use-utc",    action="store_true", default=True)
    args = parser.parse_args()

    ckpt_path = args.checkpoint if os.path.isabs(args.checkpoint) else os.path.join(_REPO_ROOT, args.checkpoint)

    print("=" * 70)
    print("1단계: 부분집합 확보(Tahir 실제 커버 flight, Delta 정합 제약)")
    print("=" * 70)
    rows = parse_raw_rows(args.csv, use_utc=args.use_utc)
    tahir_result, covered_keys, inst = run_tahir_and_get_covered_keys(
        args.tahir_csv, args.date_start, args.date_end, args.max_legs,
    )
    key_to_row = {r["tahir_key"]: r["row_id"] for r in rows}
    covered_row_ids = {key_to_row[k] for k in covered_keys if k in key_to_row}

    from evaluation import evaluate_ip
    airport_map_check = evaluate_ip.build_airport_map(evaluate_ip.config.AIRLINE_DATA["delta"])
    base_ids = evaluate_ip.bases_to_ids(evaluate_ip.config.AIRLINE_BASES["delta"], airport_map_check)
    constraint = evaluate_ip._GET_CONSTRAINT["delta"](base_ids[0])
    restricted = [
        {"id": r["row_id"], "origin": airport_map_check[r["origin_str"]],
         "dest": airport_map_check[r["dest_str"]], "dep_time": r["dep_time"], "arr_time": r["arr_time"]}
        for r in rows
        if r["row_id"] in covered_row_ids
        and r["origin_str"] in airport_map_check and r["dest_str"] in airport_map_check
    ]
    for new_id, f in enumerate(restricted):
        f["global_id"] = new_id
    print(f"  → 부분집합 크기: {len(restricted)}편")

    print()
    print("=" * 70)
    print("2단계: RL rollout pool 수집(딱 1번, 이후 변형(variant)들이 전부 재사용)")
    print("=" * 70)
    ckpt = torch.load(ckpt_path, map_location=evaluate_ip.DEVICE, weights_only=True)
    n_airports = ckpt.get("n_airports", ckpt["encoder"]["airport_emb.weight"].shape[0])
    encoder = evaluate_ip.FlightEncoder(n_airports=n_airports, constraint_dim=len(evaluate_ip.FILM_CONSTRAINT_KEYS)).to(evaluate_ip.DEVICE)
    airport_emb_dim = encoder.airport_emb.embedding_dim
    ckpt_state_dim = ckpt["decoder"]["state_mlp.0.weight"].shape[1]
    n_scalars = ckpt_state_dim - airport_emb_dim * 2 - len(evaluate_ip.FILM_CONSTRAINT_KEYS)
    decoder = evaluate_ip.PointerDecoder(constraint_dim=len(evaluate_ip.FILM_CONSTRAINT_KEYS),
                                          airport_emb_dim=airport_emb_dim, n_scalars=n_scalars).to(evaluate_ip.DEVICE)
    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        pool, _covered = evaluate_ip.collect_pool_full(
            [restricted], base_ids, constraint, encoder, decoder,
            n_rollouts_per_chunk=5, subset_size=evaluate_ip.config.EPISODE_MAX_FLIGHTS,
            connected_sampler=evaluate_ip.sample_connected_subnet_std,
        )
    print(f"  pool 수집 완료: {len(pool)}개 pairing (이후 전부 이 pool 재사용, rollout 재실행 없음)")

    # row_id -> Tahir leg (cross-objective 재채점용)
    leg_key_by_fid = {leg["flight_id"]: (leg["origin"], leg["dest"], leg["dep_abs"]) for leg in inst["legs"]}
    key_to_tahir_leg = {leg_key_by_fid[leg["flight_id"]]: leg for leg in inst["legs"]}
    row_to_tahir_leg = {r["row_id"]: key_to_tahir_leg[r["tahir_key"]] for r in rows if r["tahir_key"] in key_to_tahir_leg}

    def tahir_score(sel):
        total = 0.0
        for p in sel:
            try:
                legs = sorted((row_to_tahir_leg[rid] for rid in p["legs"]), key=lambda l: l["dep_abs"])
            except KeyError:
                continue
            total += pairing_cost(legs)
        return total

    n_total = len(restricted)
    variants = [
        ("A: baseline(원래 cost, λ_dh=10)",      None,          10.0),
        ("B: 원래 cost, λ_dh=50",                None,          50.0),
        ("C: 원래 cost, λ_dh=100",               None,         100.0),
        ("D: Tahir T_p 근사 cost, λ_dh=10",      tp_approx_cost, 10.0),
        ("E: Tahir T_p 근사 cost, λ_dh=50",      tp_approx_cost, 50.0),
    ]

    print()
    print("=" * 70)
    print("3단계: 변형별 IP 재선택 (같은 pool, cost/λ_dh만 다르게)")
    print("=" * 70)

    rows_out = []
    for name, cost_fn, lam in variants:
        if cost_fn is None:
            variant_pool = pool
        else:
            variant_pool = []
            for p in pool:
                p2 = dict(p)
                p2["cost"] = cost_fn(p)
                variant_pool.append(p2)

        result = evaluate_ip.solve_set_covering(
            variant_pool, n_flights=n_total, time_limit=args.ip_time_limit, lambda_dh=lam,
        )
        sel = result["selected"]
        fly_total = sum(p["fly"] for p in sel) if sel else 0.0
        intra_gap = sum(p.get("intra_duty_gap", 0.0) for p in sel) if sel else 0.0
        legs_total = sum(p.get("n_legs", len(p["legs"])) for p in sel) if sel else 0
        avg_legs = legs_total / len(sel) if sel else 0.0
        tscore = tahir_score(sel)
        gap_vs_tahir = (tscore / tahir_result["mip_obj"] - 1) * 100

        print(f"  [{name}] status={result['status']} n_pairings={result['n_pairings']} "
              f"coverage={result['coverage']*100:.1f}% deadhead={result['deadhead_count']} "
              f"avg_legs={avg_legs:.2f} dead_time={intra_gap:.1f}h "
              f"Tahir기준비용={tscore:.1f}(Tahir대비 {gap_vs_tahir:+.1f}%)", flush=True)
        rows_out.append((name, result['n_pairings'], result['coverage']*100, result['deadhead_count'],
                          avg_legs, intra_gap, tscore, gap_vs_tahir))

    print()
    print("=" * 70)
    print(f"요약 — Tahir 자기 solution 비용(mip_obj) = {tahir_result['mip_obj']:.1f} (기준선)")
    print("=" * 70)
    header = f"{'variant':<38} {'pairings':>8} {'cov%':>6} {'DH':>5} {'avg_legs':>8} {'dead_h':>8} {'Tahir비용':>10} {'격차%':>8}"
    print(header)
    for name, npair, cov, dh, al, dt, ts, gap in rows_out:
        print(f"{name:<38} {npair:>8d} {cov:>6.1f} {dh:>5d} {al:>8.2f} {dt:>8.1f} {ts:>10.1f} {gap:>+8.1f}")


if __name__ == "__main__":
    main()
