"""
eval_cross_objective.py — Tahir 논문(Eq.1-4)의 실제 목적함수(pairing_cost)로 우리
solution을 재채점해서, "서로 다른 objective로 서로를 채점"하던 문제를 해결한다.

배경(log/0709/tahir_모델_구조_차이_및_공정비교_계획.md):
  지금까지 dead time/FTC로 Tahir와 비교해왔는데, Tahir 논문을 읽어보니 Tahir는 dead
  time을 직접 최적화하지 않는다 — duty당 4시간 최소보장 pay(T_p, Eq.2) + 짧은
  연결/휴식에만 붙는 패널티(Eq.4)가 진짜 목적함수다. 우리가 dead time으로 채점한 건
  "우리 기준으로 Tahir를 채점"한 셈이라 불공정했다.

이 스크립트가 하는 일:
  1. eval_same_subset.py와 동일하게 Tahir I2CG를 실행해 실제 커버한 flight 부분집합을
     구하고, 우리 RL+IP도 그 부분집합으로 실행(코드 재사용, 무수정).
  2. Tahir의 pairing_cost()(Tahir/solver/constraints.py, Eq.1-4 그대로 구현된 기존
     함수, 무수정)를 우리 solution의 pairing들에 적용해 Tahir 기준 비용으로 재채점.
  3. Tahir 자신의 mip_obj(자기 objective 기준 자기 비용)와 직접 비교.
  4. 우리 solution엔 "같은 flight가 2개 pairing에 배정"(우리식 deadhead)이 있는데,
     Tahir의 pairing_cost()는 pairing 하나 단위로만 채점하므로 이 중복은 자동으로
     반영 안 됨 — Tahir의 실제 deadhead 패널티 공식(GAMMA_DH + LAMBDA_DH*duration)을
     그대로 가져와 중복 배정 건수에 별도로 곱해 "만약 이걸 진짜 deadhead로 신고했다면"
     추정치를 추가로 보여준다(과소평가 방지용 참고 수치).

Usage:
    cd /home/hyrn/ASCP-2026
    source ascp/bin/activate
    python -u eval_tahir_cross_objective.py --checkpoint checkpoints/z2db089m/model_latest.pt
"""

import sys
import os
import argparse

_THIS_DIR   = os.path.dirname(os.path.realpath(__file__))
_REPO_ROOT  = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))
_RL_DIR     = os.path.join(_REPO_ROOT, "RL")
_TAHIR_DIR  = os.path.join(_REPO_ROOT, "Tahir")
for p in (_REPO_ROOT, _RL_DIR, _TAHIR_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from eval_same_subset import (
    parse_raw_rows, run_tahir_and_get_covered_keys, run_ours_on_subset,
    DEFAULT_CSV, DEFAULT_TAHIR_CSV, DEFAULT_DATE_START, DEFAULT_DATE_END,
)
from solver.constraints import pairing_cost, GAMMA_DH, LAMBDA_DH


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/z2db089m/model_latest.pt")
    parser.add_argument("--csv",        default=DEFAULT_CSV)
    parser.add_argument("--tahir-csv",  default=DEFAULT_TAHIR_CSV)
    parser.add_argument("--date-start", default=DEFAULT_DATE_START)
    parser.add_argument("--date-end",   default=DEFAULT_DATE_END)
    parser.add_argument("--max-legs",   type=int, default=4000)
    parser.add_argument("--lambda-dh",  type=float, default=10.0)
    parser.add_argument("--ip-time-limit", type=int, default=300)
    parser.add_argument("--use-utc",    action="store_true", default=True)
    args = parser.parse_args()

    ckpt_path = args.checkpoint if os.path.isabs(args.checkpoint) else os.path.join(_REPO_ROOT, args.checkpoint)

    print("=" * 70)
    print("1단계: 원본 CSV 파싱 + Tahir I2CG 실행 → 부분집합 확보")
    print("=" * 70)
    rows = parse_raw_rows(args.csv, use_utc=args.use_utc)
    tahir_result, covered_keys, inst = run_tahir_and_get_covered_keys(
        args.tahir_csv, args.date_start, args.date_end, args.max_legs,
    )

    key_to_row = {r["tahir_key"]: r["row_id"] for r in rows}
    covered_row_ids = {key_to_row[k] for k in covered_keys if k in key_to_row}

    from evaluation import evaluate_ip
    airport_map_check = evaluate_ip.build_airport_map(evaluate_ip.config.AIRLINE_DATA["delta"])
    restricted = [
        {"id": r["row_id"], "origin": airport_map_check[r["origin_str"]],
         "dest": airport_map_check[r["dest_str"]], "dep_time": r["dep_time"], "arr_time": r["arr_time"]}
        for r in rows
        if r["row_id"] in covered_row_ids
        and r["origin_str"] in airport_map_check and r["dest_str"] in airport_map_check
    ]
    print(f"  → 부분집합 크기: {len(restricted)}편")

    print()
    print("=" * 70)
    print("2단계: 같은 부분집합으로 우리 RL+IP 실행")
    print("=" * 70)
    our_result = run_ours_on_subset(restricted, ckpt_path, args.lambda_dh, args.ip_time_limit)

    print()
    print("=" * 70)
    print("3단계: Tahir 목적함수(pairing_cost, Eq.1-4)로 우리 solution 재채점")
    print("=" * 70)

    # row_id -> Tahir leg dict(dep_abs/arr_abs/duration/flight_id)
    leg_key_by_fid = {leg["flight_id"]: (leg["origin"], leg["dest"], leg["dep_abs"]) for leg in inst["legs"]}
    key_to_tahir_leg = {leg_key_by_fid[leg["flight_id"]]: leg for leg in inst["legs"]}
    row_to_tahir_leg = {r["row_id"]: key_to_tahir_leg[r["tahir_key"]]
                         for r in rows if r["tahir_key"] in key_to_tahir_leg}

    our_pairings = our_result["selected"]
    our_cost_tahir_basis = 0.0
    n_conversion_fail = 0
    flight_use_count = {}   # row_id -> 이번 solution에서 몇 개 pairing에 쓰였는지(우리식 deadhead 탐지용)

    for p in our_pairings:
        try:
            tahir_legs = sorted(
                (row_to_tahir_leg[rid] for rid in p["legs"]),
                key=lambda l: l["dep_abs"],
            )
        except KeyError:
            n_conversion_fail += 1
            continue
        our_cost_tahir_basis += pairing_cost(tahir_legs)
        for rid in p["legs"]:
            flight_use_count[rid] = flight_use_count.get(rid, 0) + 1

    dup_flights = [rid for rid, c in flight_use_count.items() if c > 1]
    # 중복 배정 1건당 "진짜 deadhead였다면" Tahir 공식으로 추정 패널티(참고용, 과소평가 방지)
    dup_penalty_estimate = 0.0
    for rid in dup_flights:
        leg = row_to_tahir_leg.get(rid)
        if leg is not None:
            extra = flight_use_count[rid] - 1
            dup_penalty_estimate += extra * (GAMMA_DH + LAMBDA_DH * leg["duration"])

    print(f"  변환 실패(스킵): {n_conversion_fail}개 pairing")
    print(f"  중복 배정된 flight 수(우리식 deadhead): {len(dup_flights)}개")

    print()
    print("=" * 70)
    print("최종 — 동일 부분집합, Tahir 자신의 목적함수(mip_obj/pairing_cost) 기준")
    print("=" * 70)
    print(f"  Tahir 자기 solution 비용(mip_obj):                 {tahir_result['mip_obj']:>14.2f}")
    print(f"  우리 solution 비용(Tahir 공식으로 재채점, 순수):    {our_cost_tahir_basis:>14.2f}")
    print(f"  + 우리식 중복배정 {len(dup_flights)}건을 Tahir deadhead로 환산 시 추가분: "
          f"{dup_penalty_estimate:>10.2f}")
    print(f"  = 우리 solution 비용(중복배정 페널티 포함 추정):    "
          f"{our_cost_tahir_basis + dup_penalty_estimate:>14.2f}")
    print()
    ratio = (our_cost_tahir_basis + dup_penalty_estimate) / tahir_result["mip_obj"] * 100 - 100
    print(f"  → Tahir 자신의 objective로 채점해도, 우리가 {ratio:+.1f}% "
          f"{'나쁨' if ratio > 0 else '좋음'}")
    print("  (참고: 이 비교도 완벽하지 않음 — Tahir의 crew availability 제약(Eq.7)은")
    print("   우리 solution에 전혀 적용 안 됐고, dh_set을 pairing 내부에서 판정 안 하고")
    print("   pairing 간 중복만 별도 추정치로 더한 것이라 Tahir가 직접 낸 값과 계산 경로가 다름)")


if __name__ == "__main__":
    main()
