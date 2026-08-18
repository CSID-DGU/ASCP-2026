"""
eval_cross_objective_full_coverage.py — eval_cross_objective.py의 "같은 부분집합만
비교"(1,810/3,500편, Tahir가 실제로 커버한 것만) 대신, **Tahir가 커버 못 한 나머지
편을 전부 1-leg pairing으로 강제 배정**해서 Tahir 쪽도 100% 커버로 맞춘 뒤, 우리
모델도 원래 하던 대로(부분집합 아닌) 전체 3,500편에 그대로 돌려서, 양쪽 다 진짜
100% 커버 상태에서 Tahir 자신의 목적함수(pairing_cost, Eq.1-4)로 비교한다.

배경(log/0712/tahir_baseline_비교방법_쉬운설명.md, log/0711/paper/07_Tahir_RL보상수정_실험.md):
  기존 cross-objective 비교(eval_cross_objective.py)는 "Tahir가 실제로 커버한
  1,810편만 우리도 그만큼만 풀게" 강제해서 coverage confound를 없앴다. 하지만 이
  방식은 Tahir가 애초에 못 푼(연결이 어려운) 1,690편을 비교에서 통째로 빼버린다는
  단점이 있다 — Tahir 입장에서는 "쉬운 문제만 낸" 것과 비슷한 효과.

  이 스크립트는 반대 방향으로 접근한다: Tahir가 커버 못 한 flight들을 그냥 안 낸
  걸로 치지 않고, **각각을 독립된 1-leg pairing으로 강제 배정**해서 Tahir도 3,500편
  전체를 100% 커버한 것으로 만든다. 1-leg pairing은 연결이 없으므로
  `pairing_cost()`의 phi^C(연결/휴식 패널티)는 0, dh_penalty도 0(deadhead 아님)이라,
  비용은 T_p = max(duration/4, max(240min, duration)) = max(240min, duration) —
  Tahir의 Eq.2(duty당 최소 4시간 보장 pay) 그대로 적용된 "이 flight 하나만 도는
  duty"의 비용이다. (참고: `is_feasible_pairing()`은 base 출발/도착을 요구하지만
  `pairing_cost()` 자체는 그 제약을 검사하지 않으므로 base가 아닌 flight도 그대로
  비용 계산 가능 — 이건 "진짜 실행 가능한 pairing"이 아니라 "이 flight을 어떻게든
  커버해야 한다면 최소 얼마가 드는가"를 보여주는 하한에 가까운 추정치임을 명시한다.)

  우리 쪽은 부분집합 제한 없이 원래 evaluation/evaluate_ip.py 파이프라인을 3,500편 전체에
  그대로 돌린다(원래도 100% 커버가 구조적으로 보장됨). 기존 코드
  (eval_same_subset.py, evaluation/evaluate_ip.py, Tahir/solver/*)는 전혀 수정하지 않고, 이
  스크립트가 그 위에 "패딩 비용 계산"과 "부분집합 필터를 안 거는 실행 경로"만
  추가한다.

Usage:
    cd /home/hyrn/ASCP-2026
    source ascp/bin/activate
    python -u eval_tahir_full_coverage.py --checkpoint checkpoints/z2db089m/model_latest.pt
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
    parser.add_argument("--compute-gap", action="store_true",
                         help="우리(RL+IP) 쪽에 LP relaxation을 추가로 풀어 "
                              "Gap%%=(MIP_obj-LP_obj)/LP_obj*100을 계산(evaluation/evaluate_ip.py와 동일 정의)")
    parser.add_argument("--tahir-method", default="i2cg", choices=["i2cg", "i2cgp", "both"],
                         help="Tahir 쪽 알고리즘(기본 i2cg). i2cgp/both는 같은 inst/ref로 "
                              "Tahir/experiments/delta_dnn 가중치를 써서 I2CGp도 같이 실행하고 "
                              "버전2/버전3 비교를 I2CGp에 대해서도 추가로 출력한다.")
    args = parser.parse_args()

    ckpt_path = args.checkpoint if os.path.isabs(args.checkpoint) else os.path.join(_REPO_ROOT, args.checkpoint)

    print("=" * 70)
    print("1단계: 원본 CSV 파싱 + Tahir I2CG 실행 → 커버 flight 확보")
    print("=" * 70)
    rows = parse_raw_rows(args.csv, use_utc=args.use_utc)
    tahir_result, covered_keys, inst = run_tahir_and_get_covered_keys(
        args.tahir_csv, args.date_start, args.date_end, args.max_legs,
        tahir_method=args.tahir_method,
    )

    key_to_row = {r["tahir_key"]: r["row_id"] for r in rows}

    from evaluation import evaluate_ip
    airport_map_check = evaluate_ip.build_airport_map(DEFAULT_CSV)
    # 부분집합 필터 없이 매칭 가능한 전체 flight을 그대로 사용(기존 eval_cross_objective.py는
    # 여기서 covered_row_ids로 필터링했지만, 이 스크립트는 커버 여부와 무관하게 전체를 쓴다).
    restricted_all = [
        {"id": r["row_id"], "origin": airport_map_check[r["origin_str"]],
         "dest": airport_map_check[r["dest_str"]], "dep_time": r["dep_time"], "arr_time": r["arr_time"]}
        for r in rows
        if r["origin_str"] in airport_map_check and r["dest_str"] in airport_map_check
    ]
    print(f"  → 우리 쪽에서 풀 전체 flight 수: {len(restricted_all)}편")

    print()
    print("=" * 70)
    print("3단계: 우리 쪽 — 부분집합 제한 없이 전체 flight로 RL+IP 실행(원래도 100% 커버)")
    print("=" * 70)
    our_result = run_ours_on_subset(restricted_all, ckpt_path, args.lambda_dh, args.ip_time_limit,
                                     compute_gap=args.compute_gap)

    print()
    print("=" * 70)
    print("4단계: Tahir 목적함수(pairing_cost, Eq.1-4)로 우리 solution 재채점")
    print("=" * 70)
    leg_key_by_fid = {leg["flight_id"]: (leg["origin"], leg["dest"], leg["dep_abs"]) for leg in inst["legs"]}
    key_to_tahir_leg = {leg_key_by_fid[leg["flight_id"]]: leg for leg in inst["legs"]}
    row_to_tahir_leg = {r["row_id"]: key_to_tahir_leg[r["tahir_key"]]
                         for r in rows if r["tahir_key"] in key_to_tahir_leg}

    our_pairings = our_result["selected"]
    our_cost_tahir_basis = 0.0
    n_conversion_fail = 0
    flight_use_count = {}

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
    dup_penalty_estimate = 0.0
    for rid in dup_flights:
        leg = row_to_tahir_leg.get(rid)
        if leg is not None:
            extra = flight_use_count[rid] - 1
            dup_penalty_estimate += extra * (GAMMA_DH + LAMBDA_DH * leg["duration"])
    our_full_cost = our_cost_tahir_basis + dup_penalty_estimate

    print(f"  변환 실패(스킵): {n_conversion_fail}개 pairing")
    print(f"  중복 배정된 flight 수(우리식 deadhead): {len(dup_flights)}개")
    print(f"  우리 비용(Tahir 공식 재채점, 순수): {our_cost_tahir_basis:.2f}  "
          f"(+ 중복배정 페널티 포함: {our_full_cost:.2f}) — I2CG/I2CGp 공통(우리 쪽은 한 번만 계산)")

    def run_comparison(tahir_res, label):
        """tahir_res(run_i2cg/run_tahir_i2cgp 결과, i2cg와 동일 shape)로 버전2/버전3을 출력."""
        print()
        print("=" * 70)
        print(f"2단계({label}): Tahir 쪽 — 커버 못 한 flight을 1-leg pairing으로 강제 배정(패딩)")
        print("=" * 70)
        covered_fids = set()
        for pairing in tahir_res["selected_pairings"]:
            for fid in pairing:
                if fid >= 0:
                    covered_fids.add(fid)
        uncovered_fids = all_fids - covered_fids
        print(f"  전체 flight(Tahir 인스턴스 기준): {len(all_fids)}개")
        print(f"  Tahir가 이미 커버: {len(covered_fids)}개 (coverage {tahir_res['coverage']*100:.1f}%)")
        print(f"  1-leg pairing으로 강제 배정할 미커버 flight: {len(uncovered_fids)}개")

        padding_cost = 0.0
        for fid in uncovered_fids:
            padding_cost += pairing_cost([leg_by_fid[fid]])
        tahir_full_cost = tahir_res["mip_obj"] + padding_cost
        tahir_full_n_pairings = len(tahir_res["selected_pairings"]) + len(uncovered_fids)

        print(f"  Tahir 원래 비용(커버 {tahir_res['coverage']*100:.1f}%만): {tahir_res['mip_obj']:>14.2f}")
        print(f"  패딩 비용(미커버 {len(uncovered_fids)}개, 1-leg 강제):        {padding_cost:>14.2f}")
        print(f"  = Tahir 비용(100% 커버로 패딩):                        {tahir_full_cost:>14.2f}")
        print(f"  Tahir pairing 수(100% 커버로 패딩): "
              f"{len(tahir_res['selected_pairings'])} + {len(uncovered_fids)} = {tahir_full_n_pairings}")

        print()
        print("=" * 70)
        print(f"최종({label}) — 양쪽 다 100% 커버 기준, Tahir 자신의 목적함수(mip_obj/pairing_cost)")
        print("=" * 70)
        print(f"  전체 flight 수:                                     {len(restricted_all):>14d}")
        print(f"  Tahir 비용(100% 커버, 미커버분 1-leg 강제 패딩):        {tahir_full_cost:>14.2f}")
        print(f"  우리 비용(100% 커버, Tahir 공식 재채점, 순수):          {our_cost_tahir_basis:>14.2f}")
        print(f"  + 우리식 중복배정 {len(dup_flights)}건을 Tahir deadhead로 환산 시 추가분: "
              f"{dup_penalty_estimate:>10.2f}")
        print(f"  = 우리 비용(중복배정 페널티 포함 추정):                {our_full_cost:>14.2f}")
        print()
        ratio = our_full_cost / tahir_full_cost * 100 - 100
        print(f"  → 둘 다 100% 커버 기준, Tahir({label}) 자신의 objective로 채점해도 우리가 "
              f"{ratio:+.1f}% {'나쁨' if ratio > 0 else '좋음'}")
        print()
        print(f"  (참고: Tahir pairing 수(패딩 포함) {tahir_full_n_pairings} vs 우리 {our_result['n_pairings']})")
        print("  (주의: 1-leg 강제 패딩은 '실제로 실행 가능한 pairing'이 아니라 base 출발/도착")
        print("   조건을 무시한 하한 추정치다 — Tahir가 실제로 이 flight들을 커버하려면 이보다")
        print("   비용이 더 들 수도 있다. 즉 이 비교는 Tahir에게 유리한 쪽으로 편향된 추정.)")

        print()
        print("=" * 70)
        print(f"버전 3({label}) — 그냥 각각 그 자체로(강제/패딩 없이, 각자 native 지표로 비교)")
        print("=" * 70)
        tm = tahir_res["metrics"]
        tahir_dead_h = tm["total_dead_min"] / 60.0
        tahir_fly_h  = tm["total_flying_min"] / 60.0
        print(f"  {'지표':<28} {'Tahir(자기 coverage)':>20} {'우리(100% coverage)':>20}")
        print(f"  {'-'*28} {'-'*20} {'-'*20}")
        print(f"  {'coverage':<28} {tahir_res['coverage']*100:>19.1f}% {our_result['coverage']:>19.1f}%")
        print(f"  {'n_pairings':<28} {len(tahir_res['selected_pairings']):>20d} {our_result['n_pairings']:>20d}")
        print(f"  {'dead time(duty내부, h)':<28} {tahir_dead_h:>20.2f} {our_result['dead_total_h']:>20.2f}")
        print(f"  {'fly time(h)':<28} {tahir_fly_h:>20.2f} {our_result['fly_total_h']:>20.2f}")
        print(f"  {'FTC(Tahir 자체공식)':<28} {tm['ftc_pct_selfdef']:>19.2f}% {'—':>20}")
        print(f"  {'FTC(동일공식 dead/fly)':<28} {tm['ftc_pct_samedef']:>19.2f}% {our_result['ftc_pct']:>19.2f}%")
        print(f"  {'deadhead':<28} {tm['n_deadheads']:>20d} {our_result['deadhead']:>20d}")
        print(f"  {'ManDays(참고)':<28} {tm['man_days']:>20.2f} {our_result['man_days']:>20d}")
        print(f"  {'avg_legs/pairing(참고)':<28} {tm['avg_legs_per_pairing']:>20.2f} {our_result['avg_legs']:>20.2f}")
        if our_result.get("gap_pct") is not None:
            print(f"  {'Gap%(MIP vs LP, 우리쪽만)':<28} {'—':>20} {our_result['gap_pct']:>19.3f}%")

    leg_by_fid = {leg["flight_id"]: leg for leg in inst["legs"]}
    all_fids = set(leg_by_fid.keys())

    run_comparison(tahir_result, "I2CG")
    if args.tahir_method in ("i2cgp", "both") and "i2cgp" in tahir_result:
        run_comparison(tahir_result["i2cgp"], "I2CGp")

    print()
    print("  (주의: coverage가 다른 상태(Tahir는 자기가 커버한 만큼만, 우리는 100%)에서")
    print("   그냥 절대값을 비교한 것 — 커버한 flight 수 자체가 다르므로 이 표 자체로")
    print("   '누가 낫다'를 주장하는 근거로 쓰면 안 됨(log/0709/실험_결과_정리.md §5와")
    print("   같은 종류의 참고용 원시 비교). 버전 1(부분집합)·버전 2(강제 패딩)가")
    print("   coverage confound를 처리한 버전이고, 이건 그 confound를 처리하지 않은")
    print("   '가공 전' 기준선으로만 병기.)")


if __name__ == "__main__":
    main()
