"""
compute_v2_native_metrics.py — 버전2(강제 100% 커버리지, eval_cross_objective_full_coverage.py)의
Tahir측 native 지표(dead time/fly time/FTC/ManDays/avg_legs)를 실측 계산한다.

배경: eval_cross_objective_full_coverage.py(버전2 본체)는 cross-objective 비용(Tahir
pairing_cost 기준)만 산출하고, dead time/fly time 같은 native 지표는 계산하지
않는다(1-leg 패딩 pairing은 pairing_cost만 쓰고 dead/fly를 따로 집계하지 않기 때문).
이 스크립트는 그 native 지표를 추가로 계산한다 — 우리(RL+IP) 쪽은 버전2·버전3이
완전히 같은 실행(전체 3,500편, 100% coverage)이라 재실행하지 않고 기존 로그
(log/0713/eval_tahir_full_coverage_z2db089m_v2.out의 "버전3" 표)값을 그대로 쓴다.

계산 방법:
  1. eval_same_subset.run_tahir_and_get_covered_keys()로 I2CG를 그대로 재실행(결정론적
     알고리즘이라 재실행해도 mip_obj/coverage/pairings 완전히 동일 — 실측 확인됨).
  2. Tahir가 못 커버한 flight(uncovered_fids)를 1-leg pairing으로 간주했을 때의 native
     지표: dead=0(연결 없음), fly=해당 flight duration 합, man_days=해당 flight
     span(=duration) 합/1440 — eval_delta.py::compute_pairing_metrics()와 동일한 정의.
  3. 실제 397개 pairing의 기존 metrics(tm)에 패딩분을 합산.

이 스크립트는 추가로 **cross-objective 비용의 상한(upper bound) 추정치**도 계산한다
(하한은 eval_cross_objective_full_coverage.py의 기존 787,757.83). 1-leg 패딩은
base 출발/도착 요건을 무시한 값이라 "이 flight을 커버하는 가장 싼 경우"(하한)다.
실제로 pairing이 base에서 출발해 base로 복귀하려면 그 flight의 출발공항까지/
도착공항에서부터 각각 한 번씩 빈 이동(deadhead/repositioning)이 필요할 수
있는데, 실제 base↔공항 간 이동시간(거리) 데이터가 없으므로, **그 flight
자체의 비행시간만큼 왕복 데드헤드가 필요하다고 가정**하고 Tahir 자신의
phi^DH 공식(Eq.3: GAMMA_DH + LAMBDA_DH*duration)을 그대로 적용해 2회(왕복)
얹은 값을 상한으로 쓴다 — `cost_upper = max(240,duration) + 2*(GAMMA_DH +
LAMBDA_DH*duration)`. 이건 정확한 상한이 아니라 **"실제 값이 이보다 크지는
않을 것"이라는 보수적 근사치**이며, 실제 base-공항 거리 데이터가 있으면
더 정확하게 다시 계산해야 한다(§5 참고).

Usage:
    cd /home/hyrn/ASCP-2026
    source ascp/bin/activate
    python -u baselines/tahir/compute_v2_native_metrics.py
"""
import sys
import os

_THIS_DIR  = os.path.dirname(os.path.realpath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
_RL_DIR    = os.path.join(_REPO_ROOT, "RL")
_TAHIR_DIR = os.environ.get("TAHIR_DIR", os.path.join(os.path.dirname(_REPO_ROOT), "Tahir"))
for p in (_REPO_ROOT, _RL_DIR, _TAHIR_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from baselines.tahir.eval_same_subset import (
    run_tahir_and_get_covered_keys,
    DEFAULT_TAHIR_CSV, DEFAULT_DATE_START, DEFAULT_DATE_END,
)
from solver.constraints import pairing_cost, GAMMA_DH, LAMBDA_DH

MAX_LEGS = 4000  # eval_cross_objective_full_coverage.py 기본값(--max-legs)과 동일
OUR_FULL_COST = 1_398_404.42  # log/0713/eval_tahir_full_coverage_z2db089m_v2.out 최종값(불변, 재사용)


def main():
    print("Tahir I2CG 재실행 중 (결정론적, ~30초) ...")
    tahir_result, covered_keys, inst = run_tahir_and_get_covered_keys(
        DEFAULT_TAHIR_CSV, DEFAULT_DATE_START, DEFAULT_DATE_END, MAX_LEGS, verbose=False,
    )

    leg_by_fid = {leg["flight_id"]: leg for leg in inst["legs"]}
    covered_fids = set()
    for pairing in tahir_result["selected_pairings"]:
        for fid in pairing:
            if fid >= 0:
                covered_fids.add(fid)
    all_fids = set(leg_by_fid.keys())
    uncovered_fids = all_fids - covered_fids

    print(f"전체 flight: {len(all_fids)}, Tahir 커버: {len(covered_fids)}, "
          f"패딩(1-leg 강제) 대상: {len(uncovered_fids)}")

    pad_fly_min  = 0.0
    pad_man_days = 0.0
    for fid in uncovered_fids:
        leg  = leg_by_fid[fid]
        pad_fly_min  += leg["duration"]
        pad_man_days += (leg["arr_abs"] - leg["dep_abs"]) / 1440.0
    pad_dead_min   = 0.0                 # 1-leg라 연결 gap 없음
    pad_n_pairings = len(uncovered_fids)
    pad_n_legs     = len(uncovered_fids)  # 1-leg pairing이라 pairing당 operated leg 1개

    # ── cross-objective 비용 하한/상한 ──
    padding_cost_lower = 0.0
    padding_cost_upper = 0.0
    for fid in uncovered_fids:
        leg = leg_by_fid[fid]
        c_lower = pairing_cost([leg])                       # 기존 버전2(하한): base 왕복 무시
        c_upper = c_lower + 2 * (GAMMA_DH + LAMBDA_DH * leg["duration"])  # +왕복 데드헤드 근사
        padding_cost_lower += c_lower
        padding_cost_upper += c_upper

    tm = tahir_result["metrics"]
    real_dead_min   = tm["total_dead_min"]
    real_fly_min    = tm["total_flying_min"]
    real_man_days   = tm["man_days"]
    real_n_pairings = len(tahir_result["selected_pairings"])
    real_n_legs     = len(covered_fids)   # DH=0인 실행이라 covered flight 수 = operated leg 총합

    comb_dead_min   = real_dead_min + pad_dead_min
    comb_fly_min    = real_fly_min + pad_fly_min
    comb_man_days   = real_man_days + pad_man_days
    comb_n_pairings = real_n_pairings + pad_n_pairings
    comb_n_legs     = real_n_legs + pad_n_legs
    comb_avg_legs   = comb_n_legs / comb_n_pairings
    comb_ftc_samedef = comb_dead_min / comb_fly_min * 100 if comb_fly_min > 0 else None

    print()
    print("=" * 70)
    print("버전2 Tahir측 native 지표 (397 real pairing + 1690 padding, 100% coverage)")
    print("=" * 70)
    print(f"  real({real_n_pairings})   dead={real_dead_min/60:.2f}h fly={real_fly_min/60:.2f}h "
          f"man_days={real_man_days:.2f} legs={real_n_legs}")
    print(f"  pad({pad_n_pairings})  dead={pad_dead_min/60:.2f}h fly={pad_fly_min/60:.2f}h "
          f"man_days={pad_man_days:.2f} legs={pad_n_legs}")
    print(f"  combined     dead={comb_dead_min/60:.2f}h fly={comb_fly_min/60:.2f}h "
          f"man_days={comb_man_days:.2f} legs={comb_n_legs} pairings={comb_n_pairings}")
    print(f"  avg_legs/pairing = {comb_avg_legs:.2f}")
    print(f"  FTC(동일공식, dead/fly*100) = {comb_ftc_samedef:.2f}%")
    print(f"  n_pairings = {comb_n_pairings}")

    print()
    print("=" * 70)
    print("cross-objective 비용 하한 vs 상한 (Tahir 목적함수 기준)")
    print("=" * 70)
    mip_obj = tahir_result["mip_obj"]
    tahir_full_cost_lower = mip_obj + padding_cost_lower
    tahir_full_cost_upper = mip_obj + padding_cost_upper
    ratio_lower = OUR_FULL_COST / tahir_full_cost_lower * 100 - 100
    ratio_upper = OUR_FULL_COST / tahir_full_cost_upper * 100 - 100
    print(f"  Tahir mip_obj(real 397개):                    {mip_obj:>14.2f}")
    print(f"  패딩 비용 하한(1-leg, base 왕복 무시):           {padding_cost_lower:>14.2f}")
    print(f"  패딩 비용 상한(1-leg + 왕복 데드헤드 근사):       {padding_cost_upper:>14.2f}")
    print(f"  Tahir 전체 비용 하한:                          {tahir_full_cost_lower:>14.2f}")
    print(f"  Tahir 전체 비용 상한:                          {tahir_full_cost_upper:>14.2f}")
    print(f"  우리 비용(중복배정 포함, 불변):                 {OUR_FULL_COST:>14.2f}")
    print(f"  격차(하한 기준, Tahir에게 유리):                {ratio_lower:>+8.1f}%")
    print(f"  격차(상한 기준, Tahir에게 불리):                {ratio_upper:>+8.1f}%")


if __name__ == "__main__":
    main()
