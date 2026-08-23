"""
eval_same_subset.py — Tahir I2CG가 "실제로 커버한 flight 부분집합"만으로 우리 모델을
다시 평가해서, coverage confound 없이 dead time/FTC를 직접 비교한다.

배경 (log/0709/실험_결과_정리.md §5):
  Delta small-scale(3,500편) 비교에서 Tahir I2CG는 coverage 34%에서 멈췄고, 우리는
  구조상 100% 커버를 강제한다. coverage를 정규화(커버한 flight당 dead time/FTC)해도
  우리가 더 나쁘게 나왔는데, 이게 "우리가 비효율적"인지 "Tahir가 쉬운 flight만 골라
  커버해서 유리했던 것"인지 구분이 안 됐다.

이 스크립트가 하는 일 — coverage 자체를 100%로 맞춰서(둘 다 "Tahir가 고른 그 flight들만")
직접 비교:
  1. Tahir/eval_delta.py와 동일한 방식으로 I2CG를 실제로 다시 실행(코드 무수정,
     Tahir/solver/icg.py::run_i2cg를 그대로 호출)해서 실제로 어떤 flight를 커버했는지
     leg 단위로 추출(집계 JSON에는 이 정보가 없어서 원본 실행에서 직접 뽑아야 함).
  2. Tahir가 커버한 flight들을 (origin, dest, dep_abs) 키로 우리 flight 목록과 매칭.
  3. 그 부분집합만으로 우리 RL+IP 파이프라인(evaluation/evaluate_ip.py의 함수들을 그대로 import해서
     재사용 — 로직 수정 없음)을 실행.
  4. 양쪽 다 이 부분집합에서 coverage 100%인 상태로 dead time/FTC/deadhead/ManDays 비교.

기존 evaluation/evaluate_ip.py / baselines/tahir/eval_vs_baseline.py / Tahir/eval_delta.py는 전혀 수정하지 않았고
이 스크립트는 그 위에 새로 얹은 것.

Usage (반드시 저장소 루트에서 실행):
    cd /home/hyrn/ASCP-2026
    source ascp/bin/activate
    python -u baselines/tahir/eval_same_subset.py --checkpoint checkpoints/z2db089m/model_latest.pt
"""

import sys
import os
import argparse
import math
from datetime import datetime, timedelta

import torch
import pandas as pd

_THIS_DIR   = os.path.dirname(os.path.realpath(__file__))            # .../baselines/tahir
_REPO_ROOT  = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
_RL_DIR     = os.path.join(_REPO_ROOT, "RL")
_TAHIR_DIR  = os.environ.get("TAHIR_DIR", os.path.join(os.path.dirname(_REPO_ROOT), "Tahir"))
for p in (_REPO_ROOT, _RL_DIR, _TAHIR_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from evaluation import evaluate_ip                       # 재사용: 모델 로드/pool 수집/IP 함수 전부 그대로 사용
from loader import convert_time, utc_offset_hours          # RL/loader.py, 재사용
from dnn.delta_loader import load_bts_instance              # Tahir, 재사용
from dnn.reference import generate_reference_pairings       # Tahir, 재사용
from solver.icg import run_i2cg                             # Tahir, 재사용
import eval_delta as tahir_eval_delta                        # Tahir, 재사용(compute_pairing_metrics만 호출)
from baselines.tahir.i2cgp_helper import run_tahir_i2cgp     # 신규(§2026-07-14) — I2CGp 경로 추가용

EPOCH = datetime(2000, 1, 1)   # Tahir/dnn/delta_loader.py와 동일 epoch

DEFAULT_CSV        = os.path.join(_RL_DIR, "data", "small-scale", "delta_2019_01_sample_chain.csv")
DEFAULT_TAHIR_CSV  = os.path.join(_RL_DIR, "data", "small-scale", "delta_2019_01_sample_chain_tahir.csv")
DEFAULT_DATE_START = "2019-01-01"
DEFAULT_DATE_END   = "2019-01-07"


# ── 1. 원본 CSV를 직접 파싱해서 우리쪽 표현과 Tahir쪽 키를 같은 row에서 동시에 만든다 ──
# (두 로더 모두 내부적으로 dep_time 기준 재정렬을 하기 때문에, 정렬 후 순서로는 서로
#  매칭이 안 된다 — row 자체에서 나온 (origin, dest, dep_abs)를 키로 써서 정렬과 무관하게
#  매칭한다.)

def parse_raw_rows(csv_path):
    df = pd.read_csv(csv_path)
    df["FL_DATE"] = pd.to_datetime(df["FL_DATE"], format="mixed")
    window_start = df["FL_DATE"].min()

    rows = []
    for i, row in df.iterrows():
        day_offset = (row["FL_DATE"] - window_start).days
        dep_time = convert_time(row["CRS_DEP_TIME"]) + day_offset * 24.0
        dep_time -= utc_offset_hours(row["ORIGIN"], row["FL_DATE"])
        arr_time = dep_time + row["CRS_ELAPSED_TIME"] / 60.0

        hhmm = int(row["CRS_DEP_TIME"])
        dep_h, dep_m = divmod(hhmm, 100)
        dep_dt = row["FL_DATE"].to_pydatetime().replace(
            hour=dep_h % 24, minute=dep_m, second=0, microsecond=0)
        if dep_h >= 24:
            dep_dt += timedelta(days=1)
        dep_abs = int((dep_dt - EPOCH).total_seconds() // 60)   # Tahir 컨벤션(2000-01-01 기준 분)

        rows.append({
            "row_id":    i,
            "origin_str": row["ORIGIN"],
            "dest_str":   row["DEST"],
            "dep_time":   dep_time,
            "arr_time":   arr_time,
            "tahir_key":  (row["ORIGIN"], row["DEST"], dep_abs),
        })
    return rows


# ── 2. Tahir I2CG를 직접 실행해서 실제로 커버한 flight의 (origin,dest,dep_abs) 키를 추출 ──

def _tahir_metrics(r, inst):
    """selected_pairings로부터 dead time/FTC/ManDays 계산(run_i2cg/run_i2cgp 공통)."""
    leg_map = {leg["flight_id"]: leg for leg in inst["legs"]}
    pm = tahir_eval_delta.compute_pairing_metrics(r["selected_pairings"], leg_map)
    fly_for_ftc = pm.pop("_total_flying_for_ftc", 0)
    pm["ftc_pct_selfdef"] = round((r["mip_obj"] - fly_for_ftc) / fly_for_ftc * 100, 4) if fly_for_ftc > 0 else None
    pm["ftc_pct_samedef"] = round(pm["total_dead_min"] / fly_for_ftc * 100, 4) if fly_for_ftc > 0 else None
    return pm


def run_tahir_and_get_covered_keys(tahir_csv, date_start, date_end, max_legs, verbose=True,
                                   tahir_method="i2cg"):
    """tahir_method: "i2cg"(기본, 기존 동작 그대로) | "i2cgp" | "both".

    부분집합 선정(covered_keys)은 항상 I2CG 결과 기준으로 한다(기존 3버전 비교와
    일관성 유지) — I2CGp는 "both"/"i2cgp"일 때 같은 inst/ref로 추가 실행해서
    비교용 지표만 곁들인다(§1-8 검증대로 mip_obj가 I2CG와 동일하면 covered_keys도
    같아야 하지만, 이 subset에서 실제로 같은지는 별도 확인 필요 — 결과에
    "i2cgp_covered_keys_match" 로 표시).
    """
    inst = load_bts_instance(
        carrier="DL", csv_path=tahir_csv,
        date_start=date_start, date_end=date_end, max_legs=max_legs,
    )
    print(f"[Tahir] 로드: {len(inst['legs'])} legs, bases={inst['bases']}")

    ref = generate_reference_pairings(inst, method="cg", verbose=False)
    print(f"[Tahir] Reference pairings: {len(ref)}")

    r = run_i2cg(
        inst, initial_columns=[list(p) for p in ref],
        max_fail=3, max_iter=100, time_limit_mip=300,
        max_labels=300, max_pricing_cols=500, verbose=verbose,
    )
    print(f"[Tahir] I2CG: obj={r['mip_obj']:.2f} coverage={r['coverage']:.3f} "
          f"pairings={len(r['selected_pairings'])}")
    r["metrics"] = _tahir_metrics(r, inst)

    leg_key_by_fid = {
        leg["flight_id"]: (leg["origin"], leg["dest"], leg["dep_abs"])
        for leg in inst["legs"]
    }
    covered_keys = set()
    for pairing in r["selected_pairings"]:
        for fid in pairing:
            if fid >= 0:      # 음수 fid = Tahir 솔버가 내부적으로 삽입한 repositioning leg(실제 flight 아님)
                covered_keys.add(leg_key_by_fid[fid])

    if tahir_method in ("i2cgp", "both"):
        print(f"[Tahir] I2CGp 실행 중 (delta_dnn 가중치, 같은 inst/ref 재사용)...")
        r_p = run_tahir_i2cgp(
            inst, ref, max_fail=3, max_iter=100, time_limit_mip=300,
            max_labels=300, max_pricing_cols=500, verbose=verbose,
        )
        r_p["metrics"] = _tahir_metrics(r_p, inst)
        print(f"[Tahir] I2CGp: obj={r_p['mip_obj']:.2f} coverage={r_p['coverage']:.3f} "
              f"pairings={len(r_p['selected_pairings'])} "
              f"(I2CG 대비 mip_obj 차이: {r_p['mip_obj'] - r['mip_obj']:+.2f})")
        covered_keys_p = set()
        for pairing in r_p["selected_pairings"]:
            for fid in pairing:
                if fid >= 0:
                    covered_keys_p.add(leg_key_by_fid[fid])
        r["i2cgp"] = r_p
        r["i2cgp_covered_keys_match"] = (covered_keys_p == covered_keys)
        print(f"[Tahir] I2CGp covered_keys가 I2CG와 동일한가: {r['i2cgp_covered_keys_match']}")

    return r, covered_keys, inst


# ── 3. 같은 부분집합으로 우리 RL+IP 실행(evaluation/evaluate_ip.py 함수 재사용) ──────────────

def run_ours_on_subset(restricted_flights, checkpoint_path, lambda_dh, ip_time_limit, compute_gap=False):
    # 체크포인트가 small-scale 서브셋(DEFAULT_CSV, 139개 공항)으로 학습됐으므로
    # airport_map도 반드시 같은 서브셋 기준이어야 함 — 전체 delta(145개 공항) 기준으로
    # 만들면 학습 당시 embedding table(139행) 밖 인덱스가 나와 out-of-range로 죽는다.
    airport_map = evaluate_ip.build_airport_map(DEFAULT_CSV)
    base_ids = evaluate_ip.bases_to_ids(evaluate_ip.config.AIRLINE_BASES["delta"], airport_map)
    constraint = evaluate_ip._GET_CONSTRAINT["delta"](base_ids[0])

    ckpt = torch.load(checkpoint_path, map_location=evaluate_ip.DEVICE, weights_only=True)
    n_airports = ckpt.get("n_airports", ckpt["encoder"]["airport_emb.weight"].shape[0])
    encoder = evaluate_ip.FlightEncoder(
        n_airports=n_airports, constraint_dim=len(evaluate_ip.FILM_CONSTRAINT_KEYS)
    ).to(evaluate_ip.DEVICE)
    airport_emb_dim = encoder.airport_emb.embedding_dim
    ckpt_state_dim = ckpt["decoder"]["state_mlp.0.weight"].shape[1]
    n_scalars = ckpt_state_dim - airport_emb_dim * 2 - len(evaluate_ip.FILM_CONSTRAINT_KEYS)
    decoder = evaluate_ip.PointerDecoder(
        constraint_dim=len(evaluate_ip.FILM_CONSTRAINT_KEYS),
        airport_emb_dim=airport_emb_dim, n_scalars=n_scalars,
    ).to(evaluate_ip.DEVICE)
    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])
    encoder.eval()
    decoder.eval()
    print(f"[우리] 모델 로드: {checkpoint_path} (n_airports={n_airports}, n_scalars={n_scalars})")

    # solve_set_covering()은 flight ID가 0..n_flights-1 연속이라고 가정한다(covering
    # constraint를 range(n_flights)로 만듦) — restricted_flights의 원본 id는 3,500편 중
    # 일부만 뽑은 원본 CSV row_id라 듬성듬성하다(예: 3421). 그대로 global_id로 쓰면
    # "uncoverable" 판정과 covering constraint가 다 깨진다 — 반드시 0..len-1로 재부여.
    for new_id, f in enumerate(restricted_flights):
        f["global_id"] = new_id

    print(f"[우리] Pool 수집 중 ({len(restricted_flights)}편 부분집합)...")
    with torch.no_grad():
        pool, covered = evaluate_ip.collect_pool_full(
            [restricted_flights], base_ids, constraint, encoder, decoder,
            n_rollouts_per_chunk=5,
            subset_size=evaluate_ip.config.EPISODE_MAX_FLIGHTS,
            connected_sampler=evaluate_ip.sample_connected_subnet_std,
        )

    n_total = len(restricted_flights)

    # [진단] pool 자체가 "긴 pairing" 후보를 얼마나 갖고 있는지 확인 — IP가 짧은 pairing만
    # 고르는 게 "pool에 긴 후보가 없어서"인지 "있는데 IP가 cost상 짧은 걸 선호해서"인지 구분.
    pool_legs = [p.get("n_legs", len(p["legs"])) for p in pool]
    if pool_legs:
        pool_legs_sorted = sorted(pool_legs)
        n = len(pool_legs_sorted)
        print(f"[진단] pool {n}개 pairing의 legs 분포: "
              f"평균={sum(pool_legs)/n:.2f}, 최댓값={max(pool_legs)}, "
              f"중앙값={pool_legs_sorted[n//2]}, "
              f"legs>=4인 pairing 비율={sum(1 for l in pool_legs if l>=4)/n*100:.1f}%")

    print(f"[우리] IP 풀기 (n_flights={n_total}, pool={len(pool)}, "
          f"time_limit={ip_time_limit}s, lambda_dh={lambda_dh})...")
    result = evaluate_ip.solve_set_covering(
        pool, n_flights=n_total, time_limit=ip_time_limit, lambda_dh=lambda_dh,
    )

    gap_pct = None
    if compute_gap:
        print(f"[우리] LP relaxation 풀기 (Gap% 계산용, pool={len(pool)})...")
        lp_result = evaluate_ip.solve_lp_relaxation(pool, lambda_dh=lambda_dh)
        if lp_result is not None and lp_result["lp_value"]:
            gap_pct = (result["mip_obj"] - lp_result["lp_value"]) / lp_result["lp_value"] * 100
        else:
            print("  [warn] LP relaxation 풀기 실패 — Gap% 계산 불가")

    sel = result["selected"]
    fly_total    = sum(p["fly"] for p in sel) if sel else 0.0
    man_days     = sum(math.ceil(p["elapsed"] / 24.0) for p in sel) if sel else 0
    intra_gap_total = sum(p.get("intra_duty_gap", 0.0) for p in sel) if sel else 0.0
    ftc = intra_gap_total / fly_total * 100 if fly_total > 0 else 0.0
    legs_total   = sum(p.get("n_legs", len(p["legs"])) for p in sel) if sel else 0
    avg_legs     = legs_total / len(sel) if sel else 0.0

    return {
        "n_flights":   n_total,
        "n_pairings":  result["n_pairings"],
        "coverage":    result["coverage"] * 100,
        "deadhead":    result["deadhead_count"],
        "fly_total_h": fly_total,
        "dead_total_h": intra_gap_total,
        "ftc_pct":     ftc,
        "man_days":    man_days,
        "avg_legs":    avg_legs,
        "status":      result["status"],
        "gap_pct":     gap_pct,
        "selected":    sel,   # cross-objective 재채점(eval_cross_objective.py)에서 재사용
    }


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
    parser.add_argument("--tahir-method", default="i2cg", choices=["i2cg", "i2cgp", "both"],
                         help="Tahir 쪽 알고리즘(기본 i2cg, 기존 동작과 동일). "
                              "i2cgp/both는 같은 inst/ref로 Tahir/experiments/delta_dnn "
                              "가중치를 써서 I2CGp도 같이 실행(§2026-07-14 신규)")
    parser.add_argument("--compute-gap", action="store_true",
                         help="우리(RL+IP) 쪽에 LP relaxation을 추가로 풀어 "
                              "Gap%%=(MIP_obj-LP_obj)/LP_obj*100을 계산(evaluation/evaluate_ip.py와 동일 정의)")
    args = parser.parse_args()

    ckpt_path = args.checkpoint if os.path.isabs(args.checkpoint) else os.path.join(_REPO_ROOT, args.checkpoint)

    print("=" * 70)
    print("1단계: 원본 CSV 직접 파싱 (우리쪽 표현 + Tahir쪽 매칭키 동시 생성)")
    print("=" * 70)
    rows = parse_raw_rows(args.csv)
    print(f"  총 {len(rows)}행 파싱 완료")

    print()
    print("=" * 70)
    print("2단계: Tahir I2CG 실행 → 실제로 커버한 flight 추출")
    print("=" * 70)
    tahir_result, covered_keys, _tahir_inst = run_tahir_and_get_covered_keys(
        args.tahir_csv, args.date_start, args.date_end, args.max_legs,
        tahir_method=args.tahir_method,
    )

    key_to_row = {r["tahir_key"]: r["row_id"] for r in rows}
    covered_row_ids = {key_to_row[k] for k in covered_keys if k in key_to_row}
    unmatched = len(covered_keys) - len(covered_row_ids)
    print(f"\n  Tahir 커버 flight: {len(covered_keys)}개, 우리쪽과 키 매칭 성공: "
          f"{len(covered_row_ids)}개 (불일치 {unmatched}개)")
    if unmatched > 0:
        print("  [경고] 매칭 안 된 키가 있음 — dep_abs 계산식 불일치 가능성, 결과 해석 시 유의")

    airport_map_check = evaluate_ip.build_airport_map(DEFAULT_CSV)
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
    print("3단계: 같은 부분집합으로 우리 RL+IP 실행")
    print("=" * 70)
    our_result = run_ours_on_subset(restricted, ckpt_path, args.lambda_dh, args.ip_time_limit,
                                     compute_gap=args.compute_gap)

    tm = tahir_result["metrics"]
    tahir_dead_h = tm["total_dead_min"] / 60.0
    tahir_fly_h  = tm["total_flying_min"] / 60.0
    tahir_n_pairings = len(tahir_result["selected_pairings"])

    has_p = "i2cgp" in tahir_result
    if has_p:
        rp  = tahir_result["i2cgp"]
        tmp = rp["metrics"]
        rp_dead_h = tmp["total_dead_min"] / 60.0
        rp_fly_h  = tmp["total_flying_min"] / 60.0
        rp_n_pairings = len(rp["selected_pairings"])

    print()
    print("=" * 70)
    print(f"최종 비교 — 동일 부분집합 {len(restricted)}편, 전부 coverage 100% 기준")
    print("=" * 70)
    col3 = f"{'Tahir I2CGp':>16}" if has_p else ""
    print(f"  {'지표':<26} {'Tahir I2CG':>16} {col3} {'우리(RL+IP)':>16}")
    print(f"  {'-'*26} {'-'*16} {'-'*16 if has_p else ''} {'-'*16}")

    def row(label, i2cg_v, i2cgp_v, ours_v):
        mid = f"{i2cgp_v:>16}" if has_p else ""
        print(f"  {label:<26} {i2cg_v:>16} {mid} {ours_v:>16}")

    row("n_pairings", f"{tahir_n_pairings:d}", f"{rp_n_pairings:d}" if has_p else "", f"{our_result['n_pairings']:d}")
    row("coverage(정의상 100%)", "100.0%", "100.0%" if has_p else "", f"{our_result['coverage']:.1f}%")
    row("mip_obj(Tahir 자체 목적함수)", f"{tahir_result['mip_obj']:.2f}", f"{rp['mip_obj']:.2f}" if has_p else "", "—")
    row("IP status(진단용)", "-", "-" if has_p else "", our_result['status'])
    row("dead time(duty내부, h)", f"{tahir_dead_h:.2f}", f"{rp_dead_h:.2f}" if has_p else "", f"{our_result['dead_total_h']:.2f}")
    row("fly time(h)", f"{tahir_fly_h:.2f}", f"{rp_fly_h:.2f}" if has_p else "", f"{our_result['fly_total_h']:.2f}")
    row("FTC(자체공식)", f"{tm['ftc_pct_selfdef']:.2f}%", f"{tmp['ftc_pct_selfdef']:.2f}%" if has_p else "", "—")
    row("FTC(동일공식 dead/fly)", f"{tm['ftc_pct_samedef']:.2f}%", f"{tmp['ftc_pct_samedef']:.2f}%" if has_p else "", f"{our_result['ftc_pct']:.2f}%")
    row("deadhead", f"{tm['n_deadheads']:d}", f"{tmp['n_deadheads']:d}" if has_p else "", f"{our_result['deadhead']:d}")
    row("ManDays(참고)", f"{tm['man_days']:.2f}", f"{tmp['man_days']:.2f}" if has_p else "", f"{our_result['man_days']:d}")
    row("avg_legs/pairing(참고)", f"{tm['avg_legs_per_pairing']:.2f}", f"{tmp['avg_legs_per_pairing']:.2f}" if has_p else "", f"{our_result['avg_legs']:.2f}")
    if our_result.get("gap_pct") is not None:
        row("Gap%(MIP vs LP, 우리쪽만)", "-", "-" if has_p else "", f"{our_result['gap_pct']:.3f}%")
    if has_p:
        row("실행시간(s)", f"{tahir_result.get('total_time', float('nan')):.2f}", f"{rp.get('total_time', float('nan')):.2f}", "—")
        row("covered_keys가 I2CG와 동일?", "-", str(tahir_result["i2cgp_covered_keys_match"]), "-")

    print()
    print("  전부 정확히 같은 flight 부분집합(coverage 100%)을 커버한 상태이므로,")
    print("  dead time·FTC(동일공식) 행이 이번 비교의 핵심 — coverage confound 없이 직접 비교 가능.")
    if has_p:
        if tahir_result["i2cgp_covered_keys_match"] and abs(rp["mip_obj"] - tahir_result["mip_obj"]) < 1e-6:
            print("  → I2CG와 I2CGp가 이 subset에서도 완전히 동일 — §1-8(다른 윈도우) 검증이 "
                  "이 3버전 비교에도 그대로 일반화됨을 확인.")
        else:
            print("  → [주의] 이 subset에서는 I2CG와 I2CGp 결과가 다름 — §1-8 검증을 "
                  "이 3버전에 그대로 적용하면 안 되고, 이 결과를 신뢰할 것.")


if __name__ == "__main__":
    main()
