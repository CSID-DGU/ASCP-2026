"""
eval_delta.py — BTS 항공사 데이터에 I²CG/I²CGp 적용

BTS T_ONTIME_MARKETING.csv를 로드해 Tahir 인스턴스 형식으로 변환하고,
I²CG / I²CGp 알고리즘을 실행한다.

Usage:
    # Delta 항공 1일치 (자동 첫날)
    python eval_delta.py --carrier DL

    # 특정 날짜
    python eval_delta.py --carrier DL --date 2019-01-07

    # 1주 범위
    python eval_delta.py --carrier DL --date_start 2019-01-07 --date_end 2019-01-13

    # I²CG만 (DNN 가중치 없어도 실행 가능)
    python eval_delta.py --carrier DL --method i2cg

    # 사용 가능 날짜 목록 확인
    python eval_delta.py --carrier DL --discover
"""

import argparse
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.insert(0, str(Path(__file__).parent))

from dnn.delta_loader import load_bts_instance, discover_bts_instances
from dnn.dataset import build_successor_sets, build_encoders, filter_successors_by_pattern
from dnn.reference import generate_reference_pairings
from solver.icg import run_i2cg, run_i2cgp
from solver.constraints import T_C_MAX, T_R_MIN


def stratified_sample(inst: dict, n_total: int, seed: int = 42) -> dict:
    """날짜별 균등 샘플링: n_total편을 각 날짜에서 균등하게 추출."""
    random.seed(seed)
    day_groups: dict = defaultdict(list)
    for leg in inst["legs"]:
        day_groups[leg["dep_dt"].date()].append(leg)

    days = sorted(day_groups.keys())
    n_days = len(days)
    base_n = n_total // n_days
    remainder = n_total % n_days  # 앞 remainder개 날짜에 1편씩 더

    sampled = []
    for i, day in enumerate(days):
        quota = base_n + (1 if i < remainder else 0)
        pool  = day_groups[day]
        k     = min(quota, len(pool))
        sampled.extend(random.sample(pool, k))

    sampled.sort(key=lambda x: (x["dep_abs"], x["origin"], x["dest"]))
    for i, leg in enumerate(sampled):
        leg["flight_id"] = i + 1  # 1-based: avoids -0 == 0 deadhead encoding bug

    inst = dict(inst)
    inst["legs"] = sampled
    return inst


def compute_pairing_metrics(selected_pairings: list, leg_map: dict) -> dict:
    """
    selected_pairings: list of lists of flight_id (time-ordered by SPPRC)
    leg_map: {flight_id: leg_dict}  (leg_dict has dep_abs, arr_abs, duration)

    Returns additional metrics not computed by the solver itself.
    """
    if not selected_pairings:
        return {
            "n_deadheads": 0, "man_days": 0.0, "n_duties_total": 0,
            "n_overnights": 0, "avg_duties_per_pairing": 0.0,
            "avg_legs_per_pairing": 0.0, "avg_pairing_days": 0.0,
            "total_flying_min": 0, "total_dead_min": 0,
            "total_rest_min": 0, "ftc_pct": None,
        }

    man_days        = 0.0
    n_duties_total  = 0
    n_overnights    = 0
    total_flying    = 0
    total_dead      = 0
    total_rest      = 0
    total_legs      = 0
    mip_obj_check   = 0.0  # 비행시간 합 (FTC 분모)

    n_deadheads_total = 0

    for pairing in selected_pairings:
        if not pairing:
            continue
        # Separate operated (positive fid) and deadhead (negative fid) legs
        operated_fids = [f for f in pairing if f >= 0]
        dh_fids       = [f for f in pairing if f < 0]
        n_deadheads_total += len(dh_fids)

        # Sort all legs (operated + DH) by departure time for gap/duty analysis
        all_fids = pairing
        ordered  = sorted(all_fids, key=lambda f: leg_map[abs(f)]["dep_abs"])
        legs     = [leg_map[abs(f)] for f in ordered]

        # pairing span → man_days
        span_min  = legs[-1]["arr_abs"] - legs[0]["dep_abs"]
        man_days += span_min / 1440.0

        n_duties  = 1
        for k in range(1, len(legs)):
            gap = legs[k]["dep_abs"] - legs[k - 1]["arr_abs"]
            if gap >= T_R_MIN:          # overnight rest → duty 경계
                n_duties  += 1
                total_rest += gap
            elif gap >= 0:              # sit connection (duty 내 대기)
                total_dead += gap

        n_duties_total += n_duties
        n_overnights   += (n_duties - 1)

        # Only operated legs contribute to flying time
        for fid in operated_fids:
            leg = leg_map[fid]
            total_flying  += leg["duration"]
            mip_obj_check += leg["duration"]

        total_legs += len(operated_fids)  # count only operated legs

    n_pairings = len(selected_pairings)
    ftc_pct    = None
    if mip_obj_check > 0:
        # FTC는 별도 호출 시 mip_obj 주입 필요; 여기선 비행시간 합만 저장
        ftc_pct = None  # 아래 run_bts_instance에서 mip_obj로 계산

    return {
        "n_deadheads":           n_deadheads_total,
        "man_days":              round(man_days, 2),
        "n_duties_total":        n_duties_total,
        "n_overnights":          n_overnights,
        "avg_duties_per_pairing": round(n_duties_total / n_pairings, 2),
        "avg_legs_per_pairing":  round(total_legs / n_pairings, 2),
        "avg_pairing_days":      round(man_days / n_pairings, 2),
        "total_flying_min":      total_flying,
        "total_dead_min":        total_dead,
        "total_rest_min":        total_rest,
        "_total_flying_for_ftc": total_flying,  # FTC 계산용 내부값
    }


def _load_dnn(aircraft_type: str, model_dir: Path, enc: dict):
    """DNN 가중치 로드. 없으면 (None, None, None) 반환."""
    import numpy as np
    weights_path = model_dir / f"weights_AT_{aircraft_type}.h5"
    cfg_path     = model_dir / f"model_config_AT_{aircraft_type}.json"
    norm_path    = model_dir / f"norm_AT_{aircraft_type}.json"

    if not weights_path.exists():
        return None, None, None

    import tensorflow as tf
    from dnn.model import build_model

    with open(cfg_path) as f:
        cfg = json.load(f)
    model = build_model(
        n_airports=cfg["n_airports"],
        n_aircraft=cfg["n_aircraft"],
        **cfg.get("hparams", {}),
    )
    dummy = np.zeros((1, 1, 27), dtype=np.float32)
    model(tf.constant(dummy))
    model.load_weights(str(weights_path))

    norm_mean = norm_std = None
    if norm_path.exists():
        with open(norm_path) as f:
            nd = json.load(f)
        norm_mean = __import__("numpy").array(nd["mean"], dtype="float32")
        norm_std  = __import__("numpy").array(nd["std"],  dtype="float32")

    return model, norm_mean, norm_std


def _build_p_psi(inst, ref_pairings, model, enc, norm_mean, norm_std):
    """DNN probability / rank 행렬 구성."""
    import numpy as np
    import tensorflow as tf
    from dnn.dataset import build_xi_matrix
    from solver.column_gen import compute_psi

    legs    = inst["legs"]
    leg_map = {leg["flight_id"]: leg for leg in legs}
    num_cols = list(range(4, 9)) + list(range(13, 18)) + list(range(22, 27))

    succ_raw = build_successor_sets(legs)
    # deadhead(음수 fid)는 leg_map에 없으므로 제거 후 전달
    ref_clean = [[f for f in p if f > 0] for p in ref_pairings]
    ref_clean = [p for p in ref_clean if len(p) > 1]
    succ_flt = filter_successors_by_pattern(legs, succ_raw, ref_clean)

    P_combined: dict = {}
    for base in inst["bases"]:
        for fid, succ in succ_flt.items():
            if not succ:
                P_combined[fid] = {}
                continue
            X = build_xi_matrix(leg_map[fid], succ, leg_map, enc, base)
            if norm_mean is not None:
                X[:, num_cols] = (X[:, num_cols] - norm_mean) / norm_std
            X_in = tf.constant(X[np.newaxis].astype(np.float32))
            probs = model(X_in, training=False).numpy()[0]
            existing = P_combined.get(fid, {})
            for k, jid in enumerate(succ):
                existing[jid] = max(existing.get(jid, 0.0), float(probs[k]))
            P_combined[fid] = existing

    Psi, class_max = compute_psi(P_combined)
    return P_combined, Psi, class_max


def run_bts_instance(
    inst, enc, model_dir: Path,
    method: str, max_iter: int, max_fail: int,
    max_labels: int, max_pricing: int, verbose: bool,
    model_at: str = None,
) -> dict:
    n   = len(inst["legs"])
    src = inst.get("source", "BTS")
    at  = inst["aircraft_type"]
    iid = inst["instance_id"]
    dnn_at = model_at if model_at else at  # DNN 가중치 조회용 타입 (override 가능)

    print(f"\n{'='*60}")
    print(f"  {src} {at} {iid}  ({n} legs, bases={inst['bases']})")
    if model_at and model_at != at:
        print(f"  DNN 가중치: AT_{model_at} (--model_at override)")

    ref = generate_reference_pairings(inst, method="cg", verbose=False)
    print(f"  Reference pairings: {len(ref)} (method=CG)")

    result = {"aircraft_type": at, "instance_id": iid, "source": src, "n_legs": n}
    leg_map = {leg["flight_id"]: leg for leg in inst["legs"]}

    def _enrich(solver_result: dict, r: dict) -> dict:
        """selected_pairings로 추가 지표 계산 후 solver_result에 병합."""
        pm = compute_pairing_metrics(r["selected_pairings"], leg_map)
        fly = pm.pop("_total_flying_for_ftc", 0)
        mip = solver_result.get("mip_obj", 0)
        pm["ftc_pct"] = round((mip - fly) / fly * 100, 4) if fly > 0 else None
        solver_result.update(pm)
        return solver_result

    # ── I²CG ─────────────────────────────────────────────────────────────────
    if method in ("i2cg", "both"):
        print("\n  -- I2CG (full SP) --")
        r = run_i2cg(
            inst, initial_columns=[list(p) for p in ref],
            max_fail=max_fail, max_iter=max_iter,
            time_limit_mip=300, max_labels=max_labels,
            max_pricing_cols=max_pricing, verbose=verbose,
        )
        lp  = r["lp_obj"]
        gap = r.get("gap_pct",
                    abs((r["mip_obj"] - lp) / max(abs(lp), 1.0) * 100)
                    if lp < float("inf") else float("inf"))
        d = {
            "mip_obj": r["mip_obj"], "lp_obj": lp, "gap_pct": gap,
            "coverage": r["coverage"], "n_pairings": len(r["selected_pairings"]),
            "n_uncovered": r.get("n_uncovered", 0),
            "n_iters": r["n_iters"], "n_columns": r["n_columns"],
            "time": r["total_time"], "status": r["status"],
        }
        result["i2cg"] = _enrich(d, r)
        gap_str = f"{gap:.4f}%" if gap < float("inf") else "N/A"
        print(f"  I2CG: obj={r['mip_obj']:.2f} gap={gap_str} "
              f"coverage={r['coverage']:.3f} uncovered={r.get('n_uncovered',0)} "
              f"iters={r['n_iters']} time={r['total_time']:.1f}s")

    # ── I²CGp (DNN 가중치 있을 때만) ─────────────────────────────────────────
    if method in ("i2cgp", "both"):
        model, norm_mean, norm_std = _load_dnn(dnn_at, model_dir, enc)
        if model is None:
            print(f"  I2CGp: 건너뜀 (AT_{dnn_at} 가중치 없음 — "
                  f"--model_at 으로 훈련된 타입(09/319/320/727/757)을 지정하세요)")
            result["i2cgp"] = {"error": f"no weights for {dnn_at}"}
        else:
            print("\n  -- I2CGp (DNN-guided) --")
            P, Psi, class_max = _build_p_psi(
                inst, ref, model, enc, norm_mean, norm_std
            )
            r = run_i2cgp(
                inst, P, Psi, class_max,
                initial_columns=[list(p) for p in ref],
                max_fail=max_fail, max_iter=max_iter,
                time_limit_mip=300, max_labels=max_labels,
                max_pricing_cols=max_pricing, verbose=verbose,
            )
            lp  = r["lp_obj"]
            gap = r.get("gap_pct",
                        abs((r["mip_obj"] - lp) / max(abs(lp), 1.0) * 100)
                        if lp < float("inf") else float("inf"))
            d = {
                "mip_obj": r["mip_obj"], "lp_obj": lp, "gap_pct": gap,
                "coverage": r["coverage"], "n_pairings": len(r["selected_pairings"]),
                "n_uncovered": r.get("n_uncovered", 0),
                "n_iters": r["n_iters"], "n_columns": r["n_columns"],
                "time": r["total_time"], "status": r["status"],
            }
            result["i2cgp"] = _enrich(d, r)
            gap_str = f"{gap:.4f}%" if gap < float("inf") else "N/A"
            print(f"  I2CGp: obj={r['mip_obj']:.2f} gap={gap_str} "
                  f"coverage={r['coverage']:.3f} uncovered={r.get('n_uncovered',0)} "
                  f"iters={r['n_iters']} time={r['total_time']:.1f}s")

    return result


def main():
    parser = argparse.ArgumentParser(description="BTS 데이터 I²CG/I²CGp 평가")
    parser.add_argument("--carrier",    default="DL",
                        help="IATA 캐리어 코드 (기본: DL=Delta)")
    parser.add_argument("--date",       default=None,
                        help="단일 날짜 YYYY-MM-DD")
    parser.add_argument("--date_start", default=None,
                        help="시작 날짜 YYYY-MM-DD")
    parser.add_argument("--date_end",   default=None,
                        help="종료 날짜 YYYY-MM-DD")
    parser.add_argument("--discover",   action="store_true",
                        help="사용 가능 날짜 창 목록 출력 후 종료")
    parser.add_argument("--step_days",  type=int, default=7,
                        help="--discover 창 너비 (기본: 7일)")
    parser.add_argument("--method",     default="i2cg",
                        choices=["i2cg", "i2cgp", "both"],
                        help="실행 메서드 (기본: i2cg)")
    parser.add_argument("--model_dir",  default="experiments/loto",
                        help="DNN 가중치 디렉토리")
    parser.add_argument("--model_at",   default=None,
                        help="DNN 가중치 조회용 항공기 타입 override "
                             "(예: 320 → weights_AT_320.h5 사용). "
                             "BTS 데이터는 캐리어코드(DL 등)가 타입이라 "
                             "훈련된 타입(09/319/320/727/757) 중 하나로 지정 필요")
    parser.add_argument("--max_iter",   type=int, default=100)
    parser.add_argument("--max_fail",   type=int, default=3)
    parser.add_argument("--max_labels", type=int, default=300)
    parser.add_argument("--max_pricing", type=int, default=500)
    parser.add_argument("--max_legs",   type=int, default=500,
                        help="인스턴스 최대 항공편 수 (기본: 500)")
    parser.add_argument("--sample_total", type=int, default=None,
                        help="날짜별 균등 샘플링 후 총 항공편 수 (기본: 샘플링 안 함)")
    parser.add_argument("--seed",        type=int, default=42,
                        help="샘플링 랜덤 시드 (기본: 42)")
    parser.add_argument("--save_sample", action="store_true",
                        help="샘플링된 legs를 CSV로 저장 (--sample_total 사용 시 자동 활성화)")
    parser.add_argument("--verbose",    action="store_true")
    parser.add_argument("--out_dir",    default="experiments",
                        help="결과 저장 디렉토리")
    parser.add_argument("--csv",        default=None,
                        help="BTS CSV 파일 경로 (기본: data/T_ONTIME_MARKETING.csv)")
    args = parser.parse_args()

    # ── discover 모드 ──────────────────────────────────────────────────────────
    if args.discover:
        windows = discover_bts_instances(
            carrier=args.carrier, csv_path=args.csv, step_days=args.step_days
        )
        if not windows:
            print(f"[{args.carrier}] 데이터 없음")
        else:
            print(f"[{args.carrier}] {args.step_days}일 단위 윈도우 목록:")
            print(f"  {'시작':12s} {'종료':12s} {'항공편수':>8}")
            print(f"  {'-'*12} {'-'*12} {'-'*8}")
            for w in windows:
                print(f"  {w['date_start']:12s} {w['date_end']:12s} {w['n_legs']:8d}")
        return

    # ── 인스턴스 로드 ──────────────────────────────────────────────────────────
    print(f"BTS 인스턴스 로드: carrier={args.carrier}", end="")
    if args.date:
        print(f"  date={args.date}")
    elif args.date_start:
        print(f"  {args.date_start} ~ {args.date_end}")
    else:
        print("  (첫 번째 날짜 자동 선택)")

    # sample_total이 max_legs보다 크면 전체를 일단 로드 후 샘플링
    effective_max_legs = args.max_legs
    if args.sample_total and args.sample_total > args.max_legs:
        effective_max_legs = max(args.sample_total * 4, 100_000)

    inst = load_bts_instance(
        carrier=args.carrier,
        date=args.date,
        date_start=args.date_start,
        date_end=args.date_end,
        csv_path=args.csv,
        max_legs=effective_max_legs,
    )
    print(f"로드 완료: {len(inst['legs'])} legs, "
          f"{len(inst['airports'])} 공항, bases={inst['bases']}")

    # ── 날짜별 균등 샘플링 ─────────────────────────────────────────────────────
    if args.sample_total:
        inst = stratified_sample(inst, args.sample_total, seed=args.seed)
        print(f"샘플링 후: {len(inst['legs'])} legs (날짜별 균등, seed={args.seed})")

    # ── 샘플 CSV 저장 ──────────────────────────────────────────────────────────
    if args.save_sample or args.sample_total:
        import csv as _csv
        out_dir_p = Path(args.out_dir)
        out_dir_p.mkdir(exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        csv_path = out_dir_p / f"sample_{args.carrier}_{inst['instance_id']}_{ts}.csv"
        fields = ["flight_id", "origin", "dest", "dep_dt", "arr_dt",
                  "dep_abs", "arr_abs", "dep_day", "dep_min",
                  "arr_day", "arr_min", "duration", "aircraft_type"]
        with open(csv_path, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(inst["legs"])
        print(f"샘플 CSV 저장: {csv_path}")

    # ── 인코더 (I²CGp용; I²CG만 쓸 경우에도 초기화) ──────────────────────────
    # [2026-07-09] 평가 인스턴스만으로 매번 build_encoders()를 새로 만들면
    # airport→index 매핑이 그 인스턴스에 등장하는 공항 집합/정렬 순서에 좌우돼서
    # 학습 시점 encoder와 어긋날 수 있다(임베딩이 엉뚱한 공항을 가리킬 위험) —
    # train_delta.py가 저장해둔 enc_AT_{model_at}.json이 있으면 그걸 재사용한다.
    model_dir = Path(args.model_dir)
    enc_at = args.model_at or inst["aircraft_type"]
    enc_path = model_dir / f"enc_AT_{enc_at}.json"
    if enc_path.exists():
        with open(enc_path) as f:
            enc = json.load(f)
        print(f"[encoder] 학습 시점 encoder 재사용: {enc_path}")
    else:
        enc = build_encoders([inst])

    # ── 평가 실행 ──────────────────────────────────────────────────────────────
    result = run_bts_instance(
        inst=inst, enc=enc, model_dir=model_dir,
        method=args.method,
        max_iter=args.max_iter, max_fail=args.max_fail,
        max_labels=args.max_labels, max_pricing=args.max_pricing,
        verbose=args.verbose,
        model_at=args.model_at,
    )

    # ── 결과 저장 ──────────────────────────────────────────────────────────────
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    tag = f"{args.carrier}_{inst['instance_id']}"
    json_path = out_dir / f"bts_{tag}_results.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n결과 저장: {json_path}")

    # 요약 출력
    print("\n" + "="*70)
    print(f"  {'지표':<28} {'I2CG':>12} {'I2CGp':>12}")
    print(f"  {'-'*28} {'-'*12} {'-'*12}")

    def _fmt(d, key, fmt=".2f", suffix=""):
        if d is None or "error" in d:
            return "N/A"
        v = d.get(key)
        if v is None:
            return "N/A"
        if isinstance(v, float):
            return f"{v:{fmt}}{suffix}"
        return str(v) + suffix

    i2cg  = result.get("i2cg")
    i2cgp = result.get("i2cgp")

    rows = [
        ("mip_obj (min)",         "mip_obj",               ".1f", ""),
        ("lp_obj (min)",          "lp_obj",                ".1f", ""),
        ("gap_pct (%)",           "gap_pct",               ".4f", "%"),
        ("coverage (%)",          None,                    "",    ""),   # special
        ("n_pairings",            "n_pairings",            "d",   ""),
        ("n_uncovered",           "n_uncovered",           "d",   ""),
        ("n_deadheads",           "n_deadheads",           "d",   ""),
        ("man_days",              "man_days",              ".2f", ""),
        ("n_overnights",          "n_overnights",          "d",   ""),
        ("n_duties_total",        "n_duties_total",        "d",   ""),
        ("avg_duties/pairing",    "avg_duties_per_pairing",".2f", ""),
        ("avg_legs/pairing",      "avg_legs_per_pairing",  ".2f", ""),
        ("avg_pairing_days",      "avg_pairing_days",      ".2f", ""),
        ("total_flying (min)",    "total_flying_min",      "d",   ""),
        ("total_dead (min)",      "total_dead_min",        "d",   ""),
        ("total_rest (min)",      "total_rest_min",        "d",   ""),
        ("FTC (%)",               "ftc_pct",               ".4f", "%"),
        ("n_iters",               "n_iters",               "d",   ""),
        ("n_columns",             "n_columns",             "d",   ""),
        ("time (s)",              "time",                  ".2f", ""),
    ]

    for label, key, fmt, suffix in rows:
        if key is None:  # coverage special case (stored as fraction)
            def _cov(d):
                if d is None or "error" in d: return "N/A"
                v = d.get("coverage")
                return f"{v*100:.1f}%" if v is not None else "N/A"
            c1, c2 = _cov(i2cg), _cov(i2cgp)
        elif fmt == "d":
            def _fmti(d, k=key):
                if d is None or "error" in d: return "N/A"
                v = d.get(k)
                return str(v) if v is not None else "N/A"
            c1, c2 = _fmti(i2cg), _fmti(i2cgp)
        else:
            c1 = _fmt(i2cg,  key, fmt, suffix)
            c2 = _fmt(i2cgp, key, fmt, suffix)
        print(f"  {label:<28} {c1:>12} {c2:>12}")

    print("="*70)


if __name__ == "__main__":
    main()
