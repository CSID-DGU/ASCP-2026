"""
RL vs Tahir I²CGp gap comparison — small-scale CPPSC 벤치마크.

Usage (반드시 저장소 루트에서 실행 — 루트의 eval_vs_baseline_tahir.py 심볼릭 링크 경유):
    cd /home/hyrn/ASCP-2026
    source ascp/bin/activate
    python -u eval_vs_baseline_tahir.py --checkpoint checkpoints/z2db089m/model_latest.pt \
        --at 320 --tightness 1

Requires:
  - 학습된 모델 체크포인트 (예: checkpoints/z2db089m/model_latest.pt)
  - Tahir 저장소가 /home/hyrn/Tahir 심볼릭 링크로 연결되어 있을 것 (log/0708/실험설계.md §0-4)
  - Tahir/experiments/i2cgp_results.json (또는 --results로 경로 지정)

Gap formula:
    gap = (n_RL_pairings - n_baseline_pairings) / n_baseline_pairings * 100%
    양수 = RL이 baseline보다 나쁨(pairing 더 많음), 음수 = RL이 더 좋음.

[2026-07-09 현대화] 예전 버전은 이 파일 안에 자체 run_greedy/state_to_vec를 손으로
재구현했는데, 그 사이 모델이 gap_bias·79차원 state_vec·END_DUTY를 흡수한 step() 등으로
바뀌면서 완전히 어긋났다(RL.environment.step_end_duty 자체가 사라짐). 재구현 대신
evaluation/evaluate_ip.py가 쓰는 것과 동일한 RL/rollout.py::rollout_with_pairings, RL/utils.py의
state_to_vec/flights_to_tensors를 그대로 재사용하도록 바꿔 이후 모델 변경에도 같이
현행화되게 했다.
"""

import sys
import os
import json
import argparse
import torch

_THIS_DIR = os.path.dirname(os.path.realpath(__file__))          # .../RL/baseline/Tahir
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)          # `model`, `RL.*` 패키지 임포트용
_RL_DIR = os.path.join(_REPO_ROOT, "RL")
if _RL_DIR not in sys.path:
    sys.path.insert(0, _RL_DIR)             # rollout.py/utils.py 내부의 flat import(`import config` 등)용

from model import FlightEncoder, PointerDecoder
from cppsc_loader import load_cppsc_flights, get_cppsc_constraints
from constraints import FILM_CONSTRAINT_KEYS
from utils import constraint_to_tensor, flights_to_tensors
from rollout import rollout_with_pairings

DEVICE = torch.device("cpu")

TAHIR_RESULTS = os.path.join(_REPO_ROOT, "Tahir", "experiments", "i2cgp_results.json")

ALL_TYPES = ["727", "09", "94", "95", "757", "319", "320"]

WINDOW_DAYS = 5
MAX_TIME = WINDOW_DAYS * 24.0   # evaluation/evaluate_ip.py와 동일한 고정 정규화 분모(시간)


# ── greedy rollout (rollout.py 재사용) ───────────────────────────────────────

def run_greedy(flights, constraint, encoder, decoder):
    """CPPSC 절대시간(hour, 인스턴스마다 시작 오프셋이 다름)을 윈도우 시작=0으로
    재앵커링 후 greedy rollout. RL/loader.py가 학습 데이터를 윈도우 시작 기준
    0에 가깝게 앵커링하는 것과 동일한 관례를 맞춰 encoder 입력 분포를 훈련 분포에 맞춘다."""
    shift = min(f["dep_time"] for f in flights)
    local = [
        {**f, "dep_time": f["dep_time"] - shift, "arr_time": f["arr_time"] - shift}
        for f in flights
    ]

    origins, dests, dep_norm, arr_norm, fly_norm = flights_to_tensors(local, MAX_TIME, device=DEVICE)
    c_tensor = constraint_to_tensor(constraint, device=DEVICE)

    with torch.no_grad():
        encoded = encoder(origins, dests, dep_norm, arr_norm, fly_norm, c_tensor)
        pairings = rollout_with_pairings(
            local, constraint, encoder, decoder, encoded, greedy=True, device=DEVICE,
        )

    covered = {leg for p in pairings for leg in p["legs"]}
    n_uncovered = len(flights) - len(covered)
    coverage = len(covered) / len(flights) * 100
    return len(pairings), n_uncovered, coverage


# ── load baseline results from Tahir ─────────────────────────────────────────

def load_baseline(results_path: str):
    """
    Returns dict keyed by (aircraft_type, instance_id) ->
      {'n_pairings': int, 'coverage': float, 'method': str}

    Prefers i2cgp over i2cg when both are present.
    """
    if not os.path.exists(results_path):
        return {}

    with open(results_path) as f:
        data = json.load(f)

    baseline = {}
    for entry in data:
        if entry.get("source") != "CPPSC":
            continue
        at   = entry["aircraft_type"]
        iid  = entry["instance_id"]
        rec  = entry.get("i2cgp") or entry.get("i2cg")
        if rec and "n_pairings" in rec:
            baseline[(at, iid)] = {
                "n_pairings": rec["n_pairings"],
                "coverage":   rec.get("coverage", float("nan")),
                "method":     "i2cgp" if "i2cgp" in entry else "i2cg",
            }
    return baseline


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/model_latest.pt",
                         help="저장소 루트 기준 상대경로 (예: checkpoints/z2db089m/model_latest.pt)")
    parser.add_argument("--at", default=None, help="Filter: aircraft type (e.g. '09')")
    parser.add_argument("--tightness", type=int, default=None,
                        help="Filter: tightness level 1-5 (default: all)")
    parser.add_argument("--results", default=TAHIR_RESULTS,
                        help="Path to Tahir i2cgp_results.json")
    args = parser.parse_args()

    # ── load model (evaluation/evaluate_ip.py와 동일한 자동 감지 로직: v8=78dim/7scalars, v13+=79dim/8scalars) ──
    ckpt_path = args.checkpoint if os.path.isabs(args.checkpoint) else os.path.join(_REPO_ROOT, args.checkpoint)
    if not os.path.exists(ckpt_path):
        print(f"[ERROR] No checkpoint found at {ckpt_path}")
        print("  Train first: python experiments/train.py")
        sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    n_airports = ckpt.get("n_airports", ckpt["encoder"]["airport_emb.weight"].shape[0])

    encoder = FlightEncoder(n_airports=n_airports, constraint_dim=len(FILM_CONSTRAINT_KEYS)).to(DEVICE)
    airport_emb_dim = encoder.airport_emb.embedding_dim
    ckpt_state_dim  = ckpt["decoder"]["state_mlp.0.weight"].shape[1]
    n_scalars = ckpt_state_dim - airport_emb_dim * 2 - len(FILM_CONSTRAINT_KEYS)
    decoder = PointerDecoder(constraint_dim=len(FILM_CONSTRAINT_KEYS), airport_emb_dim=airport_emb_dim,
                              n_scalars=n_scalars).to(DEVICE)
    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])
    encoder.eval()
    decoder.eval()
    print(f"모델 로드: {ckpt_path}  (n_airports={n_airports}, n_scalars={n_scalars})")

    # ── load baseline ──
    baseline = load_baseline(args.results)
    if baseline:
        print(f"Tahir baseline: {len(baseline)} CPPSC entries from {args.results}")
    else:
        print(f"[WARN] No Tahir baseline found at {args.results}. Gap column will show N/A.")

    # ── determine which (type, tightness) to evaluate ──
    types      = [args.at] if args.at else ALL_TYPES
    tightnesses = [args.tightness] if args.tightness else list(range(1, 6))

    print()
    header = (f"{'AT':>5}  {'T':>2}  {'Legs':>6}  {'RL pairs':>8}  {'BL pairs':>8}  "
              f"{'RL cov':>7}  {'BL cov':>7}  {'Gap%(raw)':>10}  {'Gap%(norm)':>11}  {'Method'}")
    print(header)
    print("-" * len(header))
    print("  Gap%(raw)  = pairing 개수만 비교 (coverage 다르면 confound — 참고용)")
    print("  Gap%(norm) = pairing/covered-flight 비율로 정규화한 gap (coverage 차이 보정, 주 지표)")
    print()

    rows = []
    for at in types:
        for t in tightnesses:
            try:
                flights, airport_map, base_ids = load_cppsc_flights(at, t)
            except (FileNotFoundError, Exception) as e:
                continue

            # Use first base airport as base_airport constraint
            base_airport = base_ids[0] if base_ids else 0
            constraint = get_cppsc_constraints(base_airport)

            # RL must use same n_airports as training; remap if needed
            # (CPPSC airport count may differ from training set)
            n_ap_cppsc = max(f["origin"] for f in flights) + 1
            n_ap_cppsc = max(n_ap_cppsc, max(f["dest"] for f in flights) + 1)

            if n_ap_cppsc > n_airports:
                print(f"  AT_{at} t={t}: SKIP — instance has {n_ap_cppsc} airports, "
                      f"model trained on {n_airports}")
                continue

            n_rl, n_unc, cov_rl = run_greedy(flights, constraint, encoder, decoder)

            bl = baseline.get((at, t))
            if bl:
                n_bl   = bl["n_pairings"]
                cov_bl = bl["coverage"] * 100  # JSON은 0~1 fraction
                gap_raw = (n_rl - n_bl) / max(n_bl, 1) * 100

                # coverage가 서로 다르면 pairing 개수만 비교하는 gap_raw는 confound됨
                # (더 많이 커버할수록 pairing도 자연히 늘어남) → pairing/covered-flight
                # 비율로 정규화해 "커버한 flight 1개당 pairing을 얼마나 쓰는지"로 비교
                ppf_rl = n_rl / max(cov_rl / 100 * len(flights), 1e-9)
                ppf_bl = n_bl / max(cov_bl / 100 * len(flights), 1e-9)
                gap_norm = (ppf_rl - ppf_bl) / ppf_bl * 100

                gap_raw_str  = f"{gap_raw:+.2f}%"
                gap_norm_str = f"{gap_norm:+.2f}%"
                meth_str     = bl["method"]
            else:
                n_bl = -1
                cov_bl = float("nan")
                gap_raw_str  = "N/A"
                gap_norm_str = "N/A"
                meth_str     = "-"

            bl_str  = f"{n_bl:8d}"   if n_bl >= 0 else "       -"
            cov_bl_str = f"{cov_bl:6.1f}%" if n_bl >= 0 else "      -"
            print(f"  {at:>3}  {t:>2}  {len(flights):>6}  {n_rl:>8d}  {bl_str}  "
                  f"{cov_rl:>6.1f}%  {cov_bl_str}  {gap_raw_str:>10}  {gap_norm_str:>11}  {meth_str}")
            rows.append((at, t, len(flights), n_rl, n_bl, cov_rl, cov_bl))

    if rows:
        gaps_raw, gaps_norm = [], []
        for _, _, n_flights, n_rl, n_bl, cov_rl, cov_bl in rows:
            if n_bl > 0:
                gaps_raw.append((n_rl - n_bl) / n_bl * 100)
                ppf_rl = n_rl / max(cov_rl / 100 * n_flights, 1e-9)
                ppf_bl = n_bl / max(cov_bl / 100 * n_flights, 1e-9)
                gaps_norm.append((ppf_rl - ppf_bl) / ppf_bl * 100)
        if gaps_raw:
            print()
            print(f"평균 gap_raw  (baseline 있는 {len(gaps_raw)}개): {sum(gaps_raw)/len(gaps_raw):+.2f}%  "
                  f"(coverage 다르면 confound — 참고용)")
            print(f"평균 gap_norm (baseline 있는 {len(gaps_norm)}개): {sum(gaps_norm)/len(gaps_norm):+.2f}%  "
                  f"(coverage 보정 — 이 값을 결과 문서에 쓸 것)")
            print(f"  양수 = RL이 baseline보다 (커버한 flight당) pairing 더 많이 씀 (나쁨)")
            print(f"  음수 = RL이 baseline보다 (커버한 flight당) pairing 더 적게 씀 (좋음)")
        if any(n_bl > 0 for _, _, _, _, n_bl, cov_rl, cov_bl in rows if abs(cov_rl - cov_bl) > 1.0):
            print()
            print("[참고] RL coverage와 baseline coverage가 1%p 이상 차이나는 조합이 있음 — "
                  "gap_raw 대신 gap_norm 기준으로 해석할 것.")


if __name__ == "__main__":
    main()
