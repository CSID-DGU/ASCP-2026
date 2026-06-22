"""
evaluate_ip_full.py — 전체 1개월 데이터 커버 평가

evaluate_ip.py는 에피소드당 최대 600편 subset만 커버 (n_max=600).
이 스크립트는 전체 1개월을 window_days 단위 비겹침 윈도우로 나눠 모든 편을 커버한다.

흐름:
  1. 전체 CSV를 5일 단위 비겹침 윈도우로 분할 → 전역 flight ID 부여
  2. 윈도우별로 subset(600편) rollout 반복 → pairing pool 누적 (전역 ID 기반)
  3. 모든 윈도우 처리 후 IP → 전체 flight 커버

주의사항:
  - 윈도우 경계를 걸치는 pairing은 생성 불가 (window boundary 한계)
  - IP 규모: ~73,836편 × pool pairings → CBC 수 시간 소요 가능 (ip_time_limit 조정)
  - evaluate_ip.py는 그대로 유지 (롤백 가능)
"""

import sys
import os
import random
import argparse

import torch
import pandas as pd

sys.path.insert(0, "RL")

DEVICE = torch.device("cpu")

from loader import load_flights_rolling, build_airport_map, bases_to_ids
from constraints import (
    get_delta_constraints,
    get_alaska_constraints,
    get_jetblue_constraints,
    FILM_CONSTRAINT_KEYS,
)
_GET_CONSTRAINT = {
    "delta":   get_delta_constraints,
    "alaska":  get_alaska_constraints,
    "jetblue": get_jetblue_constraints,
}
from model import FlightEncoder, PointerDecoder
from set_partition import solve_set_covering
from utils import constraint_to_tensor, flights_to_tensors
from rollout import rollout_with_pairings
import config


# ── 1. 전체 데이터를 윈도우별로 로드, 전역 ID 부여 ─────────────────────────────

def load_windows_with_global_ids(data_path, airport_map, window_days=5):
    """전체 CSV를 window_days 단위 비겹침 윈도우로 분할, 전역 ID 부여.

    Returns:
        windows    : list of flight lists. 각 flight에 'global_id' 필드 추가됨.
        n_total    : 전체 flight 수 (= 전역 ID 상한)
    """
    df = pd.read_csv(data_path)
    df = df[["ORIGIN", "DEST", "CRS_DEP_TIME", "CRS_ARR_TIME", "FL_DATE"]].dropna()
    df["FL_DATE"] = pd.to_datetime(df["FL_DATE"], format="mixed")

    dates = sorted(df["FL_DATE"].unique())
    n_days = len(dates)

    windows = []
    global_offset = 0

    for offset in range(0, n_days, window_days):
        wf = load_flights_rolling(
            data_path,
            window_days=window_days,
            offset_days=offset,
            airport_map=airport_map,
            n_max=None,
            df=df,
        )
        for f in wf:
            f["global_id"] = global_offset + f["id"]
        global_offset += len(wf)
        windows.append(wf)
        print(
            f"  window offset={offset:2d}: {len(wf):5d}편 "
            f"(global {global_offset - len(wf)} ~ {global_offset - 1})",
            flush=True,
        )

    return windows, global_offset


# ── 2. 윈도우 내 subset 샘플링 ────────────────────────────────────────────────

def sample_connected_subset(window_flights, subset_size, base_id, constraint):
    """Connectivity-aware subset sampling.

    BFS로 base 출발편에서 시작해 연결 가능한 편을 우선 선택한다.
    무작위 샘플링 대비 subset 내 연결 밀도가 높아 RL이 multi-leg pairing을 형성할 수 있다.

    Args:
        window_flights: 윈도우 내 전체 flight 리스트
        subset_size:    선택할 편 수 (config.EPISODE_MAX_FLIGHTS)
        base_id:        crew base 공항 정수 ID
        constraint:     제약 dict (min_conn, max_conn 단위: hours)
    """
    min_conn = constraint.get("min_conn", 0.65)   # hours
    max_conn = constraint.get("max_conn", 9.0)     # hours

    # 출발 공항별 인덱스
    by_origin = {}
    for f in window_flights:
        by_origin.setdefault(f["origin"], []).append(f)

    selected_ids = set()
    selected = []

    # BFS seed: base 출발편 무작위 순서
    base_departs = [f for f in window_flights if f["origin"] == base_id]
    random.shuffle(base_departs)
    queue = list(base_departs)

    while queue and len(selected) < subset_size:
        f = queue.pop(0)
        if f["id"] in selected_ids:
            continue
        selected_ids.add(f["id"])
        selected.append(f)

        # 이 편 도착 후 연결 가능한 다음 편을 queue에 추가
        nexts = [
            g for g in by_origin.get(f["dest"], [])
            if g["id"] not in selected_ids
            and min_conn <= g["dep_time"] - f["arr_time"] <= max_conn
        ]
        random.shuffle(nexts)
        queue.extend(nexts)

    # BFS로 못 채웠으면 base 인접편(출발/도착)으로 보충
    if len(selected) < subset_size:
        others = [f for f in window_flights
                  if f["id"] not in selected_ids
                  and (f["origin"] == base_id or f["dest"] == base_id)]
        random.shuffle(others)
        for f in others[:subset_size - len(selected)]:
            selected_ids.add(f["id"])
            selected.append(f)

    # 그래도 부족하면 나머지 임의 보충
    if len(selected) < subset_size:
        others = [f for f in window_flights if f["id"] not in selected_ids]
        random.shuffle(others)
        for f in others[:subset_size - len(selected)]:
            selected.append(f)

    selected = sorted(selected, key=lambda f: f["dep_time"])
    for local_id, f in enumerate(selected):
        f["local_id"] = local_id

    return selected


# ── 3. subset rollout → global_id 기반 pairings ──────────────────────────────

def rollout_subset_global(subset, constraint, encoder, decoder, max_time, greedy=False):
    """subset(600편, global_id 포함)으로 rollout → global_id 기반 pairings 반환."""
    local_flights = [{**f, "id": f["local_id"]} for f in subset]

    origins, dests, dep_norm, arr_norm, fly_norm = flights_to_tensors(
        local_flights, max_time, device=DEVICE
    )
    c_tensor = constraint_to_tensor(constraint, device=DEVICE)

    with torch.no_grad():
        encoded = encoder(origins, dests, dep_norm, arr_norm, fly_norm, c_tensor)
        raw_pairings = rollout_with_pairings(
            local_flights, constraint, encoder, decoder, encoded,
            greedy=greedy, device=DEVICE,
        )

    id_map = {f["local_id"]: f["global_id"] for f in subset}
    for p in raw_pairings:
        p["legs"] = [id_map[leg] for leg in p["legs"]]

    return raw_pairings


# ── 4. 전체 윈도우 pool 수집 ──────────────────────────────────────────────────

def collect_pool_full(windows, base_ids, constraint, encoder, decoder,
                      n_rollouts_per_window=200,
                      subset_size=config.EPISODE_MAX_FLIGHTS):
    """모든 윈도우에서 rollout → 전역 ID 기반 pairing pool 생성."""
    pool = {}
    covered_global = set()
    max_time = 5 * 24.0

    for w_idx, window_flights in enumerate(windows):
        if not window_flights:
            continue

        window_all_ids = set(f["global_id"] for f in window_flights)
        window_covered = set()

        print(f"\n[Window {w_idx + 1}/{len(windows)}] {len(window_flights)}편", flush=True)

        for rollout_i in range(n_rollouts_per_window):
            base_id = random.choice(base_ids)
            c_b     = {**constraint, "base_airport": base_id}
            subset  = sample_connected_subset(window_flights, subset_size, base_id, c_b)

            try:
                pairings = rollout_subset_global(subset, c_b, encoder, decoder, max_time)
            except Exception as e:
                print(f"  [warn] rollout 실패 (base={base_id}): {e}", flush=True)
                continue

            for p in pairings:
                key = tuple(sorted(p["legs"]))
                if key not in pool or p["cost"] < pool[key]["cost"]:
                    pool[key] = p
                window_covered.update(p["legs"])
                covered_global.update(p["legs"])

            if (rollout_i + 1) % 50 == 0:
                print(
                    f"  rollout {rollout_i + 1}/{n_rollouts_per_window}: "
                    f"window {len(window_covered)}/{len(window_all_ids)}편 커버, "
                    f"pool={len(pool)}",
                    flush=True,
                )

            if window_covered >= window_all_ids:
                print(f"  → 윈도우 전체 커버 달성 (rollout {rollout_i + 1}번)", flush=True)
                break

        uncov = len(window_all_ids - window_covered)
        if uncov > 0:
            print(f"  미커버: {uncov}편 (IP에서 uncoverable로 처리됨)", flush=True)

    total_flights = sum(len(w) for w in windows)
    print(f"\n총 pool: {len(pool)}개 pairing")
    print(f"전체 커버: {len(covered_global)}/{total_flights}편")
    return list(pool.values()), covered_global


# ── 5. 메인 평가 함수 ──────────────────────────────────────────────────────────

def evaluate_full(
    checkpoint_path,
    airline="delta",
    data_path=None,
    n_rollouts_per_window=200,
    window_days=5,
    subset_size=config.EPISODE_MAX_FLIGHTS,
    bases=("ATL", "DTW", "MSP", "JFK", "LAX", "SEA", "SLC"),
    ip_time_limit=3600,
    device="cpu",
):
    """flight 커버 평가. data_path 미지정 시 config.AIRLINE_DATA[airline] 사용.

    소규모 데이터(예: 1주일 sample) 평가 시:
        data_path=<sample.csv>, window_days=1, n_rollouts_per_window=100
    """
    global DEVICE
    DEVICE = torch.device(device)

    data_path = data_path or config.AIRLINE_DATA[airline]

    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    n_airports = ckpt.get("n_airports", ckpt["encoder"]["airport_emb.weight"].shape[0])

    if n_airports > 145:
        map_paths = list(config.AIRLINE_DATA.values())
    else:
        map_paths = data_path
    airport_map = build_airport_map(map_paths)
    base_ids    = bases_to_ids(list(bases), airport_map)

    encoder = FlightEncoder(n_airports=n_airports, constraint_dim=len(FILM_CONSTRAINT_KEYS)).to(DEVICE)
    decoder = PointerDecoder(constraint_dim=len(FILM_CONSTRAINT_KEYS)).to(DEVICE)
    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])
    encoder.eval()
    decoder.eval()

    constraint = _GET_CONSTRAINT[airline](base_ids[0])

    print(f"\n전체 데이터 로드 중 ({airline}, window_days={window_days})...", flush=True)
    windows, n_total = load_windows_with_global_ids(data_path, airport_map, window_days)
    print(f"총 {n_total}편, {len(windows)}개 윈도우", flush=True)

    print(f"\nPool 수집 중 (rollouts/window={n_rollouts_per_window}, subset={subset_size})...", flush=True)
    with torch.no_grad():
        pool, covered = collect_pool_full(
            windows, base_ids, constraint, encoder, decoder,
            n_rollouts_per_window=n_rollouts_per_window,
            subset_size=subset_size,
        )

    print(f"\nIP 풀기 (n_flights={n_total}, pool={len(pool)}, time_limit={ip_time_limit}s)...", flush=True)
    result = solve_set_covering(pool, n_flights=n_total, time_limit=ip_time_limit)

    sel        = result["selected"]
    fly_total  = sum(p["fly"]                         for p in sel) if sel else 0.0
    dead_total = sum(p.get("dead_time", p["cost"])    for p in sel) if sel else 0.0
    legs_total = sum(p.get("n_legs", len(p["legs"])) for p in sel) if sel else 0
    avg_legs   = legs_total / len(sel) if sel else 0.0
    ftc        = dead_total / fly_total * 100 if fly_total > 0 else 0.0

    print()
    print("=" * 60)
    print(f"결과 (전체 {n_total}편 커버)")
    print("=" * 60)
    print(f"  pairing 수:       {result['n_pairings']}")
    print(f"  coverage:         {result['coverage'] * 100:.1f}%")
    print(f"  uncoverable:      {result['uncoverable']}개 flight")
    print(f"  deadhead:         {result['deadhead_count']}개 flight")
    print(f"  fly time:         {fly_total:.2f}h")
    print(f"  dead time:        {dead_total:.2f}h")
    print(f"  FTC:              {ftc:.2f}%")
    print(f"  avg legs/pairing: {avg_legs:.2f}")
    print(f"  IP status:        {result['status']}")

    return result


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="전체 1개월 flight 커버 평가")
    parser.add_argument("checkpoint", help="체크포인트 파일 경로 (예: checkpoints/jbkwcdk3/phase2_best.pt)")
    parser.add_argument("--airline",   default="delta", choices=["delta", "alaska", "jetblue"])
    parser.add_argument("--data-path", default=None,
                        help="CSV 경로. 미지정 시 config.AIRLINE_DATA[airline] 사용. "
                             "소규모 sample 평가 시 지정 (예: RL/data/sample_DL_*.csv)")
    parser.add_argument("--n-rollouts-per-window", type=int, default=200,
                        help="윈도우당 rollout 수. 많을수록 커버리지↑ (기본: 200)")
    parser.add_argument("--window-days", type=int, default=5,
                        help="윈도우 크기(일). 소규모(1주) 데이터는 1 권장 (기본: 5)")
    parser.add_argument("--subset-size", type=int, default=config.EPISODE_MAX_FLIGHTS,
                        help=f"rollout당 flight 수 (기본: {config.EPISODE_MAX_FLIGHTS})")
    parser.add_argument("--ip-time-limit", type=int, default=3600,
                        help="CBC solver 제한 시간 초 (기본: 3600)")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    ckpt = args.checkpoint
    if not os.path.exists(ckpt):
        candidate = os.path.join("checkpoints", ckpt)
        if os.path.exists(candidate):
            ckpt = candidate

    evaluate_full(
        checkpoint_path=ckpt,
        airline=args.airline,
        data_path=args.data_path,
        n_rollouts_per_window=args.n_rollouts_per_window,
        window_days=args.window_days,
        subset_size=args.subset_size,
        ip_time_limit=args.ip_time_limit,
        device=args.device,
    )
