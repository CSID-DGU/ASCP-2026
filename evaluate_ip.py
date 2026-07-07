"""
evaluate_ip.py — 전체 1개월 데이터 커버 평가

evaluate_ip.py는 에피소드당 최대 600편 subset만 커버 (n_max=600).
이 스크립트는 전체 1개월을 window_days 단위 비겹침 윈도우로 나눠 모든 편을 커버한다.

흐름:
  1. 전체 CSV를 5일 단위 비겹침 윈도우로 분할 → 전역 flight ID 부여
  2. 윈도우별로 subset(600편) rollout 반복 → pairing pool 누적 (전역 ID 기반)
  3. 모든 윈도우 처리 후 IP → 전체 flight 커버

주의사항:
  - 윈도우 경계를 걸치는 pairing은 생성 불가 (window boundary 한계)
  - IP 규모: ~73,836편 × pool pairings → CBC 수 시간 소요 가능 (ip_time_limit 조정)
"""

import sys
import os
import math
import random
import argparse

import torch
import pandas as pd

sys.path.insert(0, "RL")

DEVICE = torch.device("cpu")

from loader import load_flights_rolling, build_airport_map, bases_to_ids, sample_connected_subnet as sample_connected_subnet_std
from turkish.loader_turkish import (
    parse_legs_dir, build_airport_map_turkish, load_flights_rolling_turkish,
    sample_connected_subnet as sample_connected_subnet_turkish,
    ZEREN_FEB_FILE, ZEREN_FEB_WINDOW,
)
from turkish.constraints_turkish import get_turkish_constraints as get_turkish_constraints_hb
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
    "turkish": get_turkish_constraints_hb,  # HB1/HB2 비대칭 종료 허용
}
from model import FlightEncoder, PointerDecoder
from set_partition import solve_set_covering
from utils import constraint_to_tensor, flights_to_tensors
from rollout import rollout_with_pairings, set_environment
import config


# ── 0. Turkish 윈도우 로더 ────────────────────────────────────────────────────

def load_windows_turkish(turkish_df, airport_map, window_days=5):
    """Turkish .legs 데이터를 window_days 단위 비겹침 윈도우로 분할, 전역 ID 부여."""
    dates = sorted(turkish_df["dep_date_utc"].unique())
    n_days = len(dates)
    windows = []
    global_offset = 0

    for offset in range(0, n_days, window_days):
        wf = load_flights_rolling_turkish(
            window_days=window_days,
            offset_days=offset,
            airport_map=airport_map,
            df=turkish_df,
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


# ── 1. 전체 데이터를 윈도우별로 로드, 전역 ID 부여 ─────────────────────────────

def load_windows_with_global_ids(data_path, airport_map, window_days=5, use_utc=False):
    """전체 CSV를 window_days 단위 비겹침 윈도우로 분할, 전역 ID 부여.

    use_utc: True면 dep_time을 UTC로 앵커링(RL/loader.py 참고) — 이 옵션으로 학습한
        체크포인트를 평가할 때만 켤 것. 기존 체크포인트에 켜면 OOD.

    Returns:
        windows    : list of flight lists. 각 flight에 'global_id' 필드 추가됨.
        n_total    : 전체 flight 수 (= 전역 ID 상한)
    """
    df = pd.read_csv(data_path)
    df = df[["ORIGIN", "DEST", "CRS_DEP_TIME", "CRS_ARR_TIME", "CRS_ELAPSED_TIME", "FL_DATE"]].dropna()
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
            use_utc=use_utc,
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
    """Connectivity-aware subset sampling with random coverage guarantee.

    BFS로 base 출발편에서 시작해 연결 가능한 편을 우선 선택하되,
    subset의 BFS_RATIO만큼만 BFS로 채우고 나머지는 window 전체에서 pure random 선택.

    이유: BFS가 hub-and-spoke 밀집 구간을 빠르게 채우면 연결이 없는 고립 항공편은
         어떤 rollout에도 포함되지 않아 pool coverage가 ~85%에 그침
         15% random 보장 → 300 rollouts 기준 항공편당 기대 포함 횟수 ≈5회 → 누락 확률 <1%

    Args:
        window_flights: 윈도우 내 전체 flight 리스트
        subset_size:    선택할 편 수 (config.EPISODE_MAX_FLIGHTS)
        base_id:        crew base 공항 정수 ID
        constraint:     제약 dict (min_conn, max_conn 단위: hours)
    """
    BFS_RATIO = 0.85  # subset의 85%는 BFS-connected (RL multi-leg 밀도 유지)

    min_conn = constraint.get("min_conn", 0.65)   # hours
    max_conn = constraint.get("max_conn", 9.0)     # hours

    # 출발 공항별 인덱스
    by_origin = {}
    for f in window_flights:
        by_origin.setdefault(f["origin"], []).append(f)

    selected_ids = set()
    selected = []

    # BFS phase: subset의 BFS_RATIO까지만 채움
    bfs_quota = max(1, int(subset_size * BFS_RATIO))
    base_departs = [f for f in window_flights if f["origin"] == base_id]
    random.shuffle(base_departs)
    queue = list(base_departs)

    while queue and len(selected) < bfs_quota:
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

    # Random phase: 나머지 슬롯을 window 전체에서 pure random 선택
    # → 고립 항공편이 매 rollout마다 일정 확률로 포함되어 pool coverage 보장
    remaining = [f for f in window_flights if f["id"] not in selected_ids]
    random.shuffle(remaining)
    for f in remaining[:subset_size - len(selected)]:
        selected_ids.add(f["id"])
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


# ── 3-1. 윈도우 전체를 connectivity 기반 chunk로 파티셔닝 ─────────────────────

def partition_connected_chunks(window_flights, base_ids, chunk_size, connected_sampler):
    """윈도우 전체를 connected-subnet 기반 chunk로 파티셔닝 (완전 소진할 때까지 반복).

    학습(RL/loader.py, RL/turkish/loader_turkish.py)의 sample_connected_subnet과 동일한 로직으로
    각 chunk를 만들되, remaining이 빌 때까지 반복해 모든 flight이 정확히 1개 chunk에
    속하도록 해 coverage 100%를 유지한다 (star graph 버그 수정 후 eval도 학습과 같은
    연결 밀도 분포를 보도록 맞춤).
    """
    remaining = list(window_flights)
    chunks = []
    while remaining:
        for i, f in enumerate(remaining):
            f["id"] = i
        base_id = random.choice(base_ids)
        chunk = connected_sampler(remaining, base_id, chunk_size)
        if not chunk:
            chunk = sorted(remaining, key=lambda f: f["dep_time"])[:chunk_size]
        chosen_gids = {f["global_id"] for f in chunk}
        remaining = [f for f in remaining if f["global_id"] not in chosen_gids]
        chunks.append(sorted(chunk, key=lambda f: f["dep_time"]))
    return chunks


# ── 4. 전체 윈도우 pool 수집 ──────────────────────────────────────────────────

def collect_pool_full(windows, base_ids, constraint, encoder, decoder,
                      n_rollouts_per_chunk=5,
                      subset_size=config.EPISODE_MAX_FLIGHTS,
                      connected_sampler=sample_connected_subnet_std):
    """모든 윈도우에서 rollout → 전역 ID 기반 pairing pool 생성.

    window를 connected_sampler(학습과 동일한 sample_connected_subnet)로 connectivity 기반
    chunk로 분할한 뒤, 각 chunk에 n_rollouts_per_chunk번 stochastic + 1번 greedy rollout을
    수행한다. remaining이 빌 때까지 반복 파티셔닝하므로 모든 flight이 최소 1번 rollout에
    포함되어 coverage 100%를 보장하면서, 동시에 학습 때와 같은 연결 밀도를 유지한다.
    """
    pool = {}
    covered_global = set()
    max_time = 5 * 24.0
    base_id_set = set(base_ids)

    for w_idx, window_flights in enumerate(windows):
        if not window_flights:
            continue

        window_all_ids = set(f["global_id"] for f in window_flights)
        window_covered = set()

        chunks = partition_connected_chunks(window_flights, base_ids, subset_size, connected_sampler)

        # chunk 내 base 출발편 보장: 없으면 window에서 가장 가까운 base 편을 복사해 주입
        # (connected-subnet 파티셔닝은 대부분 base 출발편을 포함하지만 tail chunk 등 예외 대비)
        for c_idx, chunk in enumerate(chunks):
            if not any(f["origin"] in base_id_set for f in chunk):
                chunk_gids = {f["global_id"] for f in chunk}
                candidates = [f for f in window_flights
                              if f["origin"] in base_id_set and f["global_id"] not in chunk_gids]
                if candidates:
                    inject = min(candidates,
                                 key=lambda f: abs(f["dep_time"] - chunk[0]["dep_time"]))
                    new_chunk = sorted([{**inject}] + list(chunk[:-1]),
                                       key=lambda f: f["dep_time"])
                    chunks[c_idx] = new_chunk

        print(f"\n[Window {w_idx + 1}/{len(windows)}] {len(window_flights)}편 → {len(chunks)}개 chunk", flush=True)

        rollout_count = 0
        for c_idx, chunk in enumerate(chunks):
            for local_id, f in enumerate(chunk):
                f["local_id"] = local_id

            base_id = random.choice(base_ids)
            c_b = {**constraint, "base_airport": base_id}

            for _ in range(n_rollouts_per_chunk):
                try:
                    pairings = rollout_subset_global(chunk, c_b, encoder, decoder, max_time, greedy=False)
                except Exception as e:
                    print(f"  [warn] stochastic rollout 실패 (chunk={c_idx}): {e}", flush=True)
                    continue
                for p in pairings:
                    key = tuple(sorted(p["legs"]))
                    if key not in pool or p["cost"] < pool[key]["cost"]:
                        pool[key] = p
                    window_covered.update(p["legs"])
                    covered_global.update(p["legs"])
                rollout_count += 1

            try:
                pairings = rollout_subset_global(chunk, c_b, encoder, decoder, max_time, greedy=True)
            except Exception as e:
                print(f"  [warn] greedy rollout 실패 (chunk={c_idx}): {e}", flush=True)
                continue
            for p in pairings:
                key = tuple(sorted(p["legs"]))
                if key not in pool or p["cost"] < pool[key]["cost"]:
                    pool[key] = p
                window_covered.update(p["legs"])
                covered_global.update(p["legs"])
            rollout_count += 1

        print(
            f"  총 {rollout_count}회 rollout: "
            f"window {len(window_covered)}/{len(window_all_ids)}편 커버, "
            f"pool={len(pool)}",
            flush=True,
        )

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
    n_rollouts_per_chunk=5,
    window_days=5,
    subset_size=config.EPISODE_MAX_FLIGHTS,
    bases=None,
    ip_time_limit=3600,
    lambda_dh=1.0,
    device="cpu",
    turkish_files=None,
    use_utc=False,
    use_wandb=False,
    wandb_project="ASCP-2026-paper",
):
    """flight 커버 평가. data_path 미지정 시 config.AIRLINE_DATA[airline] 사용.

    소규모 데이터(예: 1주일 sample) 평가 시:
        data_path=<sample.csv>, window_days=1, n_rollouts_per_chunk=3

    use_wandb=True면 wandb에 eval 설정 + 콘솔 로그 + 최종 결과 지표를 기록한다
    (job_type="eval" — 학습 curve와는 별개 run으로 남는다).
    """
    global DEVICE
    DEVICE = torch.device(device)
    set_environment(airline)

    wandb_run = None
    if use_wandb:
        import wandb
        wandb_run = wandb.init(
            project=wandb_project,
            job_type="eval",
            name=f"eval-{airline}-{os.path.basename(checkpoint_path)}",
            config=dict(
                checkpoint=checkpoint_path, airline=airline,
                subset_size=subset_size, window_days=window_days,
                n_rollouts_per_chunk=n_rollouts_per_chunk,
                ip_time_limit=ip_time_limit, lambda_dh=lambda_dh,
                use_utc=use_utc,
            ),
        )

    if data_path is None:
        data_path = config.AIRLINE_DATA[airline]
    if bases is None:
        bases = config.AIRLINE_BASES[airline]

    # checkpoint를 먼저 로드해 vocab 크기 확인 — multi-airline 모델(n_airports=168)은
    # 통합 공항 맵이 필요. 단일 항공사 맵으로 빌드하면 ID 불일치로 임베딩 오류 발생.
    ckpt       = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    n_airports = ckpt.get("n_airports",
                          ckpt["encoder"]["airport_emb.weight"].shape[0])

    _turkish_df = None
    if airline == "turkish":
        # turkish_files 미지정 시 Zeren Feb 벤치마크 윈도우(15,742편, 목표 15,738 대비
        # 오차 0.03%)를 기본값으로 사용
        if turkish_files is None:
            _turkish_df = parse_legs_dir(data_path, files=[ZEREN_FEB_FILE], date_range=ZEREN_FEB_WINDOW)
        else:
            _turkish_df = parse_legs_dir(data_path, files=turkish_files)
        airport_map = build_airport_map_turkish(df=_turkish_df)
    else:
        if n_airports > 145:
            # Turkish(.legs 디렉토리)는 BTS CSV 로더로 처리 불가 → 제외
            map_paths = [v for k, v in config.AIRLINE_DATA.items() if k != "turkish"]
        else:
            map_paths = data_path
        airport_map = build_airport_map(map_paths)
    base_ids = bases_to_ids(list(bases), airport_map)

    encoder = FlightEncoder(n_airports=n_airports, constraint_dim=len(FILM_CONSTRAINT_KEYS)).to(DEVICE)
    # 체크포인트 state_vec 차원 자동 감지 (v8=78dim/7scalars, v13+=79dim/8scalars)
    airport_emb_dim = encoder.airport_emb.embedding_dim
    ckpt_state_dim  = ckpt["decoder"]["state_mlp.0.weight"].shape[1]
    n_scalars = ckpt_state_dim - airport_emb_dim * 2 - len(FILM_CONSTRAINT_KEYS)
    decoder = PointerDecoder(constraint_dim=len(FILM_CONSTRAINT_KEYS), airport_emb_dim=airport_emb_dim, n_scalars=n_scalars).to(DEVICE)
    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])
    encoder.eval()
    decoder.eval()

    if airline == "turkish":
        constraint = _GET_CONSTRAINT[airline](base_ids[0], base_ids=base_ids)
    else:
        constraint = _GET_CONSTRAINT[airline](base_ids[0])

    print(f"\n전체 데이터 로드 중 ({airline}, window_days={window_days})...", flush=True)
    if airline == "turkish":
        windows, n_total = load_windows_turkish(_turkish_df, airport_map, window_days)
    else:
        windows, n_total = load_windows_with_global_ids(data_path, airport_map, window_days, use_utc=use_utc)
    print(f"총 {n_total}편, {len(windows)}개 윈도우", flush=True)

    connected_sampler = sample_connected_subnet_turkish if airline == "turkish" else sample_connected_subnet_std

    print(f"\nPool 수집 중 (rollouts/chunk={n_rollouts_per_chunk}, subset={subset_size})...", flush=True)
    with torch.no_grad():
        pool, covered = collect_pool_full(
            windows, base_ids, constraint, encoder, decoder,
            n_rollouts_per_chunk=n_rollouts_per_chunk,
            subset_size=subset_size,
            connected_sampler=connected_sampler,
        )

    print(f"\nIP 풀기 (n_flights={n_total}, pool={len(pool)}, time_limit={ip_time_limit}s, lambda_dh={lambda_dh})...", flush=True)
    result = solve_set_covering(pool, n_flights=n_total, time_limit=ip_time_limit, lambda_dh=lambda_dh)

    sel          = result["selected"]
    fly_total    = sum(p["fly"]                         for p in sel) if sel else 0.0
    raw_dead_total = sum(p.get("dead_time", p["cost"])  for p in sel) if sel else 0.0
    legs_total   = sum(p.get("n_legs", len(p["legs"])) for p in sel) if sel else 0
    duties_total = sum(p.get("n_duties", 1)            for p in sel) if sel else 0
    man_days     = sum(math.ceil(p["elapsed"] / 24.0)  for p in sel) if sel else 0
    avg_legs     = legs_total   / len(sel) if sel else 0.0
    avg_duties   = duties_total / len(sel) if sel else 0.0
    # FTC는 duty 내부 gap만 반영(overnight 초과 제외) — cost는 그대로 둠(ManDays 유인 보존)
    intra_gap_total    = sum(p.get("intra_duty_gap", 0.0)    for p in sel) if sel else 0.0
    inter_excess_total = sum(p.get("inter_duty_excess", 0.0) for p in sel) if sel else 0.0
    # dead time 총합은 FTC와 같은 기준(overnight 초과 제외, duty-내부 gap만)으로 표시 —
    # raw_dead_total(cost 계산 기준, overnight 초과 포함)은 참고용으로 별도 표시
    dead_total = intra_gap_total
    ftc = intra_gap_total / fly_total * 100 if fly_total > 0 else 0.0

    print()
    print("=" * 60)
    print(f"결과 (전체 {n_total}편 커버)")
    print("=" * 60)
    print(f"  pairing 수:        {result['n_pairings']}")
    print(f"  ManDays:           {man_days}")
    print(f"  coverage:          {result['coverage'] * 100:.1f}%")
    print(f"  uncoverable:       {result['uncoverable']}개 flight")
    print(f"  deadhead:          {result['deadhead_count']}개 flight")
    print(f"  fly time:          {fly_total:.2f}h")
    print(f"  dead time(duty-내부 gap만, overnight 제외): {dead_total:.2f}h")
    print(f"  (참고) raw dead time(overnight 초과 포함, cost 계산 기준): {raw_dead_total:.2f}h")
    print(f"    - duty 내부 연결 gap:     {intra_gap_total:.2f}h ({intra_gap_total/raw_dead_total*100 if raw_dead_total>0 else 0:.1f}%)")
    print(f"    - duty 간 초과 대기(>min_rest): {inter_excess_total:.2f}h ({inter_excess_total/raw_dead_total*100 if raw_dead_total>0 else 0:.1f}%)")
    print(f"  FTC:               {ftc:.2f}%")
    print(f"  avg legs/pairing:  {avg_legs:.2f}")
    print(f"  avg duties/pairing:{avg_duties:.2f}")
    print(f"  IP status:         {result['status']}")

    if wandb_run is not None:
        import wandb
        wandb.log({
            "n_pairings":       result["n_pairings"],
            "man_days":         man_days,
            "coverage":         result["coverage"] * 100,
            "uncoverable":      result["uncoverable"],
            "deadhead":         result["deadhead_count"],
            "fly_time":         fly_total,
            "dead_time":        dead_total,
            "raw_dead_time":    raw_dead_total,
            "intra_duty_gap":   intra_gap_total,
            "inter_duty_excess": inter_excess_total,
            "ftc":              ftc,
            "avg_legs":         avg_legs,
            "avg_duties":       avg_duties,
            "ip_status":        result["status"],
        })
        wandb.finish()

    return result


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="전체 1개월 flight 커버 평가")
    parser.add_argument("checkpoint", help="체크포인트 파일 경로 (예: checkpoints/jbkwcdk3/phase2_best.pt)")
    parser.add_argument("--airline",   default="delta", choices=["delta", "alaska", "jetblue", "turkish"])
    parser.add_argument("--data-path", default=None,
                        help="CSV 경로. 미지정 시 config.AIRLINE_DATA[airline] 사용. "
                             "소규모 sample 평가 시 지정 (예: RL/data/sample_DL_*.csv)")
    parser.add_argument("--n-rollouts-per-chunk", type=int, default=5,
                        help="chunk당 stochastic rollout 수. window를 subset_size 단위 순차 chunk로 분할 (기본: 5)")
    parser.add_argument("--window-days", type=int, default=5,
                        help="윈도우 크기(일). 소규모(1주) 데이터는 1 권장 (기본: 5)")
    parser.add_argument("--subset-size", type=int, default=config.EPISODE_MAX_FLIGHTS,
                        help=f"rollout당 flight 수 (기본: {config.EPISODE_MAX_FLIGHTS})")
    parser.add_argument("--ip-time-limit", type=int, default=3600,
                        help="CBC solver 제한 시간 초 (기본: 3600)")
    parser.add_argument("--lambda-dh", type=float, default=1.0,
                        help="DH 패널티 가중치 (기본: 1.0)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--turkish-files", nargs="+", default=None,
                        help="Turkish 전용. 사용할 .legs 파일 이름 목록. 미지정 시 Zeren Feb "
                             "벤치마크 윈도우(tt201402.legs, 2/1~3/8, 15,742편) 기본 사용. "
                             "명시 지정 시 날짜 필터 없이 해당 파일 전체 사용.")
    parser.add_argument("--use-utc", action="store_true",
                        help="dep_time을 UTC 절대시간으로 앵커링. --use-utc로 학습한 체크포인트만 "
                             "이 옵션 켠 채로 평가할 것 — 기존 체크포인트에 켜면 OOD")
    parser.add_argument("--wandb", action="store_true",
                        help="eval 설정+콘솔 로그+최종 결과 지표를 wandb에 기록 (job_type=eval)")
    parser.add_argument("--wandb-project", default="ASCP-2026-paper")
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
        n_rollouts_per_chunk=args.n_rollouts_per_chunk,
        window_days=args.window_days,
        subset_size=args.subset_size,
        ip_time_limit=args.ip_time_limit,
        lambda_dh=args.lambda_dh,
        device=args.device,
        turkish_files=args.turkish_files,
        use_utc=args.use_utc,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
    )
