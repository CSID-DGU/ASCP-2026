"""
evaluation/evaluate_ip.py — 전체 1개월 데이터 커버 평가

evaluation/evaluate_ip.py는 에피소드당 최대 600편 subset만 커버 (n_max=600).
이 스크립트는 전체 1개월을 window_days 단위 비겹침 윈도우로 나눠 모든 편을 커버한다.

  1. Split the full CSV into window_days-sized non-overlapping windows -> assign global flight IDs
  2. Per window: partition into connectivity-preserving chunks -> stochastic
     + greedy policy rollouts per chunk -> accumulate the legal candidate
     pool Cθ (keyed by global ID)
  3. After all windows are processed, solve the restricted set-covering MIP
     over Cθ (final selection) to cover the whole schedule

Notes:
  - A pairing cannot be generated across a window boundary (a window-boundary limitation)
  - IP scale: ~73,836 flights x pool pairings -> CBC can take hours (tune ip_time_limit)
"""

import sys
import os
import json
import math
import random
import argparse

import torch
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "RL"))

DEVICE = torch.device("cpu")

from loader import (
    load_flights_rolling, build_airport_map, bases_to_ids,
    sample_connected_subnet as sample_connected_subnet_std,
    validate_airport_map, airport_map_hash,
)
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
    "turkish": get_turkish_constraints_hb,  # HB1/HB2 cross-base return contract
}
from model import FlightEncoder, PointerDecoder
from evaluation.set_partition import solve_set_covering, solve_lp_relaxation, column_reduction
from evaluation.dual_feedback import (
    solve_full_universe_lp, build_dual_signal, merge_unique_columns,
)
from evaluation.completion_runner import merge_rescue_columns, solve_completion_stages
from completion.rescue_generator import generate_rescue_candidates
from evaluation.completion_report import build_completion_report, render_completion_table, save_completion_report
from evaluation.validator import validate_pairing
from evaluation.validation_report import aggregate_by_source_per_chunk
from utils import constraint_to_tensor, flights_to_tensors, set_skip_decoder_constraint
from rollout import rollout_with_pairings, rollout_batch, set_environment
from base_reach import build_base_reaches
import config


# ── 0. Turkish window loader ─────────────────────────────────────────────────

def validate_window_days(window_days, constraint, airline):
    min_window_days = math.ceil(float(constraint["max_pairing_days"]))
    if window_days < min_window_days:
        raise ValueError(
            f"window_days={window_days}는 {airline} max_pairing_days="
            f"{constraint['max_pairing_days']}보다 짧습니다. "
            f"window_days를 최소 {min_window_days}로 설정해야 합니다."
        )
    return window_days


def _attach_window_lookahead(core_windows, window_days, lookahead_days):
    """고유 core flight에 ID를 한 번 부여하고 이후 날짜는 연결 context로만 겹침."""
    global_offset = 0
    for window in core_windows:
        for flight in window:
            flight["global_id"] = global_offset
            global_offset += 1

    expanded = []
    for core_index, core in enumerate(core_windows):
        core_start = core_index * window_days * 24.0
        context_end = core_start + (window_days + lookahead_days) * 24.0
        combined = []
        for source_index in range(core_index, len(core_windows)):
            source_start = source_index * window_days * 24.0
            if source_start >= context_end:
                break
            for flight in core_windows[source_index]:
                absolute_dep = flight["dep_time"] + source_start
                absolute_arr = flight["arr_time"] + source_start
                if absolute_dep >= context_end:
                    continue
                combined.append({
                    **flight,
                    "dep_time": absolute_dep - core_start,
                    "arr_time": absolute_arr - core_start,
                    "_absolute_dep_time": absolute_dep,
                    "_absolute_arr_time": absolute_arr,
                    "_is_core": source_index == core_index,
                })
        expanded.append(sorted(combined, key=lambda flight: flight["dep_time"]))
    return expanded, global_offset


def load_windows_turkish(turkish_df, airport_map, window_days=5, lookahead_days=0):
    """Turkish flight를 고유 core와 pairing 연결용 lookahead context로 분할함."""
    dates = sorted(turkish_df["dep_date_utc"].unique())
    n_days = len(dates)
    core_windows = []

    for offset in range(0, n_days, window_days):
        wf = load_flights_rolling_turkish(
            window_days=window_days,
            offset_days=offset,
            airport_map=airport_map,
            df=turkish_df,
        )
        core_windows.append(wf)
        print(
            f"  window offset={offset:2d}: {len(wf):5d}legs",
            flush=True,
        )

    return _attach_window_lookahead(core_windows, window_days, lookahead_days)


# ── 1. Load the full dataset window by window, assigning global IDs ─────────

def load_windows_with_global_ids(data_path, airport_map, window_days=5, lookahead_days=0):
    """CSV를 고유 core와 pairing 연결용 lookahead context로 분할함.

    Returns:
        windows    : list of flight lists. Each flight gets a 'global_id' field.
        n_total    : total number of flights (= upper bound on global IDs)
    """
    df = pd.read_csv(data_path)
    df = df[["ORIGIN", "DEST", "CRS_DEP_TIME", "CRS_ARR_TIME", "CRS_ELAPSED_TIME", "FL_DATE"]].dropna()
    df["FL_DATE"] = pd.to_datetime(df["FL_DATE"], format="mixed")

    dates = sorted(df["FL_DATE"].unique())
    n_days = len(dates)

    core_windows = []

    for offset in range(0, n_days, window_days):
        wf = load_flights_rolling(
            data_path,
            window_days=window_days,
            offset_days=offset,
            airport_map=airport_map,
            n_max=None,
            df=df,
        )
        core_windows.append(wf)
        print(
            f"  window offset={offset:2d}: {len(wf):5d}legs",
            flush=True,
        )

    return _attach_window_lookahead(core_windows, window_days, lookahead_days)


# ── 2. Subset sampling within a window ───────────────────────────────────────

def sample_connected_subset(window_flights, subset_size, base_id, constraint):
    """Connectivity-aware subset sampling with random coverage guarantee.

    Starts BFS from base-departing legs to preferentially select connectable
    legs, but only fills BFS_RATIO of the subset via BFS; the rest is chosen
    pure-random from the whole window.

    Rationale: if BFS alone quickly fills the hub-and-spoke-dense region,
    isolated flights with no connections are never included in any rollout,
    capping pool coverage at ~85%. Guaranteeing 15% random inclusion gives
    each flight an expected ~5 inclusions over 300 rollouts, keeping the
    omission probability under 1%.

    Args:
        window_flights: list of all flights in the window
        subset_size:    number of legs to select (config.EPISODE_MAX_FLIGHTS)
        base_id:        crew base airport integer ID
        constraint:     constraint dict (min_conn, max_conn in hours)
    """
    BFS_RATIO = 0.85  # 85% of the subset is BFS-connected (preserves RL multi-leg density)

    min_conn = constraint.get("min_conn", 0.65)   # hours
    max_conn = constraint.get("max_conn", 9.0)     # hours

    # Index by origin airport
    by_origin = {}
    for f in window_flights:
        by_origin.setdefault(f["origin"], []).append(f)

    selected_ids = set()
    selected = []

    # BFS phase: fill only up to BFS_RATIO of the subset
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

        # Add legs connectable after this arrival to the queue
        nexts = [
            g for g in by_origin.get(f["dest"], [])
            if g["id"] not in selected_ids
            and min_conn <= g["dep_time"] - f["arr_time"] <= max_conn
        ]
        random.shuffle(nexts)
        queue.extend(nexts)

    # Random phase: fill remaining slots pure-random from the whole window
    # -> guarantees isolated flights get included with some probability every rollout
    remaining = [f for f in window_flights if f["id"] not in selected_ids]
    random.shuffle(remaining)
    for f in remaining[:subset_size - len(selected)]:
        selected_ids.add(f["id"])
        selected.append(f)

    selected = sorted(selected, key=lambda f: f["dep_time"])
    for local_id, f in enumerate(selected):
        f["local_id"] = local_id

    return selected


# ── 3. Subset rollout -> global_id-keyed pairings ────────────────────────────

def constraint_for_pairing_base(pairing, constraint):
    """회전 후 pairing의 실제 시작 base에 맞춘 검증 constraint를 반환함."""
    pairing_base = pairing.get("true_start_airport", constraint["base_airport"])
    allowed_bases = set(constraint.get("base_ids") or [constraint["base_airport"]])
    if pairing_base not in allowed_bases:
        return None
    return {**constraint, "base_airport": pairing_base}


def rollout_subset_global(subset, constraint, encoder, decoder, max_time, greedy=False):
    """Rollout 결과를 독립 검증한 뒤 global ID column으로 변환함."""
    local_flights = [{**f, "id": f["local_id"]} for f in subset]
    origins, dests, dep_norm, arr_norm, fly_norm = flights_to_tensors(
        local_flights, max_time, device=DEVICE
    )
    c_tensor = constraint_to_tensor(constraint, device=DEVICE)
    id_map = {f["local_id"]: f["global_id"] for f in subset}

    with torch.no_grad():
        encoded = encoder(origins, dests, dep_norm, arr_norm, fly_norm, c_tensor)
        raw_pairings = rollout_with_pairings(
            local_flights, constraint, encoder, decoder, encoded,
            greedy=greedy, device=DEVICE,
        )

    local_by_id = {f["id"]: f for f in local_flights}
    validated_pairings = []
    for pairing in raw_pairings:
        pairing_constraint = constraint_for_pairing_base(pairing, constraint)
        if pairing_constraint is None:
            continue
        validation = validate_pairing(pairing, local_by_id, pairing_constraint)
        if not validation["is_valid"]:
            continue
        pairing["is_legal"] = True
        pairing["validator_version"] = validation["validator_version"]
        pairing["constraint_hash"] = validation["constraint_hash"]
        pairing["legs"] = [id_map[leg] for leg in pairing["legs"]]
        validated_pairings.append(pairing)
    return validated_pairings


def rollout_subset_global_batch(
    subset, constraint, encoder, decoder, max_time, B, greedy=False,
    dual_by_global_id=None, dual_weight=1.0,
):
    """rollout_subset_global()의 배치 버전 (Phase 5b, experiment/
    rollout-batch-vectorization) -- encoder를 한 번만 호출하고(예전엔 호출할 때마다
    같은 chunk에 대해 매번 새로 인코딩하던 비효율이 있었음) rollout_batch()로 B개
    episode를 한 번에 처리한 뒤, 각각 독립적으로 검증 + global ID 변환.

    반환: List[List[pairing]] -- episode별 검증된 pairing 리스트 (rollout_subset_global()을
    B번 호출한 것과 동일한 형식).
    """
    local_flights = [{**f, "id": f["local_id"]} for f in subset]
    origins, dests, dep_norm, arr_norm, fly_norm = flights_to_tensors(
        local_flights, max_time, device=DEVICE
    )
    c_tensor = constraint_to_tensor(constraint, device=DEVICE)
    id_map = {f["local_id"]: f["global_id"] for f in subset}
    local_dual = None
    if dual_by_global_id is not None:
        local_dual = {
            local_id: float(dual_by_global_id.get(global_id, 0.0))
            for local_id, global_id in id_map.items()
        }

    with torch.no_grad():
        encoded = encoder(origins, dests, dep_norm, arr_norm, fly_norm, c_tensor)
        episodes_raw = rollout_batch(
            local_flights, constraint, encoder, decoder, encoded,
            B=B, greedy=greedy, device=DEVICE,
            flight_action_scores=local_dual, dual_weight=dual_weight,
        )

    local_by_id = {f["id"]: f for f in local_flights}

    results = []
    for raw_pairings in episodes_raw:
        validated_pairings = []
        for pairing in raw_pairings:
            pairing_constraint = constraint_for_pairing_base(pairing, constraint)
            if pairing_constraint is None:
                continue
            validation = validate_pairing(pairing, local_by_id, pairing_constraint)
            if not validation["is_valid"]:
                continue
            pairing["is_legal"] = True
            pairing["validator_version"] = validation["validator_version"]
            pairing["constraint_hash"] = validation["constraint_hash"]
            pairing["legs"] = [id_map[leg] for leg in pairing["legs"]]
            validated_pairings.append(pairing)
        results.append(validated_pairings)
    return results


# ── 3-1. Partition a full window into connectivity-preserving chunks ────────

def partition_connected_chunks(window_flights, base_ids, chunk_size, connected_sampler):
    """Partition the whole window into connected-subnet chunks (Sec. "Scalable
    Inference and Global Selection": divide each window into
    connectivity-preserving chunks containing at most n_max flight legs).

    Builds each chunk with the same sample_connected_subnet logic used during
    training (RL/loader.py, RL/turkish/loader_turkish.py), repeating until
    `remaining` is empty so every flight belongs to exactly one chunk
    (100% coverage), keeping the same connectivity-density distribution seen at training time.
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


# ── 4. Collect the pool across all windows ───────────────────────────────────

def collect_pool_full(windows, base_ids, constraint, encoder, decoder,
                      n_rollouts_per_chunk=15,
                      subset_size=config.EPISODE_MAX_FLIGHTS,
                      window_days=5,
                      model_max_time=None,
                      dual_by_global_id=None, dual_weight=1.0,
                      connected_sampler=sample_connected_subnet_std,
                      airline="delta"):
    """Roll out over all windows to build the global-ID-keyed candidate pool Cθ.

    Paper Sec. "Scalable Inference and Global Selection": "For each chunk, we
    perform multiple stochastic rollouts and one greedy rollout... Candidates
    from all chunks are merged into a global pool Cθ." Each window is split
    into connectivity-preserving chunks via connected_sampler (the same
    sample_connected_subnet used during training); each chunk gets
    n_rollouts_per_chunk stochastic rollouts plus 1 greedy rollout.
    Partitioning repeats until `remaining` is empty, so every flight is
    included in at least one rollout (guaranteeing 100% coverage
    opportunity) while preserving the same connectivity density seen during training.

    Turkish는 선택된 HB1/HB2 중 하나에서 시작하고 두 home base 중 어느 쪽으로든 복귀 가능함.
    일반 항공사는 pairing이 출발한 동일 base로 복귀함.
    """
    pool = {}
    covered_global = set()
    # 데이터 분할 길이와 encoder 시간 정규화 기준을 분리함.
    max_time = float(model_max_time if model_max_time is not None else window_days * 24.0)
    if max_time <= 0:
        raise ValueError("model_max_time은 양수여야 합니다.")
    base_id_set = set(base_ids)

    for w_idx, window_flights in enumerate(windows):
        if not window_flights:
            continue

        window_all_ids = {
            f["global_id"] for f in window_flights if f.get("_is_core", True)
        }
        window_covered = set()

        chunks = partition_connected_chunks(window_flights, base_ids, subset_size, connected_sampler)

        # Guarantee a base-departing leg in each chunk: if missing, inject a
        # copy of the nearest base-departing leg from the window
        # (connected-subnet partitioning usually includes one, but tail
        # chunks etc. can be an exception).
        for c_idx, chunk in enumerate(chunks):
            if not any(f["origin"] in base_id_set for f in chunk):
                chunk_gids = {f["global_id"] for f in chunk}
                candidates = [f for f in window_flights
                              if f["origin"] in base_id_set and f["global_id"] not in chunk_gids]
                if candidates:
                    inject = min(candidates,
                                 key=lambda f: abs(f["dep_time"] - chunk[0]["dep_time"]))
                    # base flight를 보조 context로 추가하되 기존 flight를 제거하지 않음.
                    # 크기 제한보다 전체 flight universe 보존이 우선임.
                    new_chunk = sorted([{**inject}] + list(chunk),
                                       key=lambda f: f["dep_time"])
                    chunks[c_idx] = new_chunk

        print(f"\n[Window {w_idx + 1}/{len(windows)}] {len(window_flights)} legs -> {len(chunks)} chunks", flush=True)

        rollout_count = 0
        for c_idx, chunk in enumerate(chunks):
            for local_id, f in enumerate(chunk):
                f["local_id"] = local_id
            chunk_by_gid = {f["global_id"]: f for f in chunk}

            def _pairing_valid(p, _chunk_by_gid=chunk_by_gid):
                if airline != "turkish":
                    return p["ends_at_base"]
                # Turkish는 HB1→HB2와 HB2→HB1 교차 home-base 복귀도 유효함.
                first = _chunk_by_gid.get(p["legs"][0])
                last = _chunk_by_gid.get(p["legs"][-1])
                return (
                    first is not None and last is not None
                    and first["origin"] in base_id_set
                    and last["dest"] in base_id_set
                )

            base_id = random.choice(base_ids)
            c_b = {**constraint, "base_airport": base_id}
            c_b["base_ids"] = base_ids
            # local ID 기준 reachability를 모든 CPP rollout에 필수로 구성함.
            _local_flights = [{**f, "id": f["local_id"]} for f in chunk]
            return_bases = base_ids if c_b.get("allow_cross_base_return") else [base_id]
            c_b["_base_reaches"] = build_base_reaches(_local_flights, return_bases, c_b)
            c_b["_base_reach"] = c_b["_base_reaches"][base_id]

            # Phase 5b(experiment/rollout-batch-vectorization): n_rollouts_per_chunk번
            # 순차 호출 대신 rollout_batch()로 한 번에 배치 처리 -- encoder도 한 번만
            # 호출됨(예전엔 호출마다 같은 chunk를 매번 새로 인코딩하던 비효율이 있었음).
            # 개별 episode의 예외는 rollout_batch() 안에서 이미 격리되므로(한 episode
            # 실패가 나머지를 안 죽임), 여기 바깥 try/except는 encoder/flights_to_tensors
            # 등 배치 전체에 영향을 주는 실패에 대한 안전망으로만 남겨둠.
            try:
                stochastic_results = rollout_subset_global_batch(
                    chunk, c_b, encoder, decoder, max_time, B=n_rollouts_per_chunk, greedy=False,
                    dual_by_global_id=dual_by_global_id, dual_weight=dual_weight,
                )
            except Exception as e:
                raise RuntimeError(
                    f"stochastic rollout batch failed (window={w_idx}, chunk={c_idx})"
                ) from e

            for pairings in stochastic_results:
                for p in pairings:
                    # Exclude pairings that don't return to base from both the
                    # pool and coverage counts -- including them in coverage
                    # would create "phantom" coverage that the IP can never
                    # actually select, so window_covered/covered_global must
                    # be filtered the same way.
                    if not _pairing_valid(p):
                        continue
                    # C3: 이 pairing이 실제로 어느 base_airport로 생성됐는지 남겨둠 --
                    # 최종 selected pairing을 독립 재검증할 때(evaluate_full()) 그
                    # pairing이 실제로 생성될 때 쓰인 constraint를 복원하는 데 필요함
                    # (chunk마다 base_id가 랜덤으로 다시 뽑히므로).
                    p["_gen_base_airport"] = p.get("true_start_airport", base_id)
                    key = tuple(sorted(p["legs"]))
                    if key not in pool or p["cost"] < pool[key]["cost"]:
                        pool[key] = p
                    window_covered.update(set(p["legs"]) & window_all_ids)
                    covered_global.update(p["legs"])
                rollout_count += 1

            try:
                pairings = rollout_subset_global_batch(
                    chunk, c_b, encoder, decoder, max_time, B=1, greedy=True,
                    dual_by_global_id=dual_by_global_id, dual_weight=dual_weight,
                )[0]
            except Exception as e:
                raise RuntimeError(
                    f"greedy rollout failed (window={w_idx}, chunk={c_idx})"
                ) from e
            for p in pairings:
                if not _pairing_valid(p):
                    continue
                p["_gen_base_airport"] = p.get("true_start_airport", base_id)
                key = tuple(sorted(p["legs"]))
                if key not in pool or p["cost"] < pool[key]["cost"]:
                    pool[key] = p
                window_covered.update(set(p["legs"]) & window_all_ids)
                covered_global.update(p["legs"])
            rollout_count += 1

            print(
                f"    chunk {c_idx + 1}/{len(chunks)} done "
                f"(cumulative rollouts={rollout_count}, pool={len(pool)}, "
                f"window covered={len(window_covered)}/{len(window_all_ids)})",
                flush=True,
            )

        print(
            f"  {rollout_count} total rollouts: "
            f"window covered {len(window_covered)}/{len(window_all_ids)} legs, "
            f"pool={len(pool)}",
            flush=True,
        )

        uncov = len(window_all_ids - window_covered)
        if uncov > 0:
            print(f"  uncovered: {uncov} legs (reported as uncoverable by the IP)", flush=True)

    total_flights = len({f["global_id"] for w in windows for f in w})
    print(f"\ntotal pool: {len(pool)} pairings")
    print(f"total coverage: {len(covered_global)}/{total_flights} legs")
    final_pool = list(pool.values())
    for index, pairing in enumerate(final_pool):
        pairing.setdefault("source_type", "policy")
        pairing.setdefault("is_legal", True)
        pairing["column_id"] = f"{pairing['source_type']}-{index}"
    return final_pool, covered_global



def validate_rescue_columns_current_run(rescue_columns, flights_by_id, constraint, base_ids):
    """외부 rescue provenance를 신뢰하지 않고 현재 flight와 규정으로 재검증함."""
    validated = []
    for index, raw in enumerate(rescue_columns or []):
        rescue = dict(raw)
        legs = list(rescue.get("legs", []))
        if not legs or legs[0] not in flights_by_id:
            raise ValueError(f"rescue-{index}: 현재 instance에서 시작 flight를 찾을 수 없습니다.")
        start_base = flights_by_id[legs[0]]["origin"]
        if start_base not in set(base_ids):
            raise ValueError(
                f"rescue-{index}: 시작 공항 {start_base}은 configured crew base가 아닙니다."
            )
        current_constraint = {**constraint, "base_airport": start_base}
        result = validate_pairing(rescue, flights_by_id, current_constraint)
        if not result["is_valid"]:
            raise ValueError(
                f"rescue-{index}: current-run validator 위반 {result['violation_codes']}"
            )
        if rescue.get("validator_version") != result["validator_version"]:
            raise ValueError(f"rescue-{index}: validator_version이 현재 실행과 다릅니다.")
        if rescue.get("constraint_hash") != result["constraint_hash"]:
            raise ValueError(f"rescue-{index}: constraint_hash가 현재 실행과 다릅니다.")
        rescue["is_legal"] = True
        rescue["_gen_base_airport"] = start_base
        validated.append(rescue)
    return validated


def solve_pool_completion(
    pool, n_total, *, lambda_excess=1.0, time_limit=300,
    threads=1, use_gurobi=False,
    reposition_penalty=None, reserve_penalty=None,
    artificial_penalty=None, report_path=None, rescue_columns=None,
    auto_operational=False, verbose=False,
):
    """수집된 pool을 V2 단계별 master로 풀고 legacy 출력 호환 필드를 추가함."""
    if rescue_columns:
        pool = merge_rescue_columns(pool, rescue_columns, range(n_total))
    stages = solve_completion_stages(
        pool, range(n_total), lambda_excess=lambda_excess,
        time_limit=time_limit, reposition_penalty=reposition_penalty,
        threads=threads, use_gurobi=use_gurobi,
        reserve_penalty=reserve_penalty, artificial_penalty=artificial_penalty,
        auto_operational=auto_operational,
        verbose=verbose,
    )
    report = build_completion_report(stages, range(n_total))
    if report_path:
        save_completion_report(report, report_path)
    result = dict(stages[-1])
    result["completion_report"] = report
    result["uncoverable"] = len(result["uncovered_flight_ids"])
    result["deadhead_count"] = result["excess_count"]
    result["deadhead_flights"] = result["excess_flight_ids"]
    result["mip_obj"] = result["mip_objective"]
    result["total_cost"] = result["pairing_cost"]
    result["n_pairings"] = len(result["selected"])
    return result
# ── 4-1. C3: 최종 selected pairing 독립 재검증 ────────────────────────────────

def validate_selected_pairings(selected, flights_by_id, constraint, base_ids,
                                n_total_flights, strict=False):
    """C3 -- solve_set_covering()이 고른 최종 pairing을 생성 시 hard mask와
    무관하게 evaluation/validator.py로 독립 재검증하고, source_type별로 분리 집계한다.

    각 pairing은 collect_pool_full()에서 남겨둔 `_gen_base_airport`(그 pairing이
    생성될 때 실제로 쓰인 base) 기준으로 자기 constraint를 복원해서 검증한다 --
    chunk마다 base_id가 랜덤으로 다시 뽑히므로, 전부 하나의 constraint로 검증하면
    다른 base에서 생성된 pairing이 잘못 invalid로 잡힐 수 있다

    strict=True면 invalid가 하나라도 있을 때 RuntimeError를 던져서, 이 결과를
    legal한 solution으로 잘못 보고하지 않게 한다.
    """
    base_template = {**constraint, "base_ids": base_ids}

    by_base = {}
    for p in selected:
        b = p.get("_gen_base_airport", base_ids[0])
        by_base.setdefault(b, []).append(p)
    chunks = [
        (pairings, {**base_template, "base_airport": b})
        for b, pairings in by_base.items()
    ]
    report = aggregate_by_source_per_chunk(chunks, flights_by_id, n_total_flights=n_total_flights)

    # aggregate_by_source_per_chunk()가 bucket별로 이미 validate_pairing()을 돌려서
    # invalid_pairings(violation_codes 포함)를 남겨두므로, 여기서 또 재검증하지 않고
    # 그 결과를 그대로 모아 쓴다(bucket 딕셔너리만 골라내고, cross_bucket_duplicate_
    # flight_ids/_direct_coverage_source/validator_version 같은 report 레벨 메타
    # 항목은 건너뜀).
    invalid = [
        entry
        for bucket in report.values()
        if isinstance(bucket, dict) and bucket.get("invalid_pairings")
        for entry in bucket["invalid_pairings"]
    ]

    report["n_invalid_selected"] = len(invalid)
    report["invalid_selected"] = invalid

    if strict and invalid:
        raise RuntimeError(
            f"[strict-validation] 선택된 pairing {len(invalid)}개가 independent "
            f"validator 기준으로 invalid입니다 (예: {invalid[0]})."
        )
    return report


# ── 4-2. C3/C4: 최종 selected pairing을 flight ID까지 포함해서 JSON으로 저장 ───

def default_save_json_path(checkpoint_path, airline, eval_mode):
    """`--save-json`/`--no-save-json` 둘 다 안 줬을 때 자동으로 쓰는 저장 경로.
    checkpoint 파일명 + airline + eval_mode로 구성해서, 같은 checkpoint를
    strict/legacy 두 모드로 각각 평가해도 서로 덮어쓰지 않게 한다.
    """
    ckpt_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
    return f"log/eval_json/{eval_mode}_{airline}_{ckpt_name}.json"


def save_result_json(path, result, checkpoint_path, airline, eval_mode):
    """v1.md C3 "ASCP 결과 JSON/CSV에 validator version과 constraint hash 기록".

    지금까지는 이 저장 코드 자체가 없어서 evaluate_ip.py가 화면에 찍는 요약
    통계(coverage/ManDays/FTC 등)만 log 파일로 남았고, 실제 선택된 pairing이
    어떤 flight(legs)로 구성됐는지는 어디에도 저장되지 않았음 -- log/ 전체를 확인해서
    검증함). 이 함수가 그 공백을 메움-- 이후로 이 형식으로 저장된 파일은
    evaluation/ascp_output_adapter.py로 다시 읽어서 재검증 가능
    """
    payload = {
        "checkpoint": checkpoint_path,
        "airline": airline,
        "eval_mode": eval_mode,
        "n_pairings": result["n_pairings"],
        "coverage": result["coverage"],
        "uncoverable": result["uncoverable"],
        "deadhead_count": result["deadhead_count"],
        "mip_obj": result.get("mip_obj"),
        "status": result["status"],
        "validation_report": result["validation_report"],
        "completion_report": result.get("completion_report"),
        "pairings": [
            {
                "legs": p["legs"],
                "source_type": p.get("source_type", "policy"),
                "duty_break_indices": p.get("duty_break_indices"),
                "_gen_base_airport": p.get("_gen_base_airport"),
            }
            for p in result["selected"]
        ],
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)


# ── 5. Main evaluation function ──────────────────────────────────────────────

def _resolve_window_days(airline, requested_window_days):
    """명시값이 없으면 항공사별 학습 window 설정을 그대로 사용함."""
    return (
        config.AIRLINE_WINDOW_DAYS[airline]
        if requested_window_days is None
        else requested_window_days
    )


def evaluate_full(
    checkpoint_path,
    airline="delta",
    data_path=None,
    n_rollouts_per_chunk=15,
    window_days=None,
    subset_size=config.EPISODE_MAX_FLIGHTS,
    bases=None,
    ip_time_limit=3600,
    ip_threads=1,
    ip_solver="cbc",
    lambda_dh=10.0,
    device="cpu",
    turkish_files=None,
    use_wandb=False,
    wandb_project="ASCP-2026-paper",
    compute_gap=False,
    full_flight_master=False,
    completion_report_path=None,
    rescue_pool_path=None,
    auto_rescue=True,
    auto_operational=False,
    reposition_penalty=None,
    reserve_penalty=None,
    artificial_penalty=None,
    seed=None,
    strict_validation=False,
    save_json_path=None,
    no_save_json=False,
    dual_iterations=0, dual_weight=1.0, dual_mode="real",
    dual_artificial_penalty=None, dual_trace_path=None,
):
    """Full flight-coverage evaluation. Uses config.AIRLINE_DATA[airline] if data_path is unset.

    For small-scale data (e.g. a one-week sample):
        data_path=<sample.csv>, window_days=1, n_rollouts_per_chunk=3

    If use_wandb=True, logs the eval config + console output + final result
    metrics to wandb (job_type="eval" -- kept as a separate run from training curves).

    If seed is given, fixes the random/torch RNG so different checkpoints are
    evaluated on the same window partitioning and the same rollout sampling,
    enabling paired comparison (differences between checkpoints then come
    only from policy differences, not evaluation randomness).
    """
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    global DEVICE
    DEVICE = torch.device(device)
    if device == "cpu":
        # By default torch claims as many threads per process as there are
        # physical cores, which causes heavy CPU contention when evaluating
        # multiple checkpoints in parallel (and with concurrent GPU training
        # processes). Some torch builds ignore env-level limits like
        # OMP_NUM_THREADS, so pin it explicitly here as well.
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 4)))
    set_environment(airline)
    window_days = _resolve_window_days(airline, window_days)

    wandb_run = None
    # C3 "기존 checkpoint 평가와 신규 strict 평가를 서로 다른 mode 이름으로 저장" --
    # wandb run 이름/config에도 반영하고(아래), 실제 JSON 파일 저장(save_result_json)
    # 경로에도 이 이름을 씀. eval_mode 자체는 --strict-validation과 무관하게(그
    # 값과 상관없이) 항상 명시해서, 새 파이프라인(독립 validator가 붙은 버전)으로
    # 평가됐다는 걸 구분할 수 있게 한다.
    eval_mode = "strict" if strict_validation else "legacy"
    if full_flight_master:
        eval_mode = f"full_{eval_mode}"

    # 명시적으로 --no-save-json을 주지 않는 한 항상 저장하고, 경로만
    # 기본값(checkpoint/airline/eval_mode 기반)을 자동 생성한다.
    if save_json_path is None and not no_save_json:
        save_json_path = default_save_json_path(checkpoint_path, airline, eval_mode)
    if use_wandb:
        import wandb
        wandb_run = wandb.init(
            project=wandb_project,
            job_type="eval",
            name=(
                f"eval-{eval_mode}-{airline}-"
                f"{'one-shot' if dual_iterations == 0 else dual_mode}-"
                f"{os.path.basename(checkpoint_path)}"
            ),
            config=dict(
                checkpoint=checkpoint_path, airline=airline,
                subset_size=subset_size, window_days=window_days,
                n_rollouts_per_chunk=n_rollouts_per_chunk,
                ip_time_limit=ip_time_limit, lambda_dh=lambda_dh,
                ip_threads=ip_threads,
                ip_solver=ip_solver,
                time_basis="turkish_native" if airline == "turkish" else "utc",
                eval_mode=eval_mode,
                strict_validation=strict_validation,
                dual_iterations=dual_iterations,
                dual_weight=dual_weight,
                dual_mode=dual_mode,
                dual_artificial_penalty=dual_artificial_penalty,
            ),
        )

    if data_path is None:
        data_path = config.AIRLINE_DATA[airline]
    if bases is None:
        bases = config.AIRLINE_BASES[airline]

    # Load the checkpoint first to check its vocab size -- a multi-airline
    # model (n_airports=168) needs the merged airport map; building it from a
    # single-airline map would cause an embedding-index mismatch.
    ckpt       = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    n_airports = ckpt.get("n_airports",
                          ckpt["encoder"]["airport_emb.weight"].shape[0])
    expected_time_basis = "turkish_native" if airline == "turkish" else "utc"
    checkpoint_time_basis = ckpt.get("time_basis")
    if checkpoint_time_basis != expected_time_basis:
        raise ValueError(
            "체크포인트 시간 기준이 평가 데이터와 일치하지 않음: "
            f"checkpoint={checkpoint_time_basis or '미기록(레거시)'}, "
            f"required={expected_time_basis}. BTS UTC 전환 전 체크포인트는 재학습해야 함."
        )
    checkpoint_max_time = ckpt.get("max_time")
    if checkpoint_max_time is None or float(checkpoint_max_time) <= 0:
        raise ValueError(
            "checkpoint에 유효한 max_time이 없음. 학습과 평가의 시간 정규화를 "
            "일치시키기 위해 새 schema checkpoint가 필요함."
        )
    checkpoint_max_time = float(checkpoint_max_time)

    checkpoint_airport_map = ckpt.get("airport_map")
    if checkpoint_airport_map is None:
        raise ValueError(
            "checkpoint에 airport_map이 없음. 데이터별 ID 재생성을 막기 위해 "
            "새 schema로 재학습해야 함."
        )
    airport_map = validate_airport_map(checkpoint_airport_map, n_airports)
    stored_map_hash = ckpt.get("airport_map_hash")
    if stored_map_hash and stored_map_hash != airport_map_hash(airport_map):
        raise ValueError("checkpoint airport_map hash가 저장 내용과 일치하지 않음")

    checkpoint_airline = ckpt.get("airline")
    checkpoint_airlines = ckpt.get("airlines", [])
    if checkpoint_airline != "multi" and checkpoint_airline != airline:
        raise ValueError(
            f"단일 항공사 checkpoint({checkpoint_airline})를 {airline} 데이터로 평가할 수 없음"
        )
    if checkpoint_airline == "multi" and airline not in checkpoint_airlines:
        raise ValueError(
            f"multi checkpoint 학습 항공사 {checkpoint_airlines}에 {airline}이 포함되지 않음"
        )

    _turkish_df = None
    if airline == "turkish":
        # If turkish_files is unset, default to the Zeren Feb benchmark
        # window (15,742 legs, 0.03% off the target 15,738)
        if turkish_files is None:
            _turkish_df = parse_legs_dir(data_path, files=[ZEREN_FEB_FILE], date_range=ZEREN_FEB_WINDOW)
        else:
            _turkish_df = parse_legs_dir(data_path, files=turkish_files)
        unknown = (
            set(_turkish_df["ORIGIN"]) | set(_turkish_df["DEST"])
        ) - set(airport_map)
    else:
        _airport_df = pd.read_csv(data_path, usecols=["ORIGIN", "DEST"]).dropna()
        unknown = (set(_airport_df["ORIGIN"]) | set(_airport_df["DEST"])) - set(airport_map)
    if unknown:
        raise ValueError(
            "평가 데이터에 checkpoint 학습 당시 없던 공항이 있음. "
            f"실험 universe를 먼저 정의해야 함: {sorted(unknown)}"
        )
    base_ids = bases_to_ids(list(bases), airport_map)

    skip_film = bool(ckpt.get("skip_film", False))
    skip_decoder_constraint = bool(ckpt.get("skip_decoder_constraint", False))
    set_skip_decoder_constraint(skip_decoder_constraint)
    encoder = FlightEncoder(
        n_airports=n_airports,
        constraint_dim=len(FILM_CONSTRAINT_KEYS),
        use_film_before=not skip_film,
        use_film_after=not skip_film,
    ).to(DEVICE)
    # Auto-detect the checkpoint's state_vec dimension (older checkpoints used
    # fewer scalars than the current state_to_vec)
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

    validate_window_days(window_days, constraint, airline)

    print(f"\nLoading full dataset ({airline}, window_days={window_days})...", flush=True)
    if airline == "turkish":
        windows, n_total = load_windows_turkish(
            _turkish_df, airport_map, window_days,
            lookahead_days=constraint["max_pairing_days"],
        )
    else:
        windows, n_total = load_windows_with_global_ids(
            data_path, airport_map, window_days,
            lookahead_days=constraint["max_pairing_days"],
        )
    print(f"total {n_total} legs, {len(windows)} windows", flush=True)
    # C3: global_id -> flight dict, independent validator가 pairing legs를 조회하는 데 씀
    flights_by_id = {
        f["global_id"]: {
            **f,
            "id": f["global_id"],
            "dep_time": f.get("_absolute_dep_time", f["dep_time"]),
            "arr_time": f.get("_absolute_arr_time", f["arr_time"]),
        }
        for w in windows for f in w
        if f.get("_is_core", True)
    }

    connected_sampler = sample_connected_subnet_turkish if airline == "turkish" else sample_connected_subnet_std

    print("\n[base-return] CPP hard constraint ON (includes reachability pruning)", flush=True)
    if airline == "turkish":
        print("  [note] Turkish는 HB1/HB2 교차 home-base 복귀를 허용합니다.", flush=True)

    print(f"\nCollecting pool (rollouts/chunk={n_rollouts_per_chunk}, subset={subset_size})...", flush=True)
    with torch.no_grad():
        pool, covered = collect_pool_full(
            windows, base_ids, constraint, encoder, decoder,
            n_rollouts_per_chunk=n_rollouts_per_chunk,
            subset_size=subset_size,
            window_days=window_days,
            model_max_time=checkpoint_max_time,
            connected_sampler=connected_sampler,
            airline=airline,
        )

    dual_trace = []
    for dual_iteration in range(1, dual_iterations + 1):
        if dual_artificial_penalty is None:
            # 절대값 대신 그 시점 pool의 cost 규모(Cmax)에 상대적인 값을 씀
            _legal_costs = [float(c["cost"]) for c in pool
                            if math.isfinite(float(c.get("cost", 0.0))) and float(c.get("cost", 0.0)) >= 0]
            _cmax = max([1.0] + _legal_costs)
            _artificial_penalty = _cmax * 2.0
        else:
            _artificial_penalty = dual_artificial_penalty
        lp_feedback = solve_full_universe_lp(
            pool, range(n_total), lambda_excess=lambda_dh,
            artificial_penalty=_artificial_penalty,
        )
        signal = build_dual_signal(lp_feedback, dual_mode)

        with torch.no_grad():
            generated_pool, _ = collect_pool_full(
                windows, base_ids, constraint, encoder, decoder,
                n_rollouts_per_chunk=n_rollouts_per_chunk,
                subset_size=subset_size, window_days=window_days,
                model_max_time=checkpoint_max_time,
                dual_by_global_id=signal, dual_weight=dual_weight,
                connected_sampler=connected_sampler, airline=airline,
            )
        before = len(pool)
        pool = merge_unique_columns(pool, generated_pool)
        for index, pairing in enumerate(pool):
            pairing["column_id"] = f"{pairing.get('source_type', 'policy')}-{index}"
        dual_trace.append({
            "iteration": dual_iteration,
            "lp_objective": lp_feedback["lp_objective"],
            "artificial_count": lp_feedback["artificial_count"],
            "zero_cost_count": lp_feedback["zero_cost_count"],
            "zero_cost_fraction": lp_feedback["zero_cost_fraction"],
            "pool_size_before": before, "generated_count": len(generated_pool),
            "pool_size_after": len(pool), "new_unique_count": len(pool) - before,
            "dual_mode": dual_mode, "dual_weight": dual_weight,
            "dual_artificial_penalty_used": _artificial_penalty,
        })
        if len(pool) == before:
            break

    if dual_trace_path:
        os.makedirs(os.path.dirname(dual_trace_path) or ".", exist_ok=True)
        with open(dual_trace_path, "w", encoding="utf-8") as handle:
            json.dump(dual_trace, handle, indent=2)

    if not full_flight_master:
        print(f"\nColumn reduction (pool={len(pool)})...", flush=True)
        if artificial_penalty is None:
            _legal_costs = [float(p["cost"]) for p in pool
                            if math.isfinite(float(p.get("cost", 0.0))) and float(p.get("cost", 0.0)) >= 0]
            _reduction_artificial_penalty = max([1.0] + _legal_costs) * 2.0
        else:
            _reduction_artificial_penalty = artificial_penalty
        _lp_for_reduction = solve_lp_relaxation(
            pool, lambda_dh=lambda_dh, flight_ids=range(n_total),
            artificial_cost=_reduction_artificial_penalty,
        )
        if _lp_for_reduction is not None:
            _before_reduction = len(pool)
            pool = column_reduction(pool, _lp_for_reduction["reduced_costs"])
            print(f"  {_before_reduction} -> {len(pool)} pairings", flush=True)
        else:
            print("  [warn] LP relaxation failed to solve -- skipping column reduction", flush=True)

    print(f"\nSolving IP (n_flights={n_total}, pool={len(pool)}, time_limit={ip_time_limit}s, lambda_dh={lambda_dh})...", flush=True)
    if full_flight_master:
        rescue_columns = None
        if rescue_pool_path:
            with open(rescue_pool_path, "r", encoding="utf-8") as handle:
                rescue_columns = json.load(handle)
            if isinstance(rescue_columns, dict):
                rescue_columns = rescue_columns.get("columns", rescue_columns.get("rescue_columns", []))
            rescue_columns = validate_rescue_columns_current_run(
                rescue_columns, flights_by_id, constraint, base_ids
            )
        elif auto_rescue:
            # rescue_generator: policy/salvage 후보가 아예 없는 flight마다 "허용 base ->
            # target -> 허용 base"로 legal한 pairing을 예산 제한 BFS로 찾아본다. rescue는
            # (operational/artificial과 달리) 진짜 legal 근무로 인정되는 마지막 단계라 여기서
            # 최대한 시도해볼 가치가 있음
            covered = {leg for p in pool for leg in p["legs"]}
            uncovered_flight_ids = [fid for fid in range(n_total) if fid not in covered]
            print(f"\nGenerating rescue candidates (uncovered={len(uncovered_flight_ids)})...", flush=True)
            rescue_pool = {}
            for base_id in base_ids:
                base_constraint = {**constraint, "base_airport": base_id}
                generated = generate_rescue_candidates(flights_by_id, base_constraint, uncovered_flight_ids)
                for candidate in generated["candidates"]:
                    key = tuple(sorted(candidate["legs"]))
                    if key not in rescue_pool or candidate["cost"] < rescue_pool[key]["cost"]:
                        rescue_pool[key] = candidate
            rescue_columns = list(rescue_pool.values())
            print(f"  rescue candidates: {len(rescue_columns)} (bases tried: {len(base_ids)})", flush=True)
            if rescue_columns:
                rescue_columns = validate_rescue_columns_current_run(
                    rescue_columns, flights_by_id, constraint, base_ids
                )
        result = solve_pool_completion(
            pool, n_total, lambda_excess=lambda_dh, time_limit=ip_time_limit,
            threads=ip_threads,
            use_gurobi=(ip_solver == "gurobi"),
            reposition_penalty=reposition_penalty, reserve_penalty=reserve_penalty,
            artificial_penalty=artificial_penalty,
            auto_operational=auto_operational,
            report_path=completion_report_path, rescue_columns=rescue_columns, verbose=True,
        )
        print(render_completion_table(result["completion_report"]), flush=True)
    else:
        result = solve_set_covering(
            pool, n_flights=n_total, time_limit=ip_time_limit,
            lambda_dh=lambda_dh, verbose=True,
        )
    print("IP solve complete", flush=True)

    validation_report = validate_selected_pairings(
        result["selected"], flights_by_id, constraint, base_ids, n_total,
        strict=(strict_validation or full_flight_master),
    )
    print(
        f"\n[validator] independent re-check: "
        f"{validation_report['n_invalid_selected']} invalid selected pairing(s) "
        f"(validator_version={validation_report['validator_version']})",
        flush=True,
    )

    gap_pct = None
    if compute_gap and not full_flight_master:
        print(f"\nSolving LP relaxation (for Gap%, pool={len(pool)})...", flush=True)
        lp_result = solve_lp_relaxation(pool, lambda_dh=lambda_dh)
        if lp_result is not None and lp_result["lp_value"]:
            gap_pct = (result["mip_obj"] - lp_result["lp_value"]) / lp_result["lp_value"] * 100
        else:
            print("  [warn] LP relaxation failed to solve -- cannot compute Gap%")

    if not full_flight_master and (result["uncoverable"] > 0 or result["coverage"] < 1.0):
        raise RuntimeError(
            "CPP 해를 구성하지 못했습니다: coverage={:.3f}, uncoverable={}".format(
                result["coverage"], result["uncoverable"]
            )
        )

    sel          = result["selected"]
    fly_total    = sum(p["fly"]                         for p in sel) if sel else 0.0
    raw_dead_total = sum(p.get("dead_time", p["cost"])  for p in sel) if sel else 0.0
    legs_total   = sum(p.get("n_legs", len(p["legs"])) for p in sel) if sel else 0
    duties_total = sum(p.get("n_duties", 1)            for p in sel) if sel else 0
    man_days     = sum(math.ceil(p["elapsed"] / 24.0)  for p in sel) if sel else 0
    avg_legs     = legs_total   / len(sel) if sel else 0.0
    avg_duties   = duties_total / len(sel) if sel else 0.0
    # FTC reflects only within-duty gaps (excludes overnight excess); cost is
    # left as-is (preserves the ManDays incentive)
    intra_gap_total    = sum(p.get("intra_duty_gap", 0.0)    for p in sel) if sel else 0.0
    inter_excess_total = sum(p.get("inter_duty_excess", 0.0) for p in sel) if sel else 0.0
    # Total dead time is reported on the same basis as FTC (within-duty gaps
    # only, excludes overnight excess) -- raw_dead_total (the cost-computation
    # basis, includes overnight excess) is shown separately for reference
    dead_total = intra_gap_total
    ftc = intra_gap_total / fly_total * 100 if fly_total > 0 else 0.0

    print()
    print("=" * 60)
    print(f"Results (covering all {n_total} legs)")
    print("=" * 60)
    print(f"  n pairings:        {result['n_pairings']}")
    print(f"  ManDays:           {man_days}")
    print(f"  coverage:          {result['coverage'] * 100:.1f}%")
    if full_flight_master:
        print(f"  completion:        {result['completion_coverage'] * 100:.1f}%")
        print(f"  artificial:        {result['artificial_count']} legs")
    print(f"  uncoverable:       {result['uncoverable']} legs")
    print(f"  deadhead:          {result['deadhead_count']} legs")
    print(f"  fly time:          {fly_total:.2f}h")
    print(f"  dead time (within-duty gaps only, excl. overnight): {dead_total:.2f}h")
    print(f"  (ref) raw dead time (incl. overnight excess, cost-computation basis): {raw_dead_total:.2f}h")
    print(f"    - within-duty connection gap:     {intra_gap_total:.2f}h ({intra_gap_total/raw_dead_total*100 if raw_dead_total>0 else 0:.1f}%)")
    print(f"    - inter-duty excess wait (>min_rest): {inter_excess_total:.2f}h ({inter_excess_total/raw_dead_total*100 if raw_dead_total>0 else 0:.1f}%)")
    print(f"  FTC:               {ftc:.2f}%")
    print(f"  avg legs/pairing:  {avg_legs:.2f}")
    print(f"  avg duties/pairing:{avg_duties:.2f}")
    print(f"  IP status:         {result['status']}")
    if gap_pct is not None:
        print(f"  Gap% (MIP vs LP):  {gap_pct:.3f}%")

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
            "ip_is_optimal":    result.get("is_optimal", False),
            "ip_pulp_status":   result.get("pulp_status"),
            "ip_solution_status": result.get("pulp_solution_status"),
            "gap_pct":          gap_pct,
            "eval_mode":        eval_mode,
            "n_invalid_selected": validation_report["n_invalid_selected"],
            "validator_version": validation_report["validator_version"],
        })
        wandb.finish()

    result["gap_pct"] = gap_pct
    result["eval_mode"] = eval_mode
    result["dual_trace"] = dual_trace
    result["validation_report"] = validation_report

    if save_json_path:
        save_result_json(save_json_path, result, checkpoint_path, airline, eval_mode)
        print(f"\n[save] selected pairings + validation report -> {save_json_path}", flush=True)

    return result


# ── CLI ────────────────────────────────────────────────────────────────────────

def _airline_output_path(path, airline):
    """multi CLI에서 항공사별 결과가 서로 덮어쓰지 않게 suffix를 붙임."""
    if path is None:
        return None
    root, ext = os.path.splitext(path)
    return f"{root}_{airline}{ext}"


def evaluate_multi_airline(checkpoint_path, *, summary_path=None, **kwargs):
    """한 호출에서 multi checkpoint를 세 BTS 항공사에 각각 독립 평가함."""
    if kwargs.get("data_path") is not None:
        raise ValueError("--airline multi에서는 항공사별 기본 data path를 사용해야 함")
    requested_window = kwargs.pop("window_days", None)
    results = {}
    for airline in config.MULTI_AIRLINES:
        airline_kwargs = dict(kwargs)
        airline_kwargs["window_days"] = _resolve_window_days(airline, requested_window)
        for key in ("completion_report_path", "dual_trace_path", "save_json_path"):
            airline_kwargs[key] = _airline_output_path(kwargs.get(key), airline)
        print(f"\n{'#' * 72}\n# MULTI EVAL: {airline}\n{'#' * 72}", flush=True)
        results[airline] = evaluate_full(
            checkpoint_path=checkpoint_path,
            airline=airline,
            **airline_kwargs,
        )
    if summary_path:
        os.makedirs(os.path.dirname(summary_path) or ".", exist_ok=True)
        summary = {
            airline: {
                "status": result.get("status"),
                "coverage": result.get("coverage"),
                "completion_coverage": result.get("completion_coverage"),
                "artificial_count": result.get("artificial_count"),
                "mip_obj": result.get("mip_obj"),
                "n_pairings": result.get("n_pairings"),
            }
            for airline, result in results.items()
        }
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2, default=str)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full monthly-schedule flight-coverage evaluation")
    parser.add_argument("checkpoint", help="Checkpoint file path (e.g. checkpoints/jbkwcdk3/phase2_best.pt)")
    parser.add_argument("--airline",   default="delta", choices=["delta", "alaska", "jetblue", "turkish", "multi"])
    parser.add_argument("--data-path", default=None,
                        help="CSV path. Uses config.AIRLINE_DATA[airline] if unset. "
                             "Set this for small-scale sample evaluation (e.g. RL/data/sample_DL_*.csv)")
    parser.add_argument("--n-rollouts-per-chunk", type=int, default=15,
                        help="Stochastic rollouts per chunk. Each window is split into sequential subset_size-sized chunks (default: 15)")
    parser.add_argument("--window-days", type=int, default=None,
                        help="Window size in days. 미지정 시 항공사별 설정(delta=6, alaska=6, jetblue=8)")
    parser.add_argument("--subset-size", type=int, default=config.EPISODE_MAX_FLIGHTS,
                        help=f"Flights per rollout (default: {config.EPISODE_MAX_FLIGHTS})")
    parser.add_argument("--ip-time-limit", type=int, default=3600,
                        help="IP solver time limit in seconds (default: 3600)")
    parser.add_argument("--ip-threads", type=int, default=1,
                        help="IP solver threads (default: 1)")
    parser.add_argument("--ip-solver", choices=["cbc", "gurobi"], default="cbc",
                        help="full-flight master solver (default: cbc)")
    parser.add_argument("--lambda-dh", type=float, default=10.0,
                        help="DH penalty weight (default: 10.0)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--turkish-files", nargs="+", default=None,
                        help="Turkish only. List of .legs file names to use. Defaults to the "
                             "Zeren Feb benchmark window (tt201402.legs, 2/1-3/8, 15,742 legs) "
                             "if unset. If given explicitly, uses those files in full with no date filter.")
    parser.add_argument("--wandb", action="store_true",
                        help="Log the eval config + console output + final result metrics to wandb (job_type=eval)")
    parser.add_argument("--wandb-project", default="ASCP-2026-paper")
    parser.add_argument("--compute-gap", action="store_true",
                        help="After solving the MIP, also solve the LP relaxation over the same "
                             "pool to compute Gap%%=(MIP_obj-LP_obj)/LP_obj*100 (same definition as "
                             "Tahir et al. Table 6). Off by default since the LP adds extra time on large pools")
    parser.add_argument("--full-flight-master", action="store_true",
                        help="전체 flight constraint와 단계별 completion master 사용")
    parser.add_argument("--rescue-pool-path", default=None,
                        help="generator가 저장한 rescue column JSON 경로 -- 주면 자동 생성 대신 이 파일을 씀")
    parser.add_argument("--no-auto-rescue", action="store_false", dest="auto_rescue",
                        help="--full-flight-master에서 rescue candidate 자동 생성 비활성화 "
                             "(--rescue-pool-path 미지정 시 기본은 자동 생성)")
    parser.add_argument("--auto-operational", action="store_true",
                        help="실제 eligibility 입력이 없을 때 미커버 flight 전체를 reposition/reserve 가능하다고 보는 proxy를 명시적으로 활성화")
    parser.add_argument("--completion-report-path", default=None,
                        help="V2 completion JSON 저장 경로")
    parser.add_argument("--reposition-penalty", type=float, default=None)
    parser.add_argument("--reserve-penalty", type=float, default=None)
    parser.add_argument("--artificial-penalty", type=float, default=None)
    parser.add_argument("--dual-iterations", type=int, default=0,
                        help="current master dual로 pool을 반복 보강할 횟수")
    parser.add_argument("--dual-weight", type=float, default=1.0,
                        help="decoder action logit에 더할 normalized dual 가중치")
    parser.add_argument("--dual-mode", choices=["real", "zero", "uncovered-only", "shuffled", "uniform"], default="real")
    parser.add_argument("--dual-artificial-penalty", type=float, default=None,
                        help="Unset uses Cmax x2 (pool's own legal cost scale) instead of a flat value")
    parser.add_argument("--dual-trace-path", default=None)
    parser.add_argument("--seed", type=int, default=None,
                        help="Fix the random/torch RNG -- set this to run a paired comparison of "
                             "multiple checkpoints against the same evaluation instance (e.g. the "
                             "same seed for every ON/OFF checkpoint)")
    parser.add_argument("--strict-validation", action="store_true",
                        help="Raise if evaluation/validator.py finds any independently-invalid "
                             "pairing among the selected solution (v1.md C3) -- off by default so "
                             "a single unexpected invalid doesn't kill a long eval run; the count "
                             "is always printed and returned in result['validation_report'] either way.")
    parser.add_argument("--save-json", default=None,
                        help="Save selected pairings (incl. flight-ID legs) + validation report "
                             "to this JSON path (v1.md C3). Saved by default even without this "
                             "flag (path auto-derived from checkpoint/airline/eval_mode under "
                             "log/eval_json/) -- needed for evaluation/ascp_output_adapter.py to "
                             "re-score a past run later; pass --no-save-json to opt out.")
    parser.add_argument("--no-save-json", action="store_true",
                        help="Skip saving the JSON result entirely (by default it's always saved).")
    parser.add_argument("--multi-summary-path", default=None,
                        help="--airline multi 실행의 항공사별 핵심 지표 summary JSON")
    args = parser.parse_args()

    ckpt = args.checkpoint
    if not os.path.exists(ckpt):
        candidate = os.path.join("checkpoints", ckpt)
        if os.path.exists(candidate):
            ckpt = candidate

    common_kwargs = dict(
        checkpoint_path=ckpt,
        data_path=args.data_path,
        n_rollouts_per_chunk=args.n_rollouts_per_chunk,
        window_days=args.window_days,
        subset_size=args.subset_size,
        ip_time_limit=args.ip_time_limit,
        ip_threads=args.ip_threads,
        ip_solver=args.ip_solver,
        lambda_dh=args.lambda_dh,
        device=args.device,
        turkish_files=args.turkish_files,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        compute_gap=args.compute_gap,
        full_flight_master=args.full_flight_master,
        completion_report_path=args.completion_report_path,
        rescue_pool_path=args.rescue_pool_path,
        auto_rescue=args.auto_rescue,
        auto_operational=args.auto_operational,
        reposition_penalty=args.reposition_penalty,
        reserve_penalty=args.reserve_penalty,
        artificial_penalty=args.artificial_penalty,
        dual_iterations=args.dual_iterations,
        dual_weight=args.dual_weight,
        dual_mode=args.dual_mode,
        dual_artificial_penalty=args.dual_artificial_penalty,
        dual_trace_path=args.dual_trace_path,
        seed=args.seed,
        strict_validation=args.strict_validation,
        save_json_path=args.save_json,
        no_save_json=args.no_save_json,
    )
    if args.airline == "multi":
        evaluate_multi_airline(
            summary_path=args.multi_summary_path,
            **common_kwargs,
        )
    else:
        evaluate_full(airline=args.airline, **common_kwargs)
