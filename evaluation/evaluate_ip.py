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
import math
import json
import random
import argparse

import torch
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "RL"))

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
    "turkish": get_turkish_constraints_hb,  # HB1/HB2 cross-base return contract
}
from model import FlightEncoder, PointerDecoder
from evaluation.set_partition import solve_set_covering, solve_lp_relaxation
from evaluation.completion_runner import merge_rescue_columns, solve_completion_stages
from evaluation.completion_report import build_completion_report, render_completion_table, save_completion_report
from evaluation.validator import validate_pairing
from utils import constraint_to_tensor, flights_to_tensors
from rollout import rollout_with_pairings, set_environment
from base_reach import build_base_reaches
import config


# ── 0. Turkish window loader ─────────────────────────────────────────────────

def load_windows_turkish(turkish_df, airport_map, window_days=5):
    """Split Turkish .legs data into window_days-sized non-overlapping windows and assign global IDs."""
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
            f"  window offset={offset:2d}: {len(wf):5d}legs "
            f"(global {global_offset - len(wf)} ~ {global_offset - 1})",
            flush=True,
        )

    return windows, global_offset


# ── 1. Load the full dataset window by window, assigning global IDs ─────────

def load_windows_with_global_ids(data_path, airport_map, window_days=5, use_utc=False):
    """Split the full CSV into window_days-sized non-overlapping windows and assign global IDs.

    use_utc: if True, anchor dep_time to UTC (see RL/loader.py) -- only enable
        this when evaluating a checkpoint trained with the same option; using
        it with an existing checkpoint puts the model out-of-distribution.

    Returns:
        windows    : list of flight lists. Each flight gets a 'global_id' field.
        n_total    : total number of flights (= upper bound on global IDs)
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
            f"  window offset={offset:2d}: {len(wf):5d}legs "
            f"(global {global_offset - len(wf)} ~ {global_offset - 1})",
            flush=True,
        )

    return windows, global_offset


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

def rollout_subset_global(subset, constraint, encoder, decoder, max_time, greedy=False):
    """Rollout 결과를 독립 검증한 뒤 global ID column으로 변환함."""
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
    local_by_id = {f["id"]: f for f in local_flights}
    validated_pairings = []
    for pairing in raw_pairings:
        validation = validate_pairing(pairing, local_by_id, constraint)
        if not validation["is_valid"]:
            continue
        pairing["is_legal"] = True
        pairing["validator_version"] = validation["validator_version"]
        pairing["constraint_hash"] = validation["constraint_hash"]
        pairing["legs"] = [id_map[leg] for leg in pairing["legs"]]
        validated_pairings.append(pairing)
    return validated_pairings


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
                      n_rollouts_per_chunk=5,
                      subset_size=config.EPISODE_MAX_FLIGHTS,
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
    max_time = 5 * 24.0
    base_id_set = set(base_ids)

    for w_idx, window_flights in enumerate(windows):
        if not window_flights:
            continue

        window_all_ids = set(f["global_id"] for f in window_flights)
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
                    new_chunk = sorted([{**inject}] + list(chunk[:-1]),
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

            for _ in range(n_rollouts_per_chunk):
                try:
                    pairings = rollout_subset_global(chunk, c_b, encoder, decoder, max_time, greedy=False)
                except Exception as e:
                    print(f"  [warn] stochastic rollout failed (chunk={c_idx}): {e}", flush=True)
                    continue
                for p in pairings:
                    # Exclude pairings that don't return to base from both the
                    # pool and coverage counts -- including them in coverage
                    # would create "phantom" coverage that the IP can never
                    # actually select, so window_covered/covered_global must
                    # be filtered the same way.
                    if not _pairing_valid(p):
                        continue
                    key = tuple(sorted(p["legs"]))
                    if key not in pool or p["cost"] < pool[key]["cost"]:
                        pool[key] = p
                    window_covered.update(p["legs"])
                    covered_global.update(p["legs"])
                rollout_count += 1

            try:
                pairings = rollout_subset_global(chunk, c_b, encoder, decoder, max_time, greedy=True)
            except Exception as e:
                print(f"  [warn] greedy rollout failed (chunk={c_idx}): {e}", flush=True)
                continue
            for p in pairings:
                if not _pairing_valid(p):
                    continue
                key = tuple(sorted(p["legs"]))
                if key not in pool or p["cost"] < pool[key]["cost"]:
                    pool[key] = p
                window_covered.update(p["legs"])
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

    total_flights = sum(len(w) for w in windows)
    print(f"\ntotal pool: {len(pool)} pairings")
    print(f"total coverage: {len(covered_global)}/{total_flights} legs")
    final_pool = list(pool.values())
    for index, pairing in enumerate(final_pool):
        pairing.setdefault("source_type", "policy")
        pairing.setdefault("is_legal", True)
        pairing["column_id"] = f"{pairing['source_type']}-{index}"
    return final_pool, covered_global



def solve_pool_completion(
    pool, n_total, *, lambda_excess=1.0, time_limit=300,
    reposition_penalty=None, reserve_penalty=None,
    artificial_penalty=None, report_path=None, rescue_columns=None, verbose=False,
):
    """수집된 pool을 V2 단계별 master로 풀고 legacy 출력 호환 필드를 추가함."""
    if rescue_columns:
        pool = merge_rescue_columns(pool, rescue_columns, range(n_total))
    stages = solve_completion_stages(
        pool, range(n_total), lambda_excess=lambda_excess,
        time_limit=time_limit, reposition_penalty=reposition_penalty,
        reserve_penalty=reserve_penalty, artificial_penalty=artificial_penalty,
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

# ── 5. Main evaluation function ──────────────────────────────────────────────

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
    compute_gap=False,
    full_flight_master=False,
    completion_report_path=None,
    rescue_pool_path=None,
    reposition_penalty=None,
    reserve_penalty=None,
    artificial_penalty=None,
    seed=None,
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

    # Load the checkpoint first to check its vocab size -- a multi-airline
    # model (n_airports=168) needs the merged airport map; building it from a
    # single-airline map would cause an embedding-index mismatch.
    ckpt       = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    n_airports = ckpt.get("n_airports",
                          ckpt["encoder"]["airport_emb.weight"].shape[0])

    _turkish_df = None
    if airline == "turkish":
        # If turkish_files is unset, default to the Zeren Feb benchmark
        # window (15,742 legs, 0.03% off the target 15,738)
        if turkish_files is None:
            _turkish_df = parse_legs_dir(data_path, files=[ZEREN_FEB_FILE], date_range=ZEREN_FEB_WINDOW)
        else:
            _turkish_df = parse_legs_dir(data_path, files=turkish_files)
        airport_map = build_airport_map_turkish(df=_turkish_df)
    else:
        if n_airports > 145:
            # Turkish (.legs directory) can't be processed by the BTS CSV loader -> exclude
            map_paths = [v for k, v in config.AIRLINE_DATA.items() if k != "turkish"]
        else:
            map_paths = data_path
        airport_map = build_airport_map(map_paths)
    base_ids = bases_to_ids(list(bases), airport_map)

    encoder = FlightEncoder(n_airports=n_airports, constraint_dim=len(FILM_CONSTRAINT_KEYS)).to(DEVICE)
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

    print(f"\nLoading full dataset ({airline}, window_days={window_days})...", flush=True)
    if airline == "turkish":
        windows, n_total = load_windows_turkish(_turkish_df, airport_map, window_days)
    else:
        windows, n_total = load_windows_with_global_ids(data_path, airport_map, window_days, use_utc=use_utc)
    print(f"total {n_total} legs, {len(windows)} windows", flush=True)

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
            connected_sampler=connected_sampler,
            airline=airline,
        )

    print(f"\nSolving IP (n_flights={n_total}, pool={len(pool)}, time_limit={ip_time_limit}s, lambda_dh={lambda_dh})...", flush=True)
    if full_flight_master:
        rescue_columns = None
        if rescue_pool_path:
            with open(rescue_pool_path, "r", encoding="utf-8") as handle:
                rescue_columns = json.load(handle)
            if isinstance(rescue_columns, dict):
                rescue_columns = rescue_columns.get("columns", rescue_columns.get("rescue_columns", []))
        result = solve_pool_completion(
            pool, n_total, lambda_excess=lambda_dh, time_limit=ip_time_limit,
            reposition_penalty=reposition_penalty, reserve_penalty=reserve_penalty,
            artificial_penalty=artificial_penalty,
            report_path=completion_report_path, rescue_columns=rescue_columns, verbose=True,
        )
        print(render_completion_table(result["completion_report"]), flush=True)
    else:
        result = solve_set_covering(
            pool, n_flights=n_total, time_limit=ip_time_limit,
            lambda_dh=lambda_dh, verbose=True,
        )
    print("IP solve complete", flush=True)

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
            "gap_pct":          gap_pct,
        })
        wandb.finish()

    result["gap_pct"] = gap_pct
    return result


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full monthly-schedule flight-coverage evaluation")
    parser.add_argument("checkpoint", help="Checkpoint file path (e.g. checkpoints/jbkwcdk3/phase2_best.pt)")
    parser.add_argument("--airline",   default="delta", choices=["delta", "alaska", "jetblue", "turkish"])
    parser.add_argument("--data-path", default=None,
                        help="CSV path. Uses config.AIRLINE_DATA[airline] if unset. "
                             "Set this for small-scale sample evaluation (e.g. RL/data/sample_DL_*.csv)")
    parser.add_argument("--n-rollouts-per-chunk", type=int, default=5,
                        help="Stochastic rollouts per chunk. Each window is split into sequential subset_size-sized chunks (default: 5)")
    parser.add_argument("--window-days", type=int, default=5,
                        help="Window size in days. 1 is recommended for small-scale (1-week) data (default: 5)")
    parser.add_argument("--subset-size", type=int, default=config.EPISODE_MAX_FLIGHTS,
                        help=f"Flights per rollout (default: {config.EPISODE_MAX_FLIGHTS})")
    parser.add_argument("--ip-time-limit", type=int, default=3600,
                        help="CBC solver time limit in seconds (default: 3600)")
    parser.add_argument("--lambda-dh", type=float, default=1.0,
                        help="DH penalty weight (default: 1.0)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--turkish-files", nargs="+", default=None,
                        help="Turkish only. List of .legs file names to use. Defaults to the "
                             "Zeren Feb benchmark window (tt201402.legs, 2/1-3/8, 15,742 legs) "
                             "if unset. If given explicitly, uses those files in full with no date filter.")
    parser.add_argument("--use-utc", action="store_true",
                        help="Anchor dep_time as absolute UTC time. Only evaluate with this flag "
                             "for checkpoints trained with --use-utc -- enabling it for an existing "
                             "checkpoint puts the model out-of-distribution")
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
                        help="찬주 generator가 저장한 rescue column JSON 경로")
    parser.add_argument("--completion-report-path", default=None,
                        help="V2 completion JSON 저장 경로")
    parser.add_argument("--reposition-penalty", type=float, default=None)
    parser.add_argument("--reserve-penalty", type=float, default=None)
    parser.add_argument("--artificial-penalty", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None,
                        help="Fix the random/torch RNG -- set this to run a paired comparison of "
                             "multiple checkpoints against the same evaluation instance (e.g. the "
                             "same seed for every ON/OFF checkpoint)")
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
        compute_gap=args.compute_gap,
        full_flight_master=args.full_flight_master,
        completion_report_path=args.completion_report_path,
        rescue_pool_path=args.rescue_pool_path,
        reposition_penalty=args.reposition_penalty,
        reserve_penalty=args.reserve_penalty,
        artificial_penalty=args.artificial_penalty,
        seed=args.seed,
    )
