"""
regenerate_chain_sampling.py — flight-pair(연결편) 보존 샘플링으로 CAR를 근본적으로 개선.

§9(station-set 필터링)는 효과가 작았다 — Delta 윈도우는 스포크 몇 개만 넣어도 91%가 이미
연결돼 있어서, "어떤 공항을 포함할지" 필터링으로는 못 거른다. 진짜 문제는 "독립적으로 무작위
추출하면 이 편의 실제 연결편이 같이 뽑힐 확률이 낮다"는 것 — 그래서 이번엔 개별 flight이 아니라
"연결 체인"(한 편 + 그 편의 실제 다음 연결편, 2~4홉) 단위로 샘플링해서, 뽑힌 편의 연결편이
"거의 항상" 같이 포함되도록 한다.
"""
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from flight_time_distribution import (
    DATA_DIR, FILES, AIRLINE_CONFIG, compute_quality, frechet_distance_1d, _offset,
)

N_TARGET = 3500
WINDOW_START, WINDOW_END = "2019-01-01", "2019-01-08"
RNG = np.random.default_rng(0)
MAX_CHAIN_LEN = 4


def load_full_with_date(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=["ORIGIN", "DEST", "CRS_DEP_TIME", "CRS_ARR_TIME",
                            "CRS_ELAPSED_TIME", "FL_DATE"])
    df["FL_DATE"] = pd.to_datetime(df["FL_DATE"], format="mixed")
    base_date = df["FL_DATE"].min()
    origin, dest = df["ORIGIN"], df["DEST"]
    day_off = (df["FL_DATE"] - base_date).dt.days
    dep_local = df["CRS_DEP_TIME"].astype(int)
    dep_local_min = (dep_local // 100) * 60 + (dep_local % 100) + day_off * 1440
    dep_abs = dep_local_min - origin.map(_offset)
    dur = df["CRS_ELAPSED_TIME"].round().astype(int)
    arr_abs = dep_abs + dur
    return pd.DataFrame({
        "origin": origin, "dest": dest, "dep_abs": dep_abs, "arr_abs": arr_abs,
        "dur": dur, "fh": dur / 60.0, "FL_DATE": df["FL_DATE"],
        # 원본 BTS 컬럼도 같이 보존 — 샘플 출력은 이 컬럼들로 저장해서
        # RL/loader.py의 기존 로더(raw BTS 스키마)를 그대로 재사용할 수 있게 함
        "ORIGIN": origin, "DEST": dest,
        "CRS_DEP_TIME": df["CRS_DEP_TIME"], "CRS_ARR_TIME": df["CRS_ARR_TIME"],
        "CRS_ELAPSED_TIME": df["CRS_ELAPSED_TIME"],
    }).reset_index(drop=True)


def build_adjacency(window_df, min_conn, max_conn):
    """flight index -> 연결 가능한 다음 flight index 리스트."""
    by_origin = defaultdict(list)
    for idx, o, dep in zip(window_df.index, window_df["origin"], window_df["dep_abs"]):
        by_origin[o].append((dep, idx))
    for o in by_origin:
        by_origin[o].sort()

    adj = defaultdict(list)
    for idx, dest, arr in zip(window_df.index, window_df["dest"], window_df["arr_abs"]):
        candidates = by_origin.get(dest, [])
        lo, hi = arr + min_conn, arr + max_conn
        deps = [d for d, _ in candidates]
        import bisect
        i0 = bisect.bisect_left(deps, lo)
        i1 = bisect.bisect_right(deps, hi)
        adj[idx] = [j for _, j in candidates[i0:i1]]
    return adj


def chain_sample(window_df, adj, n_target, max_chain_len=MAX_CHAIN_LEN):
    all_idx = list(window_df.index)
    RNG.shuffle(all_idx)
    visited = set()
    selected = []

    for start in all_idx:
        if len(selected) >= n_target:
            break
        if start in visited:
            continue
        chain_len = RNG.integers(1, max_chain_len + 1)
        cur = start
        chain = [cur]
        visited.add(cur)
        for _ in range(chain_len - 1):
            nexts = [n for n in adj.get(cur, []) if n not in visited]
            if not nexts:
                break
            cur = nexts[RNG.integers(len(nexts))]
            chain.append(cur)
            visited.add(cur)
        selected.extend(chain)

    return window_df.loc[selected[:n_target]]


def main():
    print(f"{'airline':>8} {'BDR full':>9} {'BDR chain':>10} {'CAR full':>9} {'CAR chain':>10} {'FID chain':>10}")
    print("-" * 65)
    for airline in FILES:
        fname = FILES[airline]
        cfg = AIRLINE_CONFIG[airline]
        full_df = load_full_with_date(os.path.join(DATA_DIR, fname))
        window_df = full_df[(full_df["FL_DATE"] >= WINDOW_START) & (full_df["FL_DATE"] < WINDOW_END)]

        adj = build_adjacency(window_df, cfg["min_conn"], cfg["max_conn"])
        sampled = chain_sample(window_df, adj, N_TARGET)

        full_bdr, full_car = compute_quality(full_df, cfg)
        chain_bdr, chain_car = compute_quality(sampled, cfg)
        fid_chain = frechet_distance_1d(full_df["fh"].values, sampled["fh"].values)

        out_dir = os.path.join(DATA_DIR, "small-scale")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, fname.replace(".csv", "") + "_sample_chain.csv")
        raw_cols = ["ORIGIN", "DEST", "CRS_DEP_TIME", "CRS_ARR_TIME", "CRS_ELAPSED_TIME", "FL_DATE"]
        sampled[raw_cols].to_csv(out_path, index=False)

        print(f"{airline:>8} {full_bdr:>9.3f} {chain_bdr:>10.3f} {full_car:>9.3f} {chain_car:>10.3f} {fid_chain:>10.4f}")
        print(f"  n={len(sampled)}  저장: {out_path}")


if __name__ == "__main__":
    main()
