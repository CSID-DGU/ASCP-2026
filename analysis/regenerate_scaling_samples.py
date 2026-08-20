"""
regenerate_scaling_samples.py — delta 스케줄을 5개 크기(3.5k/7k/14k/28k/56k)로
잘라서 scaling 실험용 subinstance를 만든다. `regenerate_chain_sampling.py`의
connectivity-preserving chain-sampling 로직을 그대로 재사용하되, 타겟 크기가
클수록 윈도우(WINDOW_END)도 같이 늘려서 표본 pool이 부족해지지 않게 한다.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from regenerate_chain_sampling import (
    load_full_with_date, build_adjacency, chain_sample, RNG,
)
from flight_time_distribution import DATA_DIR, FILES, AIRLINE_CONFIG

WINDOW_START = "2019-01-01"
# (target_size, window_days) — window_days는 target보다 넉넉한 margin을 두도록 선정
SCALE_POINTS = [
    (3500, 7),
    (7000, 10),
    (14000, 14),
    (28000, 21),
    (56000, 31),
]


def main():
    airline = "Delta"
    cfg = AIRLINE_CONFIG[airline]
    full_df = load_full_with_date(os.path.join(DATA_DIR, FILES[airline]))

    out_dir = os.path.join(DATA_DIR, "scaling")
    os.makedirs(out_dir, exist_ok=True)

    for n_target, window_days in SCALE_POINTS:
        window_end = (pd.Timestamp(WINDOW_START) + pd.Timedelta(days=window_days)).strftime("%Y-%m-%d")
        window_df = full_df[(full_df["FL_DATE"] >= WINDOW_START) & (full_df["FL_DATE"] < window_end)]

        RNG.bit_generator.state = __import__("numpy").random.default_rng(0).bit_generator.state
        adj = build_adjacency(window_df, cfg["min_conn"], cfg["max_conn"])
        sampled = chain_sample(window_df, adj, n_target)

        raw_cols = ["ORIGIN", "DEST", "CRS_DEP_TIME", "CRS_ARR_TIME", "CRS_ELAPSED_TIME", "FL_DATE"]
        out_path = os.path.join(out_dir, f"delta_scaling_{n_target}.csv")
        sampled[raw_cols].to_csv(out_path, index=False)
        print(f"target={n_target:6d}  window={window_days:2d}d  actual_n={len(sampled):6d}  -> {out_path}")


if __name__ == "__main__":
    main()
