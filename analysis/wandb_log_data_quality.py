"""기존 데이터 품질분석 그래프+수치를 wandb에 게시 (학습 큐와 별도 run)"""
import os
import sys

import wandb

sys.path.insert(0, os.path.dirname(__file__))
from flight_time_distribution import (  # noqa: E402
    DATA_DIR, FILES, AIRLINE_CONFIG, TARGET_OLD,
    load_flights, compute_quality, sim_network, sim_time_only, frechet_distance_1d,
)

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT_DIR = os.path.join(DATA_DIR, "small-scale")

run = wandb.init(
    project="ASCP-2026-paper",
    job_type="data-quality",
    name="data-quality-analysis",
)

table = wandb.Table(columns=[
    "airline", "n_full", "n_small",
    "full_bdr", "small_bdr", "full_car", "small_car", "fid",
    "duration_mean_h", "duration_median_h",
    "time_only_ceiling_mean", "time_only_ceiling_std",
    "network_feasible_mean", "network_feasible_std",
    "max_duty_periods",
])

for airline, fname in FILES.items():
    full_path = os.path.join(DATA_DIR, fname)
    small_path = os.path.join(OUT_DIR, fname.replace(".csv", "") + "_sample_chain.csv")
    cfg = AIRLINE_CONFIG[airline]

    full_q = load_flights(full_path)
    small_q = load_flights(small_path)
    full_fh = full_q["dur"] / 60.0
    small_fh = small_q["dur"] / 60.0

    full_bdr, full_car = compute_quality(full_q, cfg)
    small_bdr, small_car = compute_quality(small_q, cfg)
    fid = frechet_distance_1d(full_fh.values, small_fh.values)

    print(f"[{airline}] BDR/CAR/FID 계산 완료, avg_legs 시뮬레이션 시작 (N_SIMS=8000)...")
    t = sim_time_only(full_q["dur"].values, cfg)
    n = sim_network(full_q, cfg)

    table.add_data(
        airline, len(full_q), len(small_q),
        round(float(full_bdr), 4), round(float(small_bdr), 4),
        round(float(full_car), 4), round(float(small_car), 4), round(float(fid), 4),
        round(float(full_fh.mean()), 3), round(float(full_fh.median()), 3),
        round(float(t.mean()), 3), round(float(t.std()), 3),
        round(float(n.mean()), 3), round(float(n.std()), 3),
        cfg["max_duty_periods"],
    )
    print(f"[{airline}] 완료: FID={fid:.4f}  BDR {full_bdr:.3f}->{small_bdr:.3f}  "
          f"CAR {full_car:.3f}->{small_car:.3f}")

ref_table = wandb.Table(columns=["name", "value", "meaning"], data=[
    ["MIN_LEGS_FOR_PAIRING", TARGET_OLD, "avg_legs 하한 참고선 (플롯 상수)"],
    ["LLM_v0_avg_legs", 5.72, "Delta 데이터로 학습한 LLM v0 baseline avg_legs (플롯 상수)"],
    ["RL_current_avg_legs", 1.85, "당시 RL 정책 실측 avg_legs (플롯 상수)"],
])

png_map = {
    "figures/dataset_avg_chain":                   os.path.join(OUT_DIR, "dataset_avg_chain.png"),
    "figures/quality_full_vs_smallscale":          os.path.join(OUT_DIR, "quality_full_vs_smallscale.png"),
    "figures/duration_full_vs_smallscale_bin1h":   os.path.join(OUT_DIR, "duration_full_vs_smallscale_bin1h.png"),
}

log_dict = {k: wandb.Image(v) for k, v in png_map.items() if os.path.exists(v)}
log_dict["quality/summary_table"] = table
log_dict["quality/reference_constants"] = ref_table
wandb.log(log_dict)

artifact = wandb.Artifact("data-quality-analysis-code", type="code")
for rel in ["analysis/flight_time_distribution.py",
            "analysis/dataset_analysis_chain.py",
            "analysis/compare_full_vs_new_smallscale.py",
            "analysis/wandb_log_data_quality.py"]:
    artifact.add_file(os.path.join(REPO_ROOT, rel), name=rel)
run.log_artifact(artifact)

print(f"\n완료: {run.url}")
wandb.finish()
