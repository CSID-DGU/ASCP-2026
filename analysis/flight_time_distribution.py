"""
비행 시간 분포 분석 및 이론적 avg_legs 기준 도출

목적:
  - Delta/Alaska/JetBlue 2019-01 데이터에서 비행 시간 분포를 시각화
  - max_duty(13h) 기준으로 이론적 avg_legs 상한 계산
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ─── 데이터 로드 ───────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "RL", "data")

files = {
    "Delta":   "delta_2019_01.csv",
    "Alaska":  "alaska_2019_01.csv",
    "JetBlue": "jetblue_2019_01.csv",
}

dfs = {}
for airline, fname in files.items():
    path = os.path.join(DATA_DIR, fname)
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    # CRS_ELAPSED_TIME: 분 단위 → 시간 단위
    df["flight_hours"] = df["CRS_ELAPSED_TIME"] / 60.0
    dfs[airline] = df
    print(f"{airline}: {len(df)}편, avg {df['flight_hours'].mean():.2f}h, "
          f"median {df['flight_hours'].median():.2f}h")

# ─── 이론적 avg_legs 기준 계산 ─────────────────────────────────────────────────
MAX_DUTY_H = 13.0        # FAA 최대 duty 시간 (우리 실험 기준)
MIN_CONN_H = 0.5         # 최소 연결 시간 (30분)

print("\n=== 이론적 avg_legs 상한 (max_duty=13h 기준) ===")
print(f"{'항공사':>8} {'avg flight(h)':>14} {'median flight(h)':>17} "
      f"{'avg_legs 상한(avg)':>19} {'avg_legs 상한(median)':>22}")
print("-" * 85)

for airline, df in dfs.items():
    avg_ft = df["flight_hours"].mean()
    med_ft = df["flight_hours"].median()

    # duty 내 legs 수 = floor((max_duty) / (avg_flight + min_conn))
    # 마지막 leg는 연결 없으므로: n * avg_ft + (n-1) * min_conn <= max_duty
    # → n <= (max_duty + min_conn) / (avg_ft + min_conn)
    n_avg = (MAX_DUTY_H + MIN_CONN_H) / (avg_ft + MIN_CONN_H)
    n_med = (MAX_DUTY_H + MIN_CONN_H) / (med_ft + MIN_CONN_H)

    print(f"{airline:>8} {avg_ft:>14.2f} {med_ft:>17.2f} "
          f"{n_avg:>19.2f} {n_med:>22.2f}")

# ─── 시각화 ────────────────────────────────────────────────────────────────────
# 상단: 항공사별 비행 시간 분포 bar chart (1h 단위)
# 하단: 항공사별 이론적 avg_legs 상한 bar chart
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Flight Time Distribution & Theoretical avg_legs Upper Bound (2019-01)",
             fontsize=14, fontweight="bold")

bins = np.arange(0, 15, 1)  # 0~14h, 1h 단위 (max_duty=13h 포함)
colors = {"Delta": "#003087", "Alaska": "#00665C", "JetBlue": "#003876"}

avg_legs_data = {}

for col, (airline, df) in enumerate(dfs.items()):
    ax = axes[0][col]
    ft = df["flight_hours"].dropna()

    counts, edges = np.histogram(ft, bins=bins)
    ax.bar(edges[:-1] + 0.5, counts, width=0.8, color=colors[airline], alpha=0.85,
           edgecolor="white", linewidth=0.5)

    avg_ft = ft.mean()
    med_ft = ft.median()
    n_avg = (MAX_DUTY_H + MIN_CONN_H) / (avg_ft + MIN_CONN_H)
    n_med = (MAX_DUTY_H + MIN_CONN_H) / (med_ft + MIN_CONN_H)
    avg_legs_data[airline] = {"n_avg": n_avg, "n_med": n_med,
                               "avg_ft": avg_ft, "med_ft": med_ft}

    ax.axvline(avg_ft, color="red",   linestyle="--", linewidth=1.8, label=f"mean = {avg_ft:.2f}h")
    ax.axvline(med_ft, color="orange", linestyle=":", linewidth=1.8, label=f"median = {med_ft:.2f}h")
    ax.axvline(MAX_DUTY_H, color="gray", linestyle="-", linewidth=1.5,
               alpha=0.7, label=f"max duty = {MAX_DUTY_H}h")

    ax.set_title(airline, fontsize=13, fontweight="bold", color=colors[airline])
    ax.set_xlabel("Flight Duration (hours)", fontsize=10)
    ax.set_ylabel("Number of Flights", fontsize=10)
    ax.set_xticks(range(0, 15))
    ax.set_xlim(0, 14)
    ax.legend(fontsize=9)

# 하단: avg_legs 이론 상한 비교
for col, (airline, data) in enumerate(avg_legs_data.items()):
    ax = axes[1][col]
    labels = ["Mean-based", "Median-based"]
    values = [data["n_avg"], data["n_med"]]
    bars = ax.bar(labels, values, color=[colors[airline], colors[airline]],
                  edgecolor="white", width=0.5)
    bars[0].set_alpha(0.9)
    bars[1].set_alpha(0.6)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{val:.2f}", ha="center", va="bottom", fontsize=13, fontweight="bold")

    ax.axhline(3.0, color="red", linestyle="--", linewidth=1.5, alpha=0.7, label="target (3.0)")
    ax.set_title(f"{airline} — avg_legs upper bound", fontsize=11, fontweight="bold")
    ax.set_ylabel("avg_legs (theoretical max)", fontsize=10)
    ax.set_ylim(0, 6.5)
    ax.legend(fontsize=9)

plt.tight_layout()

out_path = os.path.join(os.path.dirname(__file__), "flight_time_dist.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\n그래프 저장: {out_path}")
plt.show()

# ─── 비행 시간 분위수 테이블 ──────────────────────────────────────────────────
print("\n=== 비행 시간 분위수 (hours) ===")
for airline, df in dfs.items():
    ft = df["flight_hours"].dropna()
    quantiles = ft.quantile([0.1, 0.25, 0.5, 0.75, 0.9])
    print(f"\n{airline}:")
    for q, v in quantiles.items():
        print(f"  P{int(q*100):2d}: {v:.2f}h")
    print(f"  mean: {ft.mean():.2f}h  |  2h 이하: {(ft<=2).mean()*100:.1f}%"
          f"  |  3h 이하: {(ft<=3).mean()*100:.1f}%"
          f"  |  5h 이하: {(ft<=5).mean()*100:.1f}%")
