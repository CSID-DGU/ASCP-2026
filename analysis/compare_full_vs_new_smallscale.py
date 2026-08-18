"""
compare_full_vs_new_smallscale.py — full 데이터 vs 새로 만든 small-scale
(RL/data/small-scale/*_sample_chain.csv, raw BTS 스키마) 딱 두 개만 비교.

regenerate_chain_sampling.py가 raw BTS 컬럼(ORIGIN/DEST/CRS_DEP_TIME/...)으로 저장하도록
고쳐진 뒤 만든 파일이라, flight_time_distribution.py의 load_flights()를 그대로 재사용
가능(별도 파싱 분기 불필요).

출력(둘 다 RL/data/small-scale/ 저장):
  - duration_full_vs_smallscale.png : 항공사별 비행시간 분포 히스토그램(full vs small-scale)
  - quality_full_vs_smallscale.png  : BDR/CAR 막대 비교(full vs small-scale)
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from flight_time_distribution import (
    DATA_DIR, FILES, AIRLINE_CONFIG, load_flights, compute_quality, frechet_distance_1d,
)

OUT_DIR = os.path.join(DATA_DIR, "small-scale")
os.makedirs(OUT_DIR, exist_ok=True)

COLOR_FULL  = "#2a78d6"
COLOR_SMALL = "#eda100"

# target_avg_legs.png(flight_time_distribution.py 상단 패널)과 동일 팔레트/스타일
PALETTE  = {"Delta": "#3b3f6b", "Alaska": "#5e7e96", "JetBlue": "#9a8aa8"}
PANEL_BG = "#f3f3f8"


def plot_duration_panel(results, bin_width, out_path):
    """target_avg_legs.png 상단 패널과 동일 스타일: full=채운 막대, small-scale=흰 바탕
    해칭 막대(count를 scale만큼 키워 겹침), mean/median 점선 포함."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    fig.suptitle(
        f"Flight Duration — Full vs New Small-scale (bin={bin_width}h)",
        fontsize=13, fontweight="bold",
    )
    bins = np.arange(0, 15, bin_width)
    for ax, (airline, r) in zip(axes, results.items()):
        color = PALETTE[airline]
        ax.set_facecolor(PANEL_BG)
        ft, ft_s = r["full_fh"], r["small_fh"]

        counts, edges = np.histogram(ft, bins=bins)
        ax.bar(edges[:-1] + bin_width / 2, counts, bin_width * 0.86,
               color=color, alpha=0.9, edgecolor="white", lw=0.6, zorder=3,
               label=f"Full (n={len(ft):,})")

        s_counts, _ = np.histogram(ft_s, bins=bins)
        scale = len(ft) / len(ft_s)
        ax.bar(edges[:-1] + bin_width / 2, s_counts * scale, bin_width * 0.86,
               color="white", alpha=0.60, edgecolor=color, lw=1.8, hatch="///", zorder=4,
               label=f"Small-scale (n={len(ft_s):,}, x{scale:.0f})")

        ax.axvline(ft.mean(), color="#e4572e", ls="--", lw=1.6,
                   label=f"mean = {ft.mean():.2f}h")
        ax.axvline(ft.median(), color="#f2a541", ls=":", lw=1.8,
                   label=f"median = {ft.median():.2f}h")

        ax.set_title(
            f"{airline}   FID={r['fid']:.4f}\nCAR: {r['full_car']:.3f}→{r['small_car']:.3f}"
            f" (Δ{r['full_car']-r['small_car']:+.3f})",
            fontweight="bold", color=color, fontsize=10,
        )
        ax.set_xlabel("Flight Duration (hours)", fontsize=9.5)
        if ax is axes[0]:
            ax.set_ylabel("Number of Flights", fontsize=9.5)
        ax.set_xticks(range(0, 15, 2))
        ax.set_xlim(0, 14)
        ax.legend(fontsize=7.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print("저장:", out_path)


def main():
    results = {}
    for airline, fname in FILES.items():
        full_path  = os.path.join(DATA_DIR, fname)
        small_path = os.path.join(OUT_DIR, fname.replace(".csv", "") + "_sample_chain.csv")

        full_q  = load_flights(full_path)
        small_q = load_flights(small_path)

        full_fh  = full_q["dur"] / 60.0
        small_fh = small_q["dur"] / 60.0

        cfg = AIRLINE_CONFIG[airline]
        full_bdr, full_car   = compute_quality(full_q, cfg)
        small_bdr, small_car = compute_quality(small_q, cfg)
        fid = frechet_distance_1d(full_fh.values, small_fh.values)

        results[airline] = dict(
            full_fh=full_fh, small_fh=small_fh,
            full_bdr=full_bdr, full_car=full_car,
            small_bdr=small_bdr, small_car=small_car,
            fid=fid,
            n_full=len(full_q), n_small=len(small_q),
        )
        print(f"{airline:8s} full(n={len(full_q):6d}) BDR={full_bdr:.3f} CAR={full_car:.3f}  |  "
              f"small-scale(n={len(small_q):5d}) BDR={small_bdr:.3f} CAR={small_car:.3f}  |  FID={fid:.4f}")

    # ── Figure 1: duration histogram, full vs small-scale — target_avg_legs.png 스타일 ──
    # bin폭 0.5h/1h 두 버전 모두 남긴다(덮어쓰지 않음)
    plot_duration_panel(results, 0.5, os.path.join(OUT_DIR, "duration_full_vs_smallscale_bin0.5h.png"))
    plot_duration_panel(results, 1.0, os.path.join(OUT_DIR, "duration_full_vs_smallscale_bin1h.png"))

    # ── Figure 2: BDR/CAR bars, full vs small-scale ─────────────────────────
    airlines = list(results.keys())
    x = np.arange(len(airlines))
    w = 0.2

    fig, ax = plt.subplots(figsize=(9, 5))
    full_bdr  = [results[a]["full_bdr"]  for a in airlines]
    small_bdr = [results[a]["small_bdr"] for a in airlines]
    full_car  = [results[a]["full_car"]  for a in airlines]
    small_car = [results[a]["small_car"] for a in airlines]

    bars = [
        (x - 1.5 * w, full_bdr,  COLOR_FULL,  1.0,  "Full — BDR"),
        (x - 0.5 * w, small_bdr, COLOR_FULL,  0.55, "Small-scale — BDR"),
        (x + 0.5 * w, full_car,  "#e34948",   1.0,  "Full — CAR"),
        (x + 1.5 * w, small_car, "#e34948",   0.55, "Small-scale — CAR"),
    ]
    for pos, vals, color, alpha, label in bars:
        rects = ax.bar(pos, vals, width=w, color=color, alpha=alpha, label=label,
                        hatch="//" if alpha < 1 else None, edgecolor=color)
        ax.bar_label(rects, fmt="%.2f", fontsize=8, padding=2)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{a}\n(full n={results[a]['n_full']:,} / small n={results[a]['n_small']:,})"
                         for a in airlines])
    for xi, a in zip(x, airlines):
        ax.text(xi, 1.08, f"FID={results[a]['fid']:.4f}", ha="center", fontsize=9,
                fontweight="bold", color="#52514e")
    ax.set_ylabel("Rate (0=0%, 1=100%)")
    ax.set_ylim(0, 1.22)
    ax.set_title("Dataset Quality: Full vs New Small-scale — BDR & CAR", fontweight="bold", pad=44)
    ax.legend(fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.13), frameon=False)
    fig.tight_layout()
    out2 = os.path.join(OUT_DIR, "quality_full_vs_smallscale.png")
    fig.savefig(out2, dpi=150)
    plt.close(fig)
    print("저장:", out2)


if __name__ == "__main__":
    main()
