"""
dataset_analysis_chain.py — flight_time_distribution.py의 3패널 "Flight Dataset Analysis"
그림을 그대로 재현하되, small-scale 소스만 옛날 균등랜덤 SAMPLE_FILES 대신 새로 만든
chain-sampling 결과(RL/data/small-scale/*_sample_chain.csv, raw BTS 스키마)로 교체.

중단 패널(avg_legs 천장 vs network-feasible vs RL current)은 full 데이터만 쓰므로 원본과
동일 — 상단(히스토그램)·하단(BDR/CAR) 두 패널만 small-scale 소스가 chain으로 바뀐다.

출력: RL/data/small-scale/dataset_avg_chain.png
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from flight_time_distribution import (
    DATA_DIR, FILES, AIRLINE_CONFIG, N_SIMS, TARGET_OLD,
    load_flights, compute_quality, sim_network, sim_time_only, frechet_distance_1d,
)

OUT_DIR = os.path.join(DATA_DIR, "small-scale")
CHAIN_FILES = {
    "Delta":   "delta_2019_01_sample_chain.csv",
    "Alaska":  "alaska_2019_01_sample_chain.csv",
    "JetBlue": "jetblue_2019_01_sample_chain.csv",
}

RNG = np.random.default_rng(0)
PALETTE  = {"Delta": "#3b3f6b", "Alaska": "#5e7e96", "JetBlue": "#9a8aa8"}
PANEL_BG = "#f3f3f8"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.edgecolor": "#9aa0b0", "axes.linewidth": 0.8,
    "figure.facecolor": "white",
})


def main():
    dfs, stats = {}, {}
    for airline, fname in FILES.items():
        flights = load_flights(os.path.join(DATA_DIR, fname))
        cfg = AIRLINE_CONFIG[airline]
        t = sim_time_only(flights["dur"].values, cfg)
        n = sim_network(flights, cfg)
        stats[airline] = dict(
            t_mean=t.mean(), t_std=t.std(), n_mean=n.mean(), n_std=n.std(),
            avg_ft=flights["dur"].mean() / 60, mdp=cfg["max_duty_periods"],
        )
        dfs[airline] = flights
        print(f"{airline:8s} full n={len(flights):6d}  time-only={t.mean():.2f}±{t.std():.2f}  "
              f"network={n.mean():.2f}±{n.std():.2f}")

    chain_hours, chain_qdf, quality = {}, {}, {}
    for airline, fname in CHAIN_FILES.items():
        qdf = load_flights(os.path.join(OUT_DIR, fname))
        chain_hours[airline] = qdf["dur"] / 60.0
        chain_qdf[airline] = qdf
        cfg = AIRLINE_CONFIG[airline]
        full_bdr, full_car = compute_quality(dfs[airline], cfg)
        c_bdr, c_car = compute_quality(qdf, cfg)
        fid = frechet_distance_1d(dfs[airline]["dur"].values / 60.0, chain_hours[airline].values)
        quality[airline] = dict(full_bdr=full_bdr, full_car=full_car,
                                 c_bdr=c_bdr, c_car=c_car, fid=fid,
                                 n_full=len(dfs[airline]), n_small=len(qdf))
        print(f"{airline:8s} chain  BDR={c_bdr:.3f} CAR={c_car:.3f}  FID={fid:.4f}")

    fig = plt.figure(figsize=(14.5, 13.5))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.1, 0.9], hspace=0.50, wspace=0.22)

    # ── 상단: 비행시간 분포 (full vs chain small-scale) ─────────────────────
    bins = np.arange(0, 15, 1)
    for col, airline in enumerate(FILES):
        ax = fig.add_subplot(gs[0, col])
        ax.set_facecolor(PANEL_BG)
        ft = dfs[airline]["dur"] / 60.0

        counts, edges = np.histogram(ft, bins=bins)
        ax.bar(edges[:-1] + 0.5, counts, 0.86, color=PALETTE[airline], alpha=0.9,
               edgecolor="white", lw=0.6, zorder=3, label=f"Full (n={len(ft):,})")

        ft_s = chain_hours[airline]
        s_counts, _ = np.histogram(ft_s, bins=bins)
        scale = len(ft) / len(ft_s)
        ax.bar(edges[:-1] + 0.5, s_counts * scale, 0.86, color="white", alpha=0.60,
               edgecolor=PALETTE[airline], lw=1.8, hatch="///", zorder=4,
               label=f"Chain small-scale (n={len(ft_s):,}, x{scale:.0f})")

        ax.axvline(ft.mean(), color="#e4572e", ls="--", lw=1.6, label=f"mean = {ft.mean():.2f}h")
        ax.axvline(ft.median(), color="#f2a541", ls=":", lw=1.8, label=f"median = {ft.median():.2f}h")

        ax.set_title(f"{airline}  (FID={quality[airline]['fid']:.4f})",
                     fontweight="bold", color=PALETTE[airline], pad=8)
        ax.set_xlabel("Flight Duration (hours)", fontsize=9.5)
        if col == 0:
            ax.set_ylabel("Number of Flights", fontsize=9.5)
        ax.set_xticks(range(0, 15, 2))
        ax.set_xlim(0, 14)
        ax.grid(axis="y", color="white", lw=1.1, zorder=0)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.legend(fontsize=7.5, framealpha=0.9, loc="upper right", ncol=1)

    # ── 중단: avg_legs 천장 vs network-feasible vs RL current (full 데이터만, 원본과 동일) ──
    axb = fig.add_subplot(gs[1, :])
    axb.set_facecolor(PANEL_BG)
    airlines = list(stats.keys()); x = np.arange(len(airlines)); w = 0.30
    t_means = [stats[a]["t_mean"] for a in airlines]
    t_stds  = [stats[a]["t_std"] for a in airlines]
    n_means = [stats[a]["n_mean"] for a in airlines]
    n_stds  = [stats[a]["n_std"] for a in airlines]
    cols = [PALETTE[a] for a in airlines]

    axb.bar(x - w, t_means, w, color=cols, alpha=0.40, edgecolor="white", lw=1.0,
            hatch="//", zorder=3, label="Time-only ceiling")
    axb.bar(x, n_means, w, color=cols, alpha=0.95, edgecolor="white", lw=1.0,
            zorder=3, label="Network-feasible (realistic)")
    rl_current = 1.85
    axb.bar(x + w, [rl_current] * 3, w, color="#aaaaaa", alpha=0.85, edgecolor="white",
            lw=1.0, zorder=3, label=f"RL current (~{rl_current})")
    axb.errorbar(x - w, t_means, yerr=t_stds, fmt="none", ecolor="#22252e",
                 elinewidth=1.3, capsize=6, capthick=1.3, zorder=5)
    axb.errorbar(x, n_means, yerr=n_stds, fmt="none", ecolor="#22252e",
                 elinewidth=1.3, capsize=6, capthick=1.3, zorder=5)
    axb.axhline(TARGET_OLD, color="#e4572e", ls="--", lw=1.5, alpha=0.8, zorder=2,
                label=f"MIN_LEGS_FOR_PAIRING = {int(TARGET_OLD)}")
    axb.axhline(5.72, color="#2ca02c", ls=":", lw=1.5, alpha=0.8, zorder=2, label="LLM v0 = 5.72")
    for xi, m, s in zip(x - w, t_means, t_stds):
        axb.text(xi, m + s + 0.15, f"{m:.1f}", ha="center", va="bottom",
                  fontsize=9, color="#666", fontweight="bold")
    for xi, m, s in zip(x, n_means, n_stds):
        axb.text(xi, m + s + 0.15, f"{m:.2f}±{s:.2f}", ha="center", va="bottom",
                  fontsize=10, color="#22252e", fontweight="bold")
    axb.set_xticks(x)
    axb.set_xticklabels([f"{a}\n(overnight≤{stats[a]['mdp']})" for a in airlines],
                         fontsize=11, fontweight="bold")
    for tick, a in zip(axb.get_xticklabels(), airlines):
        tick.set_color(PALETTE[a])
    axb.set_ylabel("avg_legs  (per pairing)", fontsize=11)
    axb.set_ylim(0, max(t_means) + max(t_stds) + 2.0)
    axb.grid(axis="y", color="white", lw=1.1, zorder=0)
    axb.set_axisbelow(True)
    for sp in ("top", "right"):
        axb.spines[sp].set_visible(False)
    axb.legend(fontsize=9, framealpha=0.9, loc="upper right", ncol=2)
    axb.set_title("avg_legs per Pairing: Ceiling vs Network-feasible vs RL current  (mean ± std)",
                   fontweight="bold", pad=10)

    # ── 하단: 데이터셋 품질 (Full vs Chain small-scale) ─────────────────────
    ax_q = fig.add_subplot(gs[2, :])
    ax_q.set_facecolor(PANEL_BG)
    x_q = np.arange(len(airlines)); w_q = 0.18
    BDR_COLOR, CAR_COLOR = "#4472C4", "#ED7D31"
    full_bdr_vals = [quality[a]["full_bdr"] for a in airlines]
    small_bdr_vals = [quality[a]["c_bdr"] for a in airlines]
    full_car_vals = [quality[a]["full_car"] for a in airlines]
    small_car_vals = [quality[a]["c_car"] for a in airlines]

    bar_groups = [
        ax_q.bar(x_q - 1.5 * w_q, full_bdr_vals, w_q, color=BDR_COLOR, alpha=0.85,
                 edgecolor="white", lw=0.8, zorder=3, label="Full — Base Dep. Ratio (BDR)"),
        ax_q.bar(x_q - 0.5 * w_q, small_bdr_vals, w_q, color=BDR_COLOR, alpha=0.38,
                 edgecolor=BDR_COLOR, lw=1.5, hatch="///", zorder=3, label="Chain small-scale — BDR"),
        ax_q.bar(x_q + 0.5 * w_q, full_car_vals, w_q, color=CAR_COLOR, alpha=0.85,
                 edgecolor="white", lw=0.8, zorder=3, label="Full — Connection Avail. Rate (CAR)"),
        ax_q.bar(x_q + 1.5 * w_q, small_car_vals, w_q, color=CAR_COLOR, alpha=0.38,
                 edgecolor=CAR_COLOR, lw=1.5, hatch="///", zorder=3, label="Chain small-scale — CAR"),
    ]
    for bar_group in bar_groups:
        for bar in bar_group:
            h = bar.get_height()
            ax_q.text(bar.get_x() + bar.get_width() / 2, h + 0.008, f"{h:.2f}",
                      ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#22252e")

    ax_q.set_xticks(x_q)
    ax_q.set_xticklabels(
        [f"{a}\n(full n={quality[a]['n_full']:,} / small n={quality[a]['n_small']:,})" for a in airlines],
        fontsize=10, fontweight="bold")
    for tick, a in zip(ax_q.get_xticklabels(), airlines):
        tick.set_color(PALETTE[a])
    ax_q.set_ylabel("Rate  (0 = 0%, 1 = 100%)", fontsize=11)
    ax_q.set_ylim(0, 1.18)
    ax_q.grid(axis="y", color="white", lw=1.1, zorder=0)
    ax_q.set_axisbelow(True)
    for sp in ("top", "right"):
        ax_q.spines[sp].set_visible(False)
    ax_q.legend(fontsize=9, framealpha=0.9, loc="upper right", ncol=2)
    ax_q.set_title("Dataset Quality: Full vs Chain Small-scale  —  BDR & CAR", fontweight="bold", pad=10)
    ax_q.text(
        0.5, -0.22,
        "BDR: fraction of flights departing from hub bases  |  "
        "CAR: fraction of flights with >=1 feasible next connection  [min_conn, max_conn]\n"
        "Chain small-scale (connectivity-preserving sampling) bars (hatched) vs full bars",
        ha="center", va="top", transform=ax_q.transAxes,
        fontsize=8, style="italic", color="#6b6f7d")

    fig.suptitle("Flight Dataset Analysis — Full vs Chain Small-scale  (2019-01, per-pairing, RL-aligned)",
                 fontsize=15, fontweight="bold", y=0.995)
    cfg_d = AIRLINE_CONFIG["Delta"]
    fig.text(
        0.5, 0.002,
        f"Network sim: base-start forced, conn in [{cfg_d['min_conn']},{cfg_d['max_conn']}]min, "
        f"duty<={cfg_d['max_duty']//60}h, max_legs={cfg_d['max_legs']}/duty, "
        f"min_rest={cfg_d['min_rest']}min, max_pairing_days={cfg_d['max_pairing_days']}, "
        f"{N_SIMS} sims/airline  |  base-return: allowed to fail (penalty only)",
        ha="center", fontsize=8, style="italic", color="#6b6f7d")

    plt.tight_layout(rect=[0, 0.015, 1, 0.993])
    out_path = os.path.join(OUT_DIR, "dataset_avg_chain.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    print("저장:", out_path)


if __name__ == "__main__":
    main()
