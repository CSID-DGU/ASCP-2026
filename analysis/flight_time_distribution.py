"""
목표 avg_legs 산정 스크립트 (per-pairing 기준)
  - 시간기준 천장(time-only): 비행시간 분포만으로 pairing 전체에 몇 leg 들어가나 (낙관적 상한)
  - 네트워크 제약(network-feasible): 도착공항=다음출발공항 + 연결시간창 + duty 제약을
    실제로 이어붙여 pairing 전체 leg 수를 추정한 현실적 avg_legs 분포
  - 두 값을 비교해 현실적인 목표 avg_legs(per-pairing)를 제안

필요 컬럼: ORIGIN, DEST, CRS_DEP_TIME, CRS_ARR_TIME, CRS_ELAPSED_TIME
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ─── 설정 ──────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "RL", "data")
files = {"Delta": "delta_2019_01.csv",
         "Alaska": "alaska_2019_01.csv",
         "JetBlue": "jetblue_2019_01.csv"}

MAX_DUTY_MIN = 13 * 60     # 최대 duty (분)
MIN_CONN_MIN = 30          # 최소 연결시간 (분)
MAX_CONN_MIN = 180         # 최대 연결시간(=최대 sit). 길수록 leg가 많아짐. 조정 포인트.
N_SIMS_NET   = 6000        # 네트워크 시뮬레이션 pairing 수
N_SIMS_TIME  = 8000
TARGET_OLD   = 3.0         # 기존 목표(비교용)
RNG = np.random.default_rng(0)

# 항공사별 max_duty_periods (overnight rest 횟수)
# Delta CBA §12: 2-day trip → 2 overnights=3 duties, Alaska CBA 동일, JetBlue CBA: 3 overnights=4 duties
AIRLINE_PARAMS = {
    "Delta":   {"max_duty_periods": 2},   # overnight 2회 → duty 3개
    "Alaska":  {"max_duty_periods": 2},
    "JetBlue": {"max_duty_periods": 3},   # overnight 3회 → duty 4개
}

# ─── 시간 파싱 (hhmm → 자정 기준 분) ─────────────────────────────────────────
def hhmm_to_min(arr):
    arr = pd.to_numeric(pd.Series(arr), errors="coerce")
    h = (arr // 100) % 24
    m = arr % 100
    return (h * 60 + m).values

# ─── 시뮬레이션 ────────────────────────────────────────────────────────────────
def sim_time_only(elapsed_hours, max_duty_periods, n_sims=N_SIMS_TIME, max_legs=30):
    """비행시간 분포만으로 채우는 낙관적 천장 (per-pairing).

    duty가 독립적으로 채워진다고 가정 → n_duties개 duty leg 합산.
    """
    n_duties = max_duty_periods + 1
    fh = np.asarray(elapsed_hours)
    max_duty_h = MAX_DUTY_MIN / 60.0
    min_conn_h = MIN_CONN_MIN / 60.0
    total = np.zeros(n_sims, dtype=int)
    for _ in range(n_duties):
        samp = RNG.choice(fh, size=(n_sims, max_legs))
        cost = samp.copy(); cost[:, 1:] += min_conn_h
        legs = (np.cumsum(cost, axis=1) <= max_duty_h).sum(axis=1)
        total += np.maximum(legs, 1)
    return total

def sim_network(df, max_duty_periods, n_sims=N_SIMS_NET):
    """연결공항+시각 제약으로 pairing 전체 leg 수를 시뮬레이션 (per-pairing).

    duty 종료 후 overnight: 같은 공항에서 다음 날 자정 기준으로 duty 재시작.
    """
    d = df.dropna(subset=["ORIGIN", "DEST", "CRS_DEP_TIME",
                          "CRS_ARR_TIME", "CRS_ELAPSED_TIME"]).copy()
    d["dep"] = hhmm_to_min(d["CRS_DEP_TIME"])
    d["arr"] = hhmm_to_min(d["CRS_ARR_TIME"])
    d["el"]  = d["CRS_ELAPSED_TIME"].astype(float)
    d = d.dropna(subset=["dep", "arr", "el"])

    # 출발공항별 인덱스 (dep 정렬)
    idx = {}
    for ap, g in d.groupby("ORIGIN"):
        g = g.sort_values("dep")
        idx[ap] = {"dep": g["dep"].values, "arr": g["arr"].values,
                   "el": g["el"].values, "dest": g["DEST"].values}

    arr_all = d["arr"].values; el_all = d["el"].values; dest_all = d["DEST"].values
    N = len(d)
    n_duties = max_duty_periods + 1
    legs_out = np.empty(n_sims, int)

    for s in range(n_sims):
        # 첫 번째 duty: 랜덤 비행편에서 시작
        i = RNG.integers(N)
        airport = dest_all[i]
        clock = arr_all[i]
        duty_elapsed = el_all[i]
        total_legs = 1

        for d_idx in range(n_duties):
            if d_idx > 0:
                # overnight 후 새 duty: 같은 공항, clock 자정 리셋
                duty_elapsed = 0
                clock = 0

            # 현재 duty 내 연결 가능한 leg 추가
            while True:
                nb = idx.get(airport)
                if nb is None:
                    break
                lo = np.searchsorted(nb["dep"], clock + MIN_CONN_MIN, "left")
                hi = np.searchsorted(nb["dep"], clock + MAX_CONN_MIN, "right")
                if hi <= lo:
                    break
                gaps = nb["dep"][lo:hi] - clock
                new_elapsed = duty_elapsed + gaps + nb["el"][lo:hi]
                feas = np.where(new_elapsed <= MAX_DUTY_MIN)[0]
                if feas.size == 0:
                    break
                pick = lo + feas[RNG.integers(feas.size)]
                duty_elapsed += (nb["dep"][pick] - clock) + nb["el"][pick]
                clock = nb["arr"][pick]
                airport = nb["dest"][pick]
                total_legs += 1

        legs_out[s] = total_legs
    return legs_out

# ─── 실행 ──────────────────────────────────────────────────────────────────────
dfs, stats = {}, {}
for airline, fname in files.items():
    df = pd.read_csv(os.path.join(DATA_DIR, fname))
    df.columns = df.columns.str.strip()
    df["flight_hours"] = df["CRS_ELAPSED_TIME"] / 60.0
    dfs[airline] = df

    mdp = AIRLINE_PARAMS[airline]["max_duty_periods"]
    t = sim_time_only(df["flight_hours"].dropna().values, max_duty_periods=mdp)
    n = sim_network(df, max_duty_periods=mdp)
    stats[airline] = {"t_mean": t.mean(), "t_std": t.std(),
                      "n_mean": n.mean(), "n_std": n.std(),
                      "avg_ft": df["flight_hours"].mean(),
                      "max_duty_periods": mdp}

print(f"{'airline':>8} {'mdp':>4} {'avg_ft(h)':>9} {'time-only(천장)':>18} {'network(현실)':>16}")
print("-" * 65)
for a, st in stats.items():
    print(f"{a:>8} {st['max_duty_periods']:>4} {st['avg_ft']:>9.2f} "
          f"{st['t_mean']:>9.2f} ± {st['t_std']:.2f}   "
          f"{st['n_mean']:>7.2f} ± {st['n_std']:.2f}")

# ─── 목표 제안 ─────────────────────────────────────────────────────────────────
print("\n=== 목표 avg_legs 제안 ===")
common = min(st["n_mean"] for st in stats.values())
print(f"항공사 공통(보수적) 목표 ≈ {np.floor(common*2)/2:.1f}  "
      f"(가장 빡빡한 항공사 현실 평균 {common:.2f} 이하)")
print("항공사별 목표(현실 평균 기준):")
for a, st in stats.items():
    print(f"  {a:>8}: {st['n_mean']:.2f}  → 권장 target {np.floor(st['n_mean']*2)/2:.1f}")

# ─── 시각화 ────────────────────────────────────────────────────────────────────
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": "#9aa0b0",
                     "axes.linewidth": 0.8, "figure.facecolor": "white"})
PALETTE  = {"Delta": "#3b3f6b", "Alaska": "#5e7e96", "JetBlue": "#9a8aa8"}
PANEL_BG = "#f3f3f8"

fig = plt.figure(figsize=(14.5, 8.8))
gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.2], hspace=0.42, wspace=0.22)

# 상단: 비행시간 분포
bins = np.arange(0, 15, 1)
for col, (airline, df) in enumerate(dfs.items()):
    ax = fig.add_subplot(gs[0, col]); ax.set_facecolor(PANEL_BG)
    ft = df["flight_hours"].dropna()
    counts, edges = np.histogram(ft, bins=bins)
    ax.bar(edges[:-1] + 0.5, counts, width=0.86, color=PALETTE[airline],
           alpha=0.9, edgecolor="white", linewidth=0.6, zorder=3)
    ax.axvline(ft.mean(), color="#e4572e", ls="--", lw=1.6, label=f"mean = {ft.mean():.2f}h")
    ax.axvline(ft.median(), color="#f2a541", ls=":", lw=1.8, label=f"median = {ft.median():.2f}h")
    ax.set_title(airline, fontweight="bold", color=PALETTE[airline], pad=8)
    ax.set_xlabel("Flight Duration (hours)", fontsize=9.5)
    if col == 0: ax.set_ylabel("Number of Flights", fontsize=9.5)
    ax.set_xticks(range(0, 15, 2)); ax.set_xlim(0, 14)
    ax.grid(axis="y", color="white", lw=1.1, zorder=0); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.legend(fontsize=8, framealpha=0.9, loc="upper right")

# 하단: 천장 vs 현실 (그룹 막대 + I 에러바)
axb = fig.add_subplot(gs[1, :]); axb.set_facecolor(PANEL_BG)
airlines = list(stats.keys()); x = np.arange(len(airlines)); w = 0.34
t_means = [stats[a]["t_mean"] for a in airlines]; t_stds = [stats[a]["t_std"] for a in airlines]
n_means = [stats[a]["n_mean"] for a in airlines]; n_stds = [stats[a]["n_std"] for a in airlines]
cols = [PALETTE[a] for a in airlines]

axb.bar(x - w/2, t_means, w, color=cols, alpha=0.45, edgecolor="white",
        linewidth=1.0, hatch="//", zorder=3, label="Time-only ceiling")
axb.bar(x + w/2, n_means, w, color=cols, alpha=0.95, edgecolor="white",
        linewidth=1.0, zorder=3, label="Network-feasible (realistic)")
axb.errorbar(x - w/2, t_means, yerr=t_stds, fmt="none", ecolor="#22252e",
             elinewidth=1.3, capsize=6, capthick=1.3, zorder=5)
axb.errorbar(x + w/2, n_means, yerr=n_stds, fmt="none", ecolor="#22252e",
             elinewidth=1.3, capsize=6, capthick=1.3, zorder=5)
axb.axhline(TARGET_OLD, color="#e4572e", ls="--", lw=1.6, alpha=0.85, zorder=2,
            label=f"old target = {TARGET_OLD:.1f}")

for xi, m, s in zip(x - w/2, t_means, t_stds):
    axb.text(xi, m + s + 0.08, f"{m:.2f}", ha="center", va="bottom",
             fontsize=10, color="#555", fontweight="bold")
for xi, m, s in zip(x + w/2, n_means, n_stds):
    axb.text(xi, m + s + 0.08, f"{m:.2f}±{s:.2f}", ha="center", va="bottom",
             fontsize=10.5, color="#22252e", fontweight="bold")

axb.set_xticks(x); axb.set_xticklabels(airlines, fontsize=12, fontweight="bold")
for tick, a in zip(axb.get_xticklabels(), airlines): tick.set_color(PALETTE[a])
axb.set_ylabel("avg_legs  (legs per pairing)", fontsize=11)
axb.set_ylim(0, max(t_means) + max(t_stds) + 1.0)
axb.grid(axis="y", color="white", lw=1.1, zorder=0); axb.set_axisbelow(True)
for sp in ("top", "right"): axb.spines[sp].set_visible(False)
axb.legend(fontsize=9.5, framealpha=0.9, loc="upper right", ncol=1)
axb.set_title("avg_legs per Pairing: Time-only Ceiling vs Network-feasible  (mean ± std)",
              fontweight="bold", pad=10)

fig.suptitle("Target avg_legs Analysis  (2019-01, per-pairing)", fontsize=15, fontweight="bold", y=0.98)
fig.text(0.5, 0.005,
         f"Network sim: connect DEST→ORIGIN, conn∈[{MIN_CONN_MIN},{MAX_CONN_MIN}]min, "
         f"duty≤{MAX_DUTY_MIN//60}h, overnight→same airport clock reset, {N_SIMS_NET} pairings/airline  "
         f"| max_duty_periods: Delta=2, Alaska=2, JetBlue=3",
         ha="center", fontsize=9, style="italic", color="#6b6f7d")

out_path = os.path.join(os.path.dirname(__file__), "target_avg_legs.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
print(f"\n그래프 저장: {out_path}")
plt.show()