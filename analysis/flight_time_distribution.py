"""
  - 절대시각(abs_min) 사용 — eval_llm_final.py의 load_flights와 동일
  - base 출발 강제 / 미복귀 허용(base_penalty만 존재, invalid 아님) — RL과 동일
  - min_conn=39min, max_conn=540min, max_legs=8(duty당), min_rest=600min
  - max_pairing_days=5 종료 조건 추가
  - CSV 전체 기간(멀티데이) 사용
  - 항공사별 base 목록 분리

필요 컬럼: ORIGIN, DEST, FL_DATE, CRS_DEP_TIME, CRS_ARR_TIME, CRS_ELAPSED_TIME
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ─── 경로 ──────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "RL", "data")
FILES = {
    "Delta":   "delta_2019_01.csv",
    "Alaska":  "alaska_2019_01.csv",
    "JetBlue": "jetblue_2019_01.csv",
}

# ─── 항공사별 제약 (config.py DEFAULT_CONSTRAINTS + AIRLINE_BASES) ─────────────
AIRLINE_CONFIG = {
    "Delta": {
        "min_conn":         39,    # 분 (0.65h)
        "max_conn":         540,   # 분 (9.0h)
        "max_duty":         780,   # 분 (13.0h)
        "max_legs":         8,     # duty당
        "min_rest":         600,   # 분 (10.0h)
        "max_duty_periods": 2,     # overnight 횟수 → duty 최대 3개
        "max_pairing_days": 5,
        "bases": {"ATL", "DTW", "MSP", "JFK", "LAX", "SEA", "SLC"},
    },
    "Alaska": {
        "min_conn":         39,
        "max_conn":         540,
        "max_duty":         780,
        "max_legs":         8,
        "min_rest":         600,
        "max_duty_periods": 2,
        "max_pairing_days": 5,
        "bases": {"SEA", "PDX", "ANC", "LAX", "SFO"},
    },
    "JetBlue": {
        "min_conn":         39,
        "max_conn":         540,
        "max_duty":         780,
        "max_legs":         8,
        "min_rest":         600,
        "max_duty_periods": 3,     # overnight 3회 → duty 최대 4개
        "max_pairing_days": 5,
        "bases": {"JFK", "BOS", "FLL", "LAX", "MCO"},
    },
}

N_SIMS   = 8000
RNG      = np.random.default_rng(0)
TARGET_OLD = 3.0

# ─── UTC 오프셋 (eval_llm_final.py와 동일) ────────────────────────────────────
_UTC = {
    **{ap: -600 for ap in ['HNL','KOA','LIH','OGG']},
    **{ap: -540 for ap in ['ANC','FAI']},
    **{ap: -480 for ap in ['GEG','LAS','LAX','OAK','ONT','PDX','PSP','RNO','SAN','SEA',
                            'SFO','SJC','SMF','SNA']},
    **{ap: -420 for ap in ['ABQ','BIL','BOI','BZN','COS','DEN','EGE','ELP','FCA','HDN',
                            'JAC','MSO','MTJ','PHX','SLC','TUS']},
    **{ap: -360 for ap in ['ATW','AUS','BHM','BIS','BNA','BTR','CID','DAL','DFW','DSM',
                            'ECP','FAR','FSD','GPT','GRB','HOU','HSV','IAH','ICT','JAN',
                            'LFT','LIT','MCI','MDW','MEM','MKE','MOB','MSN','MSP','MSY',
                            'OKC','OMA','ORD','PNS','SAT','STL','TUL','VPS','XNA']},
    **{ap: -300 for ap in ['ABE','AGS','ALB','ATL','AVL','AVP','BDL','BOS','BUF','BWI',
                            'CAE','CAK','CHA','CHO','CHS','CLE','CLT','CMH','CRW','CVG',
                            'DAB','DAY','DCA','DTW','EWR','EYW','FAY','FLL','FNT','GNV',
                            'GRR','GSO','GSP','HPN','IAD','ILM','IND','JAX','JFK','LEX',
                            'LGA','MCO','MDT','MHT','MIA','MLB','MYR','ORF','PBI','PHF',
                            'PHL','PIT','PVD','PWM','RDU','RIC','ROA','ROC','RSW','SAV',
                            'SDF','SRQ','SYR','TLH','TPA','TRI','TYS']},
    **{ap: -240 for ap in ['SJU','STT','STX']},
}
def _offset(ap): return _UTC.get(ap, -300)

# ─── 데이터 로드 (eval_llm_final.py load_flights와 동일한 절대시각) ────────────
def load_flights(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=["ORIGIN","DEST","CRS_DEP_TIME","CRS_ARR_TIME",
                            "CRS_ELAPSED_TIME","FL_DATE"])
    df["FL_DATE"] = pd.to_datetime(df["FL_DATE"], format="mixed")
    base_date = df["FL_DATE"].min()

    rows = []
    for _, r in df.iterrows():
        origin, dest = r["ORIGIN"], r["DEST"]
        day_off = (r["FL_DATE"] - base_date).days
        dep_local = int(r["CRS_DEP_TIME"])
        dep_local_min = (dep_local // 100) * 60 + (dep_local % 100) + day_off * 1440
        dep_abs = dep_local_min - _offset(origin)
        dur = int(round(float(r["CRS_ELAPSED_TIME"])))
        arr_abs = dep_abs + dur
        rows.append({
            "origin": origin, "dest": dest,
            "dep_abs": dep_abs, "arr_abs": arr_abs, "dur": dur,
        })

    flights = pd.DataFrame(rows)
    return flights

# ─── 공항별 출발 인덱스 구축 ──────────────────────────────────────────────────
def build_index(flights):
    idx = {}
    for ap, g in flights.groupby("origin"):
        g = g.sort_values("dep_abs")
        idx[ap] = {
            "dep":  g["dep_abs"].values,
            "arr":  g["arr_abs"].values,
            "dur":  g["dur"].values,
            "dest": g["dest"].values,
        }
    return idx

# ─── 네트워크 시뮬레이션 (per-pairing, RL 정합) ───────────────────────────────
def sim_network(flights, cfg, n_sims=N_SIMS):
    """
    RL rollout 구조를 따라 pairing을 시뮬레이션:
      - base에서 출발 강제
      - duty 내: min_conn ~ max_conn, max_legs, max_duty 제약
      - overnight: min_rest 후 같은 공항에서 연속 절대시각으로 다음 duty
      - 종료: max_duty_periods 소진 or max_pairing_days 초과 or 연결 불가
      - 미복귀 허용 (RL과 동일 — base_penalty만, invalid 아님)
    """
    min_conn  = cfg["min_conn"]
    max_conn  = cfg["max_conn"]
    max_duty  = cfg["max_duty"]
    max_legs  = cfg["max_legs"]
    min_rest  = cfg["min_rest"]
    max_overs = cfg["max_duty_periods"]   # overnight 최대 횟수
    max_days  = cfg["max_pairing_days"]
    bases     = cfg["bases"]

    idx = build_index(flights)

    # base 출발 비행 목록
    base_flights = flights[flights["origin"].isin(bases)].reset_index(drop=True)
    if len(base_flights) == 0:
        print("[경고] base 출발 비행 없음")
        return np.ones(n_sims, int)

    bf_dep  = base_flights["dep_abs"].values
    bf_arr  = base_flights["arr_abs"].values
    bf_dest = base_flights["dest"].values
    BN = len(base_flights)

    legs_out = np.empty(n_sims, int)

    for s in range(n_sims):
        # ── pairing 시작: base 출발 비행 무작위 선택 ──────────────────────────
        i = RNG.integers(BN)
        pairing_start = bf_dep[i]
        airport       = bf_dest[i]
        clock         = bf_arr[i]          # 절대 분
        duty_start    = bf_dep[i]
        duty_elapsed  = bf_arr[i] - bf_dep[i]   # = dur
        duty_legs     = 1
        total_legs    = 1
        overnights    = 0

        while True:
            # ── duty 내 다음 leg 탐색 ─────────────────────────────────────────
            nb = idx.get(airport)
            extended = False

            if nb is not None and duty_legs < max_legs:
                dep = nb["dep"]
                lo = np.searchsorted(dep, clock + min_conn, "left")
                hi = np.searchsorted(dep, clock + max_conn, "right")

                if lo < hi:
                    gaps      = dep[lo:hi] - clock
                    new_elap  = duty_elapsed + gaps + nb["dur"][lo:hi]
                    feas      = np.where(new_elap <= max_duty)[0]

                    if feas.size > 0:
                        pick = lo + feas[RNG.integers(feas.size)]
                        duty_elapsed += (dep[pick] - clock) + nb["dur"][pick]
                        clock         = nb["arr"][pick]
                        airport       = nb["dest"][pick]
                        duty_legs    += 1
                        total_legs   += 1
                        extended      = True

            if extended:
                continue   # 같은 duty 내 계속 탐색

            # ── duty 내 연결 불가 → overnight 시도 ───────────────────────────
            if overnights >= max_overs:
                break   # overnight 소진 → pairing 종료

            # overnight 후 새 duty 시작: 같은 공항, clock += min_rest
            new_duty_start = clock + min_rest

            # max_pairing_days 체크
            if (new_duty_start - pairing_start) / 1440.0 > max_days:
                break

            # 새 duty에서 연결 가능한 비행이 있는지 확인
            nb2 = idx.get(airport)
            if nb2 is None:
                break
            lo2 = np.searchsorted(nb2["dep"], new_duty_start + min_conn, "left")
            hi2 = np.searchsorted(nb2["dep"], new_duty_start + max_conn, "right")
            if lo2 >= hi2:
                break   # overnight 해도 연결 불가 → pairing 종료

            # overnight 확정 → 새 duty
            overnights   += 1
            clock         = new_duty_start
            duty_elapsed  = 0
            duty_legs     = 0

        legs_out[s] = total_legs

    return legs_out

# ─── 시간기준 천장 (time-only, per-pairing) ───────────────────────────────────
def sim_time_only(elapsed_min, cfg, n_sims=N_SIMS, max_sample=30):
    """비행시간 분포만으로 duty를 채우고 n_duties개 합산 (낙관적 상한)."""
    n_duties  = cfg["max_duty_periods"] + 1
    max_duty  = cfg["max_duty"]
    min_conn  = cfg["min_conn"]
    fh = np.asarray(elapsed_min, dtype=float)
    total = np.zeros(n_sims, int)
    for _ in range(n_duties):
        samp = RNG.choice(fh, size=(n_sims, max_sample))
        cost = samp.copy(); cost[:, 1:] += min_conn
        legs = (np.cumsum(cost, axis=1) <= max_duty).sum(axis=1)
        total += np.maximum(legs, 1)
    return total

# ─── 메인 ──────────────────────────────────────────────────────────────────────
dfs, stats = {}, {}

for airline, fname in FILES.items():
    path = os.path.join(DATA_DIR, fname)
    print(f"\n[{airline}] 로드 중: {path}")
    flights = load_flights(path)
    cfg = AIRLINE_CONFIG[airline]

    print(f"  총 비행편: {len(flights)}, "
          f"base 출발: {(flights['origin'].isin(cfg['bases'])).sum()}")

    t = sim_time_only(flights["dur"].values, cfg)
    n = sim_network(flights, cfg)

    stats[airline] = {
        "t_mean": t.mean(), "t_std": t.std(),
        "n_mean": n.mean(), "n_std": n.std(),
        "avg_ft": flights["dur"].mean() / 60,
        "mdp":    cfg["max_duty_periods"],
    }
    print(f"  time-only : {t.mean():.2f} ± {t.std():.2f}")
    print(f"  network   : {n.mean():.2f} ± {n.std():.2f}")
    dfs[airline] = flights

# ─── 목표 제안 출력 ───────────────────────────────────────────────────────────
print("\n" + "="*60)
print("항공사별 현실적 avg_legs (per-pairing, network-feasible)")
print("="*60)
print(f"{'airline':>8} {'mdp':>4} {'avg_ft(h)':>9} "
      f"{'time-only':>12} {'network':>12}")
print("-"*55)
for a, st in stats.items():
    print(f"{a:>8} {st['mdp']:>4} {st['avg_ft']:>9.2f} "
          f"{st['t_mean']:>7.2f}±{st['t_std']:.2f}  "
          f"{st['n_mean']:>7.2f}±{st['n_std']:.2f}")

print("\n=== RL target avg_legs 제안 ===")
print(f"  LLM 비교 기준 (v0)    : 5.72  (per-pairing, 같은 Delta 데이터)")
print(f"  RL 현재               : ~1.8  (per-pairing)")
for a, st in stats.items():
    tgt = round(st["n_mean"] * 0.75, 1)   # 현실 평균의 75%
    print(f"  {a:>8} 현실 평균 {st['n_mean']:.2f} → "
          f"권장 target {tgt}  "
          f"(MIN_LEGS_FOR_PAIRING 상향 참고)")

# ─── 시각화 ───────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.edgecolor": "#9aa0b0", "axes.linewidth": 0.8,
    "figure.facecolor": "white",
})
PALETTE  = {"Delta": "#3b3f6b", "Alaska": "#5e7e96", "JetBlue": "#9a8aa8"}
PANEL_BG = "#f3f3f8"

fig = plt.figure(figsize=(14.5, 8.8))
gs  = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.2], hspace=0.42, wspace=0.22)

# 상단: 비행시간 분포
# 원본 CSV에서 다시 읽어 시간 단위로
raw_dfs = {}
bins = np.arange(0, 15, 1)
for col, (airline, fname) in enumerate(FILES.items()):
    raw = pd.read_csv(os.path.join(DATA_DIR, fname))
    raw.columns = raw.columns.str.strip()
    raw["flight_hours"] = raw["CRS_ELAPSED_TIME"] / 60.0
    raw_dfs[airline] = raw

    ax = fig.add_subplot(gs[0, col]); ax.set_facecolor(PANEL_BG)
    ft = raw["flight_hours"].dropna()
    counts, edges = np.histogram(ft, bins=bins)
    ax.bar(edges[:-1]+0.5, counts, 0.86, color=PALETTE[airline],
           alpha=0.9, edgecolor="white", lw=0.6, zorder=3)
    ax.axvline(ft.mean(),   color="#e4572e", ls="--", lw=1.6,
               label=f"mean = {ft.mean():.2f}h")
    ax.axvline(ft.median(), color="#f2a541", ls=":",  lw=1.8,
               label=f"median = {ft.median():.2f}h")
    ax.set_title(airline, fontweight="bold", color=PALETTE[airline], pad=8)
    ax.set_xlabel("Flight Duration (hours)", fontsize=9.5)
    if col == 0: ax.set_ylabel("Number of Flights", fontsize=9.5)
    ax.set_xticks(range(0, 15, 2)); ax.set_xlim(0, 14)
    ax.grid(axis="y", color="white", lw=1.1, zorder=0); ax.set_axisbelow(True)
    for sp in ("top","right"): ax.spines[sp].set_visible(False)
    ax.legend(fontsize=8, framealpha=0.9, loc="upper right")

# 하단: 천장 vs 현실 + RL 현재값
axb = fig.add_subplot(gs[1, :]); axb.set_facecolor(PANEL_BG)
airlines = list(stats.keys()); x = np.arange(len(airlines)); w = 0.30
t_means = [stats[a]["t_mean"] for a in airlines]
t_stds  = [stats[a]["t_std"]  for a in airlines]
n_means = [stats[a]["n_mean"] for a in airlines]
n_stds  = [stats[a]["n_std"]  for a in airlines]
cols    = [PALETTE[a] for a in airlines]

# 시간 천장 (빗금)
axb.bar(x - w, t_means, w, color=cols, alpha=0.40, edgecolor="white",
        lw=1.0, hatch="//", zorder=3, label="Time-only ceiling")
# 네트워크 현실값
axb.bar(x,     n_means, w, color=cols, alpha=0.95, edgecolor="white",
        lw=1.0, zorder=3, label="Network-feasible (realistic)")
# RL 현재 (점선 수평선 대신 회색 막대)
rl_current = 1.85
axb.bar(x + w, [rl_current]*3, w, color="#aaaaaa", alpha=0.85,
        edgecolor="white", lw=1.0, zorder=3, label=f"RL current (~{rl_current})")

# 에러바
axb.errorbar(x - w, t_means, yerr=t_stds, fmt="none", ecolor="#22252e",
             elinewidth=1.3, capsize=6, capthick=1.3, zorder=5)
axb.errorbar(x,     n_means, yerr=n_stds, fmt="none", ecolor="#22252e",
             elinewidth=1.3, capsize=6, capthick=1.3, zorder=5)

# 기준선
axb.axhline(TARGET_OLD, color="#e4572e", ls="--", lw=1.5, alpha=0.8, zorder=2,
            label=f"MIN_LEGS_FOR_PAIRING = {int(TARGET_OLD)}")
axb.axhline(5.72, color="#2ca02c", ls=":", lw=1.5, alpha=0.8, zorder=2,
            label="LLM v0 = 5.72")

# 값 라벨
for xi, m, s in zip(x - w, t_means, t_stds):
    axb.text(xi, m+s+0.15, f"{m:.1f}", ha="center", va="bottom",
             fontsize=9, color="#666", fontweight="bold")
for xi, m, s in zip(x, n_means, n_stds):
    axb.text(xi, m+s+0.15, f"{m:.2f}±{s:.2f}", ha="center", va="bottom",
             fontsize=10, color="#22252e", fontweight="bold")

axb.set_xticks(x)
axb.set_xticklabels(
    [f"{a}\n(overnight≤{stats[a]['mdp']})" for a in airlines],
    fontsize=11, fontweight="bold"
)
for tick, a in zip(axb.get_xticklabels(), airlines):
    tick.set_color(PALETTE[a])
axb.set_ylabel("avg_legs  (per pairing)", fontsize=11)
axb.set_ylim(0, max(t_means) + max(t_stds) + 2.0)
axb.grid(axis="y", color="white", lw=1.1, zorder=0); axb.set_axisbelow(True)
for sp in ("top","right"): axb.spines[sp].set_visible(False)
axb.legend(fontsize=9, framealpha=0.9, loc="upper right", ncol=2)
axb.set_title(
    "avg_legs per Pairing: Ceiling vs Network-feasible vs RL current  (mean ± std)",
    fontweight="bold", pad=10
)

fig.suptitle("Target avg_legs Analysis  (2019-01, per-pairing, RL-aligned)",
             fontsize=15, fontweight="bold", y=0.98)
cfg_d = AIRLINE_CONFIG["Delta"]
fig.text(
    0.5, 0.004,
    f"Network sim: base-start forced, conn∈[{cfg_d['min_conn']},{cfg_d['max_conn']}]min, "
    f"duty≤{cfg_d['max_duty']//60}h, max_legs={cfg_d['max_legs']}/duty, "
    f"min_rest={cfg_d['min_rest']}min, max_pairing_days={cfg_d['max_pairing_days']}, "
    f"{N_SIMS} sims/airline  |  base-return: allowed to fail (penalty only)",
    ha="center", fontsize=8.5, style="italic", color="#6b6f7d"
)

plt.tight_layout(rect=[0, 0.02, 1, 0.97])
out_path = os.path.join(os.path.dirname(__file__), "target_avg_legs.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
print(f"\n그래프 저장: {out_path}")
plt.show()