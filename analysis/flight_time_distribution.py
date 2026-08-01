"""
  - Uses absolute time (abs_min) -- same as load_flights in eval_llm_final.py
  - Base departure forced / non-return allowed (only a base_penalty applies, not invalid) -- same as RL
  - min_conn=39min, max_conn=540min, max_legs=8 (per duty), min_rest=600min
  - Added max_pairing_days=5 termination condition
  - Uses the full CSV period (multi-day)
  - Base lists split per airline

Required columns: ORIGIN, DEST, FL_DATE, CRS_DEP_TIME, CRS_ARR_TIME, CRS_ELAPSED_TIME
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Paths ------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "RL", "data")
FILES = {
    "Delta":   "delta_2019_01.csv",
    "Alaska":  "alaska_2019_01.csv",
    "JetBlue": "jetblue_2019_01.csv",
}
SAMPLE_FILES = {
    "Delta":   "sample_DL_20190101_20190107.csv",   # processed format (dep_abs/arr_abs)
    "Alaska":  "sample_AL_20190101_20190107.csv",   # raw BTS format
    "JetBlue": "sample_JE_20190101_20190107.csv",   # raw BTS format
}

# --- Per-airline constraints (config.py DEFAULT_CONSTRAINTS + AIRLINE_BASES) ---
AIRLINE_CONFIG = {
    "Delta": {
        "min_conn":         39,    # minutes (0.65h)
        "max_conn":         540,   # minutes (9.0h)
        "max_duty":         780,   # minutes (13.0h)
        "max_legs":         8,     # per duty
        "min_rest":         600,   # minutes (10.0h)
        "max_duty_periods": 2,     # number of overnights -> up to 3 duties
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
        "max_duty_periods": 3,     # 3 overnights -> up to 4 duties
        "max_pairing_days": 5,
        "bases": {"JFK", "BOS", "FLL", "LAX", "MCO"},
    },
}

N_SIMS   = 8000
RNG      = np.random.default_rng(0)
TARGET_OLD = 3.0

# --- UTC offsets (same as eval_llm_final.py) --------------------------------
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

# --- Data loading (same absolute-time scheme as eval_llm_final.py's load_flights) ---
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

# --- small-scale sample loading ----------------------------------------------
def load_sample(airline, fname):
    """
    Returns (flight_hours Series, quality_df with origin/dest/dep_abs/arr_abs)
    sample_DL: processed format (dep_abs/arr_abs directly available)
    sample_AL/JE: raw BTS format — reuse load_flights()
    """
    path = os.path.join(DATA_DIR, fname)
    raw = pd.read_csv(path)
    raw.columns = raw.columns.str.strip()

    if "duration" in raw.columns:   # sample_DL processed format
        fh = raw["duration"] / 60.0
        qdf = raw[["origin", "dest", "dep_abs", "arr_abs"]].copy()
    else:                            # raw BTS format (sample_AL, sample_JE)
        fh = raw["CRS_ELAPSED_TIME"] / 60.0
        qdf = load_flights(path)

    return fh, qdf

# --- Dataset quality metrics --------------------------------------------------
def compute_quality(flights_df, cfg):
    """
    BDR  : Base Departure Ratio  -- fraction of all flights that depart from a base
           Directly reflects the likelihood of starting an RL rollout
    CAR  : Connection Availability Rate -- fraction of flights with >=1 feasible next connection
           An indicator of multi-leg pairing formation potential (more direct than avg_legs)
    """
    bases     = cfg["bases"]
    min_conn  = cfg["min_conn"]
    max_conn  = cfg["max_conn"]

    bdr = float(flights_df["origin"].isin(bases).mean())

    # origin -> sorted dep_abs index
    dep_idx = {}
    for ap, grp in flights_df.groupby("origin"):
        dep_idx[ap] = np.sort(grp["dep_abs"].values)

    conn = 0
    for dest, dest_grp in flights_df.groupby("dest"):
        nb = dep_idx.get(dest)
        if nb is None:
            continue
        arr_vals = dest_grp["arr_abs"].values
        lo = np.searchsorted(nb, arr_vals + min_conn, "left")
        hi = np.searchsorted(nb, arr_vals + max_conn, "right")
        conn += int((lo < hi).sum())

    car = conn / len(flights_df)
    return bdr, car

# --- Build per-airport departure index -----------------------------------------
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

# --- Network simulation (per-pairing, aligned with RL) -------------------------
def sim_network(flights, cfg, n_sims=N_SIMS):
    min_conn  = cfg["min_conn"]
    max_conn  = cfg["max_conn"]
    max_duty  = cfg["max_duty"]
    max_legs  = cfg["max_legs"]
    min_rest  = cfg["min_rest"]
    max_overs = cfg["max_duty_periods"]
    max_days  = cfg["max_pairing_days"]
    bases     = cfg["bases"]

    idx = build_index(flights)

    base_flights = flights[flights["origin"].isin(bases)].reset_index(drop=True)
    if len(base_flights) == 0:
        print("[warning] no base-departing flights")
        return np.ones(n_sims, int)

    bf_dep  = base_flights["dep_abs"].values
    bf_arr  = base_flights["arr_abs"].values
    bf_dest = base_flights["dest"].values
    BN = len(base_flights)

    legs_out = np.empty(n_sims, int)

    for s in range(n_sims):
        i = RNG.integers(BN)
        pairing_start = bf_dep[i]
        airport       = bf_dest[i]
        clock         = bf_arr[i]
        duty_start    = bf_dep[i]
        duty_elapsed  = bf_arr[i] - bf_dep[i]
        duty_legs     = 1
        total_legs    = 1
        overnights    = 0

        while True:
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
                continue

            if overnights >= max_overs:
                break

            new_duty_start = clock + min_rest

            if (new_duty_start - pairing_start) / 1440.0 > max_days:
                break

            nb2 = idx.get(airport)
            if nb2 is None:
                break
            lo2 = np.searchsorted(nb2["dep"], new_duty_start + min_conn, "left")
            hi2 = np.searchsorted(nb2["dep"], new_duty_start + max_conn, "right")
            if lo2 >= hi2:
                break

            overnights   += 1
            clock         = new_duty_start
            duty_elapsed  = 0
            duty_legs     = 0

        legs_out[s] = total_legs

    return legs_out

# --- Time-based ceiling (time-only, per-pairing) --------------------------------
def sim_time_only(elapsed_min, cfg, n_sims=N_SIMS, max_sample=30):
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

# --- Main -----------------------------------------------------------------------
dfs, stats = {}, {}

for airline, fname in FILES.items():
    path = os.path.join(DATA_DIR, fname)
    print(f"\n[{airline}] loading: {path}")
    flights = load_flights(path)
    cfg = AIRLINE_CONFIG[airline]

    print(f"  total flights: {len(flights)}, "
          f"base departures: {(flights['origin'].isin(cfg['bases'])).sum()}")

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

# --- Load sample data + quality metrics ----------------------------------------
print("\n" + "="*60)
print("Sample dataset quality metrics (BDR / CAR)")
print("="*60)

sample_hours = {}
quality      = {}

for airline, fname in SAMPLE_FILES.items():
    fh, qdf = load_sample(airline, fname)
    sample_hours[airline] = fh
    cfg = AIRLINE_CONFIG[airline]

    full_bdr, full_car = compute_quality(dfs[airline],  cfg)
    samp_bdr, samp_car = compute_quality(qdf,           cfg)

    quality[airline] = {
        "full_bdr": full_bdr, "full_car": full_car,
        "samp_bdr": samp_bdr, "samp_car": samp_car,
        "n_full":   len(dfs[airline]),
        "n_samp":   len(qdf),
    }
    print(f"  {airline:>8}: full  BDR={full_bdr:.3f}  CAR={full_car:.3f}  (n={len(dfs[airline]):,})")
    print(f"  {airline:>8}: small BDR={samp_bdr:.3f}  CAR={samp_car:.3f}  (n={len(qdf):,})")

# --- Print target proposal -------------------------------------------------------
print("\n" + "="*60)
print("Realistic avg_legs per airline (per-pairing, network-feasible)")
print("="*60)
print(f"{'airline':>8} {'mdp':>4} {'avg_ft(h)':>9} "
      f"{'time-only':>12} {'network':>12}")
print("-"*55)
for a, st in stats.items():
    print(f"{a:>8} {st['mdp']:>4} {st['avg_ft']:>9.2f} "
          f"{st['t_mean']:>7.2f}±{st['t_std']:.2f}  "
          f"{st['n_mean']:>7.2f}±{st['n_std']:.2f}")

print("\n=== RL target avg_legs proposal ===")
print(f"  LLM comparison baseline (v0)    : 5.72  (per-pairing, same Delta data)")
print(f"  RL current               : ~1.8  (per-pairing)")
for a, st in stats.items():
    tgt = round(st["n_mean"] * 0.75, 1)
    print(f"  {a:>8} realistic mean {st['n_mean']:.2f} -> "
          f"recommended target {tgt}  "
          f"(reference for raising MIN_LEGS_FOR_PAIRING)")

# --- Visualization -----------------------------------------------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.edgecolor": "#9aa0b0", "axes.linewidth": 0.8,
    "figure.facecolor": "white",
})
PALETTE  = {"Delta": "#3b3f6b", "Alaska": "#5e7e96", "JetBlue": "#9a8aa8"}
PANEL_BG = "#f3f3f8"

# 3 rows: [top histogram / middle avg_legs / bottom quality comparison]
fig = plt.figure(figsize=(14.5, 13.5))
gs  = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.1, 0.9],
                        hspace=0.50, wspace=0.22)

# --- Top: flight duration distribution + small-scale overlay -----------------------
raw_dfs = {}
bins = np.arange(0, 15, 1)

for col, (airline, fname) in enumerate(FILES.items()):
    raw = pd.read_csv(os.path.join(DATA_DIR, fname))
    raw.columns = raw.columns.str.strip()
    raw["flight_hours"] = raw["CRS_ELAPSED_TIME"] / 60.0
    raw_dfs[airline] = raw

    ax = fig.add_subplot(gs[0, col])
    ax.set_facecolor(PANEL_BG)
    ft = raw["flight_hours"].dropna()

    # -- full dataset bars --
    counts, edges = np.histogram(ft, bins=bins)
    ax.bar(edges[:-1]+0.5, counts, 0.86,
           color=PALETTE[airline], alpha=0.9, edgecolor="white",
           lw=0.6, zorder=3, label=f"Full (n={len(ft):,})")

    # -- small-scale overlay (normalized to full scale for shape comparison) --
    ft_s = sample_hours[airline].dropna()
    s_counts, _ = np.histogram(ft_s, bins=bins)
    scale = len(ft) / len(ft_s)
    ax.bar(edges[:-1]+0.5, s_counts * scale, 0.86,
           color="white", alpha=0.60,
           edgecolor=PALETTE[airline], lw=1.8,
           hatch="///", zorder=4,
           label=f"Small-scale (n={len(ft_s):,}, ×{scale:.0f})")

    # -- mean / median lines --
    ax.axvline(ft.mean(),   color="#e4572e", ls="--", lw=1.6,
               label=f"mean = {ft.mean():.2f}h")
    ax.axvline(ft.median(), color="#f2a541", ls=":",  lw=1.8,
               label=f"median = {ft.median():.2f}h")

    ax.set_title(airline, fontweight="bold", color=PALETTE[airline], pad=8)
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

# --- Middle: ceiling vs realistic + RL current value -------------------------------
axb = fig.add_subplot(gs[1, :])
axb.set_facecolor(PANEL_BG)
airlines = list(stats.keys()); x = np.arange(len(airlines)); w = 0.30
t_means = [stats[a]["t_mean"] for a in airlines]
t_stds  = [stats[a]["t_std"]  for a in airlines]
n_means = [stats[a]["n_mean"] for a in airlines]
n_stds  = [stats[a]["n_std"]  for a in airlines]
cols    = [PALETTE[a] for a in airlines]

axb.bar(x - w, t_means, w, color=cols, alpha=0.40, edgecolor="white",
        lw=1.0, hatch="//", zorder=3, label="Time-only ceiling")
axb.bar(x,     n_means, w, color=cols, alpha=0.95, edgecolor="white",
        lw=1.0, zorder=3, label="Network-feasible (realistic)")

rl_current = 1.85
axb.bar(x + w, [rl_current]*3, w, color="#aaaaaa", alpha=0.85,
        edgecolor="white", lw=1.0, zorder=3, label=f"RL current (~{rl_current})")

axb.errorbar(x - w, t_means, yerr=t_stds, fmt="none", ecolor="#22252e",
             elinewidth=1.3, capsize=6, capthick=1.3, zorder=5)
axb.errorbar(x,     n_means, yerr=n_stds, fmt="none", ecolor="#22252e",
             elinewidth=1.3, capsize=6, capthick=1.3, zorder=5)

axb.axhline(TARGET_OLD, color="#e4572e", ls="--", lw=1.5, alpha=0.8, zorder=2,
            label=f"MIN_LEGS_FOR_PAIRING = {int(TARGET_OLD)}")
axb.axhline(5.72, color="#2ca02c", ls=":", lw=1.5, alpha=0.8, zorder=2,
            label="LLM v0 = 5.72")

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
axb.grid(axis="y", color="white", lw=1.1, zorder=0)
axb.set_axisbelow(True)
for sp in ("top", "right"):
    axb.spines[sp].set_visible(False)
axb.legend(fontsize=9, framealpha=0.9, loc="upper right", ncol=2)
axb.set_title(
    "avg_legs per Pairing: Ceiling vs Network-feasible vs RL current  (mean ± std)",
    fontweight="bold", pad=10
)

# --- Bottom: dataset quality comparison (Full vs Small-scale) ----------------------
# BDR  : Base Departure Ratio -- fraction of flights that can start an RL rollout
# CAR  : Connection Availability Rate -- fraction of flights with a next connection
#        (a more direct "is multi-leg possible" indicator than avg_legs)
ax_q = fig.add_subplot(gs[2, :])
ax_q.set_facecolor(PANEL_BG)

x_q = np.arange(len(airlines))
w_q = 0.18

BDR_COLOR = "#4472C4"   # blue
CAR_COLOR = "#ED7D31"   # orange

full_bdr_vals  = [quality[a]["full_bdr"]  for a in airlines]
samp_bdr_vals  = [quality[a]["samp_bdr"]  for a in airlines]
full_car_vals  = [quality[a]["full_car"]  for a in airlines]
samp_car_vals  = [quality[a]["samp_car"]  for a in airlines]

# Full BDR (solid blue)
b1 = ax_q.bar(x_q - 1.5*w_q, full_bdr_vals, w_q,
              color=BDR_COLOR, alpha=0.85, edgecolor="white", lw=0.8,
              zorder=3, label="Full — Base Dep. Ratio (BDR)")
# Small BDR (hatched blue)
b2 = ax_q.bar(x_q - 0.5*w_q, samp_bdr_vals, w_q,
              color=BDR_COLOR, alpha=0.38, edgecolor=BDR_COLOR, lw=1.5,
              hatch="///", zorder=3, label="Small-scale — BDR")
# Full CAR (solid orange)
b3 = ax_q.bar(x_q + 0.5*w_q, full_car_vals, w_q,
              color=CAR_COLOR, alpha=0.85, edgecolor="white", lw=0.8,
              zorder=3, label="Full — Connection Avail. Rate (CAR)")
# Small CAR (hatched orange)
b4 = ax_q.bar(x_q + 1.5*w_q, samp_car_vals, w_q,
              color=CAR_COLOR, alpha=0.38, edgecolor=CAR_COLOR, lw=1.5,
              hatch="///", zorder=3, label="Small-scale — CAR")

# value labels
for bar_group in [b1, b2, b3, b4]:
    for bar in bar_group:
        h = bar.get_height()
        ax_q.text(bar.get_x() + bar.get_width()/2, h + 0.008,
                  f"{h:.2f}", ha="center", va="bottom",
                  fontsize=8.5, fontweight="bold", color="#22252e")

ax_q.set_xticks(x_q)
ax_q.set_xticklabels(
    [f"{a}\n(full n={quality[a]['n_full']:,} / small n={quality[a]['n_samp']:,})"
     for a in airlines],
    fontsize=10, fontweight="bold"
)
for tick, a in zip(ax_q.get_xticklabels(), airlines):
    tick.set_color(PALETTE[a])

ax_q.set_ylabel("Rate  (0 = 0%, 1 = 100%)", fontsize=11)
ax_q.set_ylim(0, 1.18)
ax_q.grid(axis="y", color="white", lw=1.1, zorder=0)
ax_q.set_axisbelow(True)
for sp in ("top", "right"):
    ax_q.spines[sp].set_visible(False)
ax_q.legend(fontsize=9, framealpha=0.9, loc="upper right", ncol=2)
ax_q.set_title(
    "Dataset Quality: Full vs Small-scale  —  BDR (base departure) & CAR (connection availability)",
    fontweight="bold", pad=10
)
ax_q.text(
    0.5, -0.22,
    "BDR: fraction of flights departing from hub bases  |  "
    "CAR: fraction of flights with ≥1 feasible next connection  [min_conn, max_conn]\n"
    "Small-scale bars (hatched) close to full bars → small-scale data is representative",
    ha="center", va="top", transform=ax_q.transAxes,
    fontsize=8, style="italic", color="#6b6f7d"
)

# --- Overall styling --------------------------------------------------------------
fig.suptitle("Flight Dataset Analysis  (2019-01, per-pairing, RL-aligned)",
             fontsize=15, fontweight="bold", y=0.995)

cfg_d = AIRLINE_CONFIG["Delta"]
fig.text(
    0.5, 0.002,
    f"Network sim: base-start forced, conn∈[{cfg_d['min_conn']},{cfg_d['max_conn']}]min, "
    f"duty≤{cfg_d['max_duty']//60}h, max_legs={cfg_d['max_legs']}/duty, "
    f"min_rest={cfg_d['min_rest']}min, max_pairing_days={cfg_d['max_pairing_days']}, "
    f"{N_SIMS} sims/airline  |  base-return: allowed to fail (penalty only)",
    ha="center", fontsize=8, style="italic", color="#6b6f7d"
)

plt.tight_layout(rect=[0, 0.015, 1, 0.993])

out_path  = os.path.join(os.path.dirname(__file__), "target_avg_legs.png")
out_path2 = os.path.join(DATA_DIR, "dataset_avg.png")
plt.savefig(out_path,  dpi=150, bbox_inches="tight", facecolor="white")
plt.savefig(out_path2, dpi=150, bbox_inches="tight", facecolor="white")
print(f"\nfigure saved: {out_path}")
print(f"figure saved: {out_path2}")
plt.show()
