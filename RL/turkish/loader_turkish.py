"""
Loader for Turkish Airlines .legs files.

File format: ORIGIN|DEST|FLEET|AC_NUM|DEP_DT_LOCAL|ARR_DT_LOCAL|DEP_TZ_MIN|ARR_TZ_MIN
UTC conversion: dep_utc = dep_local - DEP_TZ_MIN minutes

Returns the same flight dict format as load_flights_rolling(), so existing code
(train.py / evaluate_ip.py, etc.) can be reused as-is.
"""
import os
import glob
import random
import pandas as pd
from collections import Counter

# Reproduces Table 2 of Zeren & Ozkol (2016, Expert Systems With Applications 55) --
# scanning N = 0 to 15 for "planning month + N days" converges to within 1.3% error for all
# six months at N=6-7. For Feb, N=7 gives 15,742 flights vs. a target of 15,738 (0.03% error).
# This value is adopted as the default window for the turkish dataset.
ZEREN_FEB_FILE = "tt201402.legs"
ZEREN_FEB_WINDOW = ("2014-02-01", "2014-03-08")  # buffer=7 days, half-open interval [start, end)


def _parse_legs_file(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) != 8:
                continue
            try:
                rows.append({
                    "ORIGIN": parts[0],
                    "DEST":   parts[1],
                    "FLEET":  parts[2],
                    "DEP_DT": parts[4],
                    "ARR_DT": parts[5],
                    "DEP_TZ": int(parts[6]),
                    "ARR_TZ": int(parts[7]),
                })
            except (ValueError, IndexError):
                continue
    return pd.DataFrame(rows)


def parse_legs_dir(dir_path, fleet_prefix="3", files=None, date_range=None):
    """
    Parse .legs files in a directory -> return a DataFrame with UTC-based times.

    Returned columns:
        ORIGIN, DEST          airport codes
        dep_utc, arr_utc      UTC datetime
        dep_date_utc          date of dep_utc (used for the rolling window)

    Args:
        dir_path:     directory containing the .legs files
        fleet_prefix: fleet code prefix to include. None = all. Default "3" = Airbus narrow body.
        files:        list of file names to use (e.g. ["tt201401.legs"]). None = all files in
                       the directory.
                       Note: monthly files have overlapping date ranges (each file includes a
                       buffer before/after), so passing multiple files at once double-counts
                       flights in the overlapping region.
        date_range:   (start, end) string tuple. Filters dep_date_utc to [start, end).
                       None = no filtering. See ZEREN_FEB_WINDOW.
    """
    if files is not None:
        legs_files = [os.path.join(dir_path, f) for f in files]
        missing = [p for p in legs_files if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(f".legs file(s) not found: {missing}")
    else:
        legs_files = sorted(glob.glob(os.path.join(dir_path, "*.legs")))
    if not legs_files:
        raise FileNotFoundError(f"No .legs files found: {dir_path}")

    dfs = [_parse_legs_file(p) for p in legs_files]
    df  = pd.concat(dfs, ignore_index=True)

    if fleet_prefix is not None:
        df = df[df["FLEET"].str.startswith(fleet_prefix)].copy()

    # Local time -> UTC: dep_utc = dep_local - DEP_TZ minutes
    df["dep_local"] = pd.to_datetime(df["DEP_DT"])
    df["arr_local"] = pd.to_datetime(df["ARR_DT"])
    df["dep_utc"]   = df["dep_local"] - pd.to_timedelta(df["DEP_TZ"], unit="min")
    df["arr_utc"]   = df["arr_local"] - pd.to_timedelta(df["ARR_TZ"], unit="min")
    df["dep_date_utc"] = df["dep_utc"].dt.normalize()

    # Filter out same origin-dest and arr <= dep (timezone inversion on short same-direction hops)
    df = df[(df["ORIGIN"] != df["DEST"]) & (df["arr_utc"] > df["dep_utc"])].copy()

    if date_range is not None:
        start, end = date_range
        df = df[(df["dep_date_utc"] >= start) & (df["dep_date_utc"] < end)].copy()

    return df[["ORIGIN", "DEST", "dep_utc", "arr_utc", "dep_date_utc"]].reset_index(drop=True)


def build_airport_map_turkish(dir_path=None, fleet_prefix="3", df=None):
    """
    Build an airport-to-int ID map.

    Sorted by descending frequency (index 0 = the highest-frequency airport = HB1/HB2 homebase).

    Args:
        dir_path:     directory path to pass to parse_legs_dir. Can be omitted if df is given.
        fleet_prefix: fleet filter to pass to parse_legs_dir. Ignored if df is given.
        df:           a DataFrame already loaded via parse_legs_dir(). Skips re-parsing if given.
    """
    if df is None:
        if dir_path is None:
            raise ValueError("Either dir_path or df must be provided.")
        df = parse_legs_dir(dir_path, fleet_prefix=fleet_prefix)
    counts = Counter(list(df["ORIGIN"]) + list(df["DEST"]))
    airports_sorted = sorted(counts.keys(), key=lambda a: -counts[a])
    return {a: i for i, a in enumerate(airports_sorted)}


def sample_connected_subnet(flights_window, base_id, n_max):
    """Subnet sampling based on the airport set -- solves the star-graph problem (same logic as loader.py)."""
    deg = Counter()
    for f in flights_window:
        deg[f["origin"]] += 1
        deg[f["dest"]] += 1
    spokes = [a for a, _ in deg.most_common() if a != base_id]
    chosen = {base_id}
    out = []
    for s in spokes:
        chosen.add(s)
        candidate = [f for f in flights_window
                     if f["origin"] in chosen and f["dest"] in chosen]
        if len(candidate) >= n_max:
            out = candidate
            break
        out = candidate
    out = sorted(out, key=lambda f: f["dep_time"])[:n_max]
    for i, f in enumerate(out):
        f["id"] = i
    return out


def load_flights_rolling_turkish(
    window_days=5,
    offset_days=0,
    airport_map=None,
    base_airport=None,
    df=None,
    n_max=None,
):
    """
    Load flights from Turkish .legs data using a sliding window approach.

    Returns the same flight dict format as load_flights_rolling().
    dep_time / arr_time: time relative to the window start, in UTC hours.

    Args:
        window_days:  window size (days), default 5
        offset_days:  starting index into the full date list (randomized per episode)
        airport_map:  airport-to-int map produced by build_airport_map_turkish()
        base_airport: episode base airport ID; used for base-first sampling
        df:           DataFrame pre-loaded via parse_legs_dir() (required)
        n_max:        max flights per episode; base-first sampling applies when exceeded

    Returns:
        list of flight dicts (ascending by dep_time)
    """
    if df is None:
        raise ValueError("df must be pre-loaded via parse_legs_dir() and passed in.")
    if airport_map is None:
        raise ValueError("airport_map must be built via build_airport_map_turkish() and passed in.")

    dates = sorted(df["dep_date_utc"].unique())
    window_dates = dates[offset_days: offset_days + window_days]
    if not window_dates:
        return []

    df_win = df[df["dep_date_utc"].isin(window_dates)].copy()

    # Time relative to the window start UTC (hours)
    base_dt = pd.Timestamp(min(window_dates))
    df_win["dep_time"] = (df_win["dep_utc"] - base_dt).dt.total_seconds() / 3600.0
    df_win["arr_time"] = (df_win["arr_utc"] - base_dt).dt.total_seconds() / 3600.0

    df_win = df_win.sort_values("dep_time").reset_index(drop=True)

    df_win["origin"] = df_win["ORIGIN"].map(airport_map)
    df_win["dest"]   = df_win["DEST"].map(airport_map)
    df_win = df_win.dropna(subset=["origin", "dest"])

    flights = []
    for _, row in df_win.iterrows():
        flights.append({
            "id":       len(flights),
            "origin":   int(row["origin"]),
            "dest":     int(row["dest"]),
            "dep_time": float(row["dep_time"]),
            "arr_time": float(row["arr_time"]),
        })

    # connected subnet sampling: solves the star-graph problem
    if n_max is not None and len(flights) > n_max and base_airport is not None:
        flights = sample_connected_subnet(flights, base_airport, n_max)

    return flights
