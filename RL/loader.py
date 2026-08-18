import random
import pandas as pd
from collections import Counter


def convert_time(hhmm):
    hhmm = int(hhmm)
    h = hhmm // 100
    m = hhmm % 100
    return h + m / 60


# UTC offset per airport (minutes, standard time). BTS CRS_DEP_TIME/CRS_ARR_TIME are each
# airport's local time, so for connections between airports in different time zones, subtracting
# dep_time directly yields a negative/bogus gap and masks out connections that are actually
# feasible (e.g. after landing at LAX (PT) from ATL (ET), the next flight's dep_time is in PT,
# which doesn't line up with an ET-based current_time clock).
# Anchoring dep_time to UTC absolute time means arr_time = dep_time + elapsed (block time,
# timezone-independent) automatically yields a correct UTC arrival time too.
# (same source as the _UTC table in analysis/flight_time_distribution.py)
_UTC_OFFSET_MIN = {
    **{ap: -600 for ap in ['HNL', 'KOA', 'LIH', 'OGG']},
    **{ap: -540 for ap in ['ANC', 'FAI']},
    **{ap: -480 for ap in ['GEG', 'LAS', 'LAX', 'OAK', 'ONT', 'PDX', 'PSP', 'RNO', 'SAN', 'SEA',
                            'SFO', 'SJC', 'SMF', 'SNA']},
    **{ap: -420 for ap in ['ABQ', 'BIL', 'BOI', 'BZN', 'COS', 'DEN', 'EGE', 'ELP', 'FCA', 'HDN',
                            'JAC', 'MSO', 'MTJ', 'PHX', 'SLC', 'TUS']},
    **{ap: -360 for ap in ['ATW', 'AUS', 'BHM', 'BIS', 'BNA', 'BTR', 'CID', 'DAL', 'DFW', 'DSM',
                            'ECP', 'FAR', 'FSD', 'GPT', 'GRB', 'HOU', 'HSV', 'IAH', 'ICT', 'JAN',
                            'LFT', 'LIT', 'MCI', 'MDW', 'MEM', 'MKE', 'MOB', 'MSN', 'MSP', 'MSY',
                            'OKC', 'OMA', 'ORD', 'PNS', 'SAT', 'STL', 'TUL', 'VPS', 'XNA']},
    **{ap: -300 for ap in ['ABE', 'AGS', 'ALB', 'ATL', 'AVL', 'AVP', 'BDL', 'BOS', 'BUF', 'BWI',
                            'CAE', 'CAK', 'CHA', 'CHO', 'CHS', 'CLE', 'CLT', 'CMH', 'CRW', 'CVG',
                            'DAB', 'DAY', 'DCA', 'DTW', 'EWR', 'EYW', 'FAY', 'FLL', 'FNT', 'GNV',
                            'GRR', 'GSO', 'GSP', 'HPN', 'IAD', 'ILM', 'IND', 'JAX', 'JFK', 'LEX',
                            'LGA', 'MCO', 'MDT', 'MHT', 'MIA', 'MLB', 'MYR', 'ORF', 'PBI', 'PHF',
                            'PHL', 'PIT', 'PVD', 'PWM', 'RDU', 'RIC', 'ROA', 'ROC', 'RSW', 'SAV',
                            'SDF', 'SRQ', 'SYR', 'TLH', 'TPA', 'TRI', 'TYS']},
    **{ap: -240 for ap in ['SJU', 'STT', 'STX']},
}


def utc_offset_hours(airport_code):
    """UTC offset for an airport (hours). Defaults to Eastern (-5h) if not in the table."""
    return _UTC_OFFSET_MIN.get(airport_code, -300) / 60.0


def build_airport_map(path):
    """Build an airport-to-int map from the full BTS CSV.

    path: a str or list of str. Multiple carrier CSVs can be combined to build a unified
    airport ID space.
    Sorted by descending frequency: index 0 = the highest-frequency airport (hub/base candidate).
    IDs stay consistent across episodes even when different rolling windows are used.
    """
    paths = [path] if isinstance(path, str) else path
    counts = Counter()
    for p in paths:
        df = pd.read_csv(p, usecols=["ORIGIN", "DEST"]).dropna()
        counts.update(list(df["ORIGIN"]) + list(df["DEST"]))
    airports_sorted = sorted(counts.keys(), key=lambda a: -counts[a])
    return {a: i for i, a in enumerate(airports_sorted)}


def bases_to_ids(bases, airport_map):
    """Convert a list of base code strings to integer IDs.

    Args:
        bases:       list of airport code strings (e.g. ["ATL", "DTW", "MSP"])
        airport_map: airport-to-int map produced by build_airport_map()

    Returns:
        A list of integer IDs. Codes not present in airport_map are dropped with a warning.
    """
    ids = [airport_map[b] for b in bases if b in airport_map]
    missing = [b for b in bases if b not in airport_map]
    if missing:
        print(f"[bases_to_ids] warning: bases not in airport_map were excluded: {missing}")
    if not ids:
        raise ValueError(f"No valid bases found. bases={bases}")
    return ids


def get_bases(flights, n_bases=3):
    """Return the top n_bases airport IDs by frequency from a list of flight dicts.

    Since airport_map is sorted by descending frequency, the result is typically
    [0, 1, ..., n_bases-1].
    """
    counts = Counter()
    for f in flights:
        counts[f["origin"]] += 1
        counts[f["dest"]] += 1
    return [a for a, _ in counts.most_common(n_bases)]


def load_flights(path, limit=50, seed=42, n_days_max=None, use_utc=False):
    """Load flights from BTS data.

    Airport index: descending frequency over the full CSV -> index 0 = hub.
    The same airport_map is always used regardless of limit.

    use_utc: if True, anchors dep_time to UTC absolute time.
        Defaults to False -- existing checkpoints were trained with the loader's earlier
        (local-time) behavior, so unconditionally enabling this at eval time would feed an
        OOD distribution the model never saw during training. Only enable this when evaluating
        a model that was trained with this option turned on.
    """
    df = pd.read_csv(path)
    df = df[[
        "ORIGIN", "DEST", "CRS_DEP_TIME", "CRS_ARR_TIME", "CRS_ELAPSED_TIME", "FL_DATE"
    ]].dropna()
    df["FL_DATE"] = pd.to_datetime(df["FL_DATE"], format="mixed")

    # Build airport_map over the full dataset (computed before slicing to a subset)
    airport_counts = Counter(list(df["ORIGIN"]) + list(df["DEST"]))
    airports_sorted = sorted(airport_counts.keys(), key=lambda a: -airport_counts[a])
    airport_map = {a: i for i, a in enumerate(airports_sorted)}

    if n_days_max is not None:
        dates = sorted(df["FL_DATE"].unique())[:n_days_max]
        n_per_day = max(1, limit // len(dates))
        pieces = [
            day_df.sample(min(n_per_day, len(day_df)), random_state=seed)
            for date in dates
            for day_df in [df[df["FL_DATE"] == date]]
        ]
        df = pd.concat(pieces).reset_index(drop=True).head(limit)
    else:
        df = df.head(limit)

    base_date = df["FL_DATE"].min()
    df["day_offset"] = (df["FL_DATE"] - base_date).dt.days
    df["dep_time"] = df["CRS_DEP_TIME"].apply(convert_time) + df["day_offset"] * 24
    if use_utc:
        # Anchor dep_time to UTC absolute time (origin local time minus UTC offset)
        # -> makes connection-gap calculations correct across airports in different time zones
        df["dep_time"] -= df["ORIGIN"].map(utc_offset_hours)
    # CRS_ELAPSED_TIME (minutes, block time) is timezone-independent -> adding it to dep_time
    # keeps arr_time on the same basis
    df["arr_time"] = df["dep_time"] + df["CRS_ELAPSED_TIME"] / 60.0

    df = df.sort_values("dep_time").reset_index(drop=True)

    df["origin"] = df["ORIGIN"].map(airport_map)
    df["dest"]   = df["DEST"].map(airport_map)

    flights = []
    for _, row in df.iterrows():
        flights.append({
            "id":       len(flights),
            "origin":   int(row["origin"]),
            "dest":     int(row["dest"]),
            "dep_time": float(row["dep_time"]),
            "arr_time": float(row["arr_time"]),
        })

    return flights



def sample_connected_subnet(flights_window, base_id, n_max):
    """Subnet sampling based on the airport (station) set -- solves the star-graph problem.
      1. Always include the base.
      2. Add spokes one at a time, ordered by traffic volume.
      3. Keep only edges 'internal' to the chosen airport set (origin in chosen AND dest in
         chosen) -> spoke-spoke edges survive, enabling long chains like ATL->MSP->SLC->ATL.
      4. Trim to chronological order if n_max is exceeded.
    """
    from collections import Counter
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


def load_flights_rolling(
    path,
    window_days=5,
    offset_days=0,
    airport_map=None,
    base_airport=None,
    n_max=None,
    df=None,
    use_utc=False,
):
    """Load real date-based data using a sliding window approach.

    Varying offset_days per episode allows training on different flight compositions.
    All routes (including spoke-spoke) are used as-is, with no hub_only restriction.

    Base-first sampling (when base_airport + n_max are given):
        base_flights = origin=base or dest=base -> all included
        mid_flights  = the remaining spoke-spoke flights -> randomly sampled to fill
                       the remaining slots
    Returns all flights if n_max is not given.

    Args:
        path:          path to the BTS CSV
        window_days:   window size (days), default 5
        offset_days:   starting index into the full date list (randomized per episode)
        airport_map:   airport ID map computed over the full dataset; if None, recomputed
                       from the full CSV (slow).
        base_airport:  episode base airport ID; used for base-first sampling
        n_max:         max flights per episode; base-first sampling applies when exceeded
        df:            pre-loaded DataFrame; if given, skips re-loading the CSV (optimization
                       for repeated per-episode calls)

    Returns:
        list of flight dicts (sorted ascending by dep_time)
    """
    if df is None:
        df = pd.read_csv(path)
        df = df[[
            "ORIGIN", "DEST", "CRS_DEP_TIME", "CRS_ARR_TIME", "CRS_ELAPSED_TIME", "FL_DATE"
        ]].dropna()
        df["FL_DATE"] = pd.to_datetime(df["FL_DATE"], format="mixed")

    # If airport_map is not given, build it over the full dataset (keeps IDs consistent across episodes)
    if airport_map is None:
        counts = Counter(list(df["ORIGIN"]) + list(df["DEST"]))
        airports_sorted = sorted(counts.keys(), key=lambda a: -counts[a])
        airport_map = {a: i for i, a in enumerate(airports_sorted)}

    # Extract window dates
    dates = sorted(df["FL_DATE"].unique())
    window_dates = dates[offset_days: offset_days + window_days]
    if not window_dates:
        return []

    df = df[df["FL_DATE"].isin(window_dates)].copy()

    # Time conversion + day offset relative to the window start date
    base_date = min(window_dates)
    df["day_offset"] = (df["FL_DATE"] - base_date).dt.days
    df["dep_time"] = df["CRS_DEP_TIME"].apply(convert_time) + df["day_offset"] * 24
    if use_utc:
        # Anchor dep_time to UTC absolute time -- same reasoning as load_flights()
        df["dep_time"] -= df["ORIGIN"].map(utc_offset_hours)
    df["arr_time"] = df["dep_time"] + df["CRS_ELAPSED_TIME"] / 60.0

    df = df.sort_values("dep_time").reset_index(drop=True)

    df["origin"] = df["ORIGIN"].map(airport_map)
    df["dest"]   = df["DEST"].map(airport_map)
    df = df.dropna(subset=["origin", "dest"])

    flights = []
    for _, row in df.iterrows():
        flights.append({
            "id":       len(flights),
            "origin":   int(row["origin"]),
            "dest":     int(row["dest"]),
            "dep_time": float(row["dep_time"]),
            "arr_time": float(row["arr_time"]),
        })

    # connected subnet sampling: solves the star-graph problem (base-first random sampling -> airport-set based)
    if n_max is not None and len(flights) > n_max and base_airport is not None:
        flights = sample_connected_subnet(flights, base_airport, n_max)

    return flights
