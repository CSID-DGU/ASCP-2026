import random
import hashlib
import json
import pandas as pd
from collections import Counter
from zoneinfo import ZoneInfo


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
    **{ap: -600 for ap in ['ADK', 'HNL', 'KOA', 'LIH', 'OGG']},
    **{ap: -540 for ap in ['ADQ', 'ANC', 'BET', 'BRW', 'CDV', 'FAI', 'JNU', 'KTN', 'OME', 'OTZ',
                            'PSG', 'SCC', 'SIT', 'WRG', 'YAK']},
    **{ap: -480 for ap in ['BLI', 'BUR', 'GEG', 'LAS', 'LAX', 'LGB', 'OAK', 'ONT', 'PDX', 'PSP',
                            'RNO', 'SAN', 'SBA', 'SEA', 'SFO', 'SJC', 'SMF', 'SNA']},
    **{ap: -420 for ap in ['ABQ', 'BIL', 'BOI', 'BZN', 'COS', 'DEN', 'EGE', 'ELP', 'FCA', 'HDN',
                            'JAC', 'MSO', 'MTJ', 'PHX', 'SLC', 'TUS']},
    **{ap: -360 for ap in ['ATW', 'AUS', 'BHM', 'BIS', 'BNA', 'BTR', 'CID', 'DAL', 'DFW', 'DSM',
                            'ECP', 'FAR', 'FSD', 'GPT', 'GRB', 'HOU', 'HSV', 'IAH', 'ICT', 'JAN',
                            'LFT', 'LIT', 'MCI', 'MDW', 'MEM', 'MKE', 'MOB', 'MSN', 'MSP', 'MSY',
                            'OKC', 'OMA', 'ORD', 'PNS', 'SAT', 'STL', 'TUL', 'VPS', 'XNA']},
    **{ap: -300 for ap in ['ABE', 'AGS', 'ALB', 'ATL', 'AVL', 'AVP', 'BDL', 'BOS', 'BTV', 'BUF', 'BWI',
                            'CAE', 'CAK', 'CHA', 'CHO', 'CHS', 'CLE', 'CLT', 'CMH', 'CRW', 'CVG',
                            'DAB', 'DAY', 'DCA', 'DTW', 'EWR', 'EYW', 'FAY', 'FLL', 'FNT', 'GNV',
                            'GRR', 'GSO', 'GSP', 'HPN', 'IAD', 'ILM', 'IND', 'JAX', 'JFK', 'LEX',
                            'LGA', 'MCO', 'MDT', 'MHT', 'MIA', 'MLB', 'MYR', 'ORF', 'PBI', 'PHF',
                            'ORH', 'PHL', 'PIT', 'PVD', 'PWM', 'RDU', 'RIC', 'ROA', 'ROC', 'RSW',
                            'SAV', 'SDF', 'SRQ', 'SWF', 'SYR', 'TLH', 'TPA', 'TRI', 'TYS']},
    **{ap: -240 for ap in ['BQN', 'PSE', 'SJU', 'STT', 'STX']},
}

_AIRPORT_TIMEZONE = {
    **{ap: "Pacific/Honolulu" for ap in ["HNL", "KOA", "LIH", "OGG"]},
    "ADK": "America/Adak",
    **{ap: "America/Anchorage" for ap in [
        "ADQ", "ANC", "BET", "BRW", "CDV", "FAI", "JNU", "KTN", "OME", "OTZ",
        "PSG", "SCC", "SIT", "WRG", "YAK",
    ]},
    **{ap: "America/Los_Angeles" for ap, offset in _UTC_OFFSET_MIN.items() if offset == -480},
    **{ap: "America/Denver" for ap, offset in _UTC_OFFSET_MIN.items() if offset == -420},
    **{ap: "America/Chicago" for ap, offset in _UTC_OFFSET_MIN.items() if offset == -360},
    **{ap: "America/New_York" for ap, offset in _UTC_OFFSET_MIN.items() if offset == -300},
    **{ap: "America/Puerto_Rico" for ap, offset in _UTC_OFFSET_MIN.items() if offset == -240},
    "PHX": "America/Phoenix",
}


def utc_offset_hours(airport_code, local_datetime=None):
    """공항의 UTC offset을 반환하며, 날짜가 있으면 서머타임까지 반영함."""
    try:
        if local_datetime is None:
            return _UTC_OFFSET_MIN[airport_code] / 60.0
        dt = pd.Timestamp(local_datetime).to_pydatetime().replace(tzinfo=None)
        offset = dt.replace(tzinfo=ZoneInfo(_AIRPORT_TIMEZONE[airport_code])).utcoffset()
        return offset.total_seconds() / 3600.0
    except KeyError as exc:
        raise ValueError(f"UTC offset이 등록되지 않은 BTS 공항: {airport_code}") from exc


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


def airport_map_hash(airport_map):
    """공항 embedding 사전의 의미와 순서를 검증하는 안정적인 hash를 반환함."""
    payload = json.dumps(airport_map, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_airport_map(airport_map, n_airports=None):
    """Checkpoint에서 복원한 공항 사전이 연속 ID를 갖는지 검증함."""
    if not isinstance(airport_map, dict) or not airport_map:
        raise ValueError("checkpoint에 유효한 airport_map이 없음")
    normalized = {str(code): int(idx) for code, idx in airport_map.items()}
    if sorted(normalized.values()) != list(range(len(normalized))):
        raise ValueError("airport_map ID가 0부터 시작하는 연속 범위가 아님")
    if n_airports is not None and len(normalized) != int(n_airports):
        raise ValueError(
            f"airport_map 크기({len(normalized)})와 embedding 크기({n_airports})가 다름"
        )
    return normalized


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


def load_flights(path, limit=50, seed=42, n_days_max=None):
    """Load flights from BTS data.

    Airport index: descending frequency over the full CSV -> index 0 = hub.
    The same airport_map is always used regardless of limit.

    BTS의 공항별 현지 출발시각을 항상 UTC 절대시각으로 변환함.
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
    # 공항별 현지시각을 UTC 절대시각으로 맞춰 서로 다른 시간대의 연결시간을 비교함.
    df["dep_time"] -= df.apply(
        lambda row: utc_offset_hours(row["ORIGIN"], row["FL_DATE"]), axis=1
    )
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
    # BTS 데이터는 항상 UTC 절대시각으로 맞춤. Turkish는 이 로더를 사용하지 않음.
    df["dep_time"] -= df.apply(
        lambda row: utc_offset_hours(row["ORIGIN"], row["FL_DATE"]), axis=1
    )
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
