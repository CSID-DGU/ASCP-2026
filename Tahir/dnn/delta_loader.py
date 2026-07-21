"""
BTS On-Time Marketing Data Loader
Converts BTS T_ONTIME_MARKETING.csv → Tahir instance format

BTS CSV columns used:
  FL_DATE             : flight date  (e.g. "1/1/2019 12:00:00 AM")
  MKT_UNIQUE_CARRIER  : marketing carrier code (e.g. "DL", "AA")
  ORIGIN, DEST        : IATA airport codes
  CRS_DEP_TIME        : scheduled departure HHMM  (e.g. 1150 = 11:50)
  CRS_ELAPSED_TIME    : scheduled flight duration in minutes

All absolute times are stored as minutes since 2000-01-01 00:00 UTC
(same epoch used by the CPP/CPPSC benchmark instances).

Usage:
    # Load one day of Delta flights
    from dnn.delta_loader import load_bts_instance
    inst = load_bts_instance(carrier="DL", date="2019-01-07")

    # Discover available windows
    from dnn.delta_loader import discover_bts_instances
    windows = discover_bts_instances(carrier="DL", step_days=7)
"""

import csv
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

BTS_CSV = Path(__file__).parent.parent / "data" / "T_ONTIME_MARKETING.csv"

# Reference epoch (minutes since 2000-01-01 00:00)
EPOCH = datetime(2000, 1, 1)

# Primary hub airports per carrier
# Used as default "base" airports when bases=None
CARRIER_HUBS: Dict[str, List[str]] = {
    "DL": ["ATL", "SLC", "DTW", "MSP"],
    "AA": ["DFW", "CLT", "ORD", "MIA", "PHL", "JFK", "LAX"],
    "UA": ["ORD", "IAH", "EWR", "DEN", "SFO", "IAD"],
    "WN": ["DAL", "MDW", "HOU", "PHX", "LAS", "DEN"],
    "AS": ["SEA", "PDX", "LAX", "ANC"],
}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _parse_hhmm(s: str) -> Tuple[int, int]:
    """
    Parse HHMM integer string → (hour, minute).

    BTS stores departure/arrival times as integers without zero-padding:
      "940"  → 09:40
      "1150" → 11:50
      "0"    → 00:00
      "2359" → 23:59

    Returns (hour, minute).  Caller handles day overflow.
    """
    try:
        t = int(float(s))
    except (ValueError, TypeError):
        return 0, 0
    h, m = divmod(t, 100)
    # Clamp minute to [0, 59] in case of data glitches
    return h, min(m, 59)


def _parse_fl_date(s: str) -> datetime:
    """
    Parse BTS FL_DATE string to datetime.

    Handles common BTS formats:
      "1/1/2019 12:00:00 AM"
      "2019-01-01"
      "1/1/2019"
    """
    for fmt in (
        "%m/%d/%Y %I:%M:%S %p",
        "%m/%d/%Y %H:%M:%S",
        "%Y-%m-%d",
        "%m/%d/%Y",
    ):
        try:
            return datetime.strptime(s.strip(), fmt)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse FL_DATE: {s!r}")


# ── Public API ────────────────────────────────────────────────────────────────

def load_bts_instance(
    carrier: str = "DL",
    date: Optional[str] = None,
    date_start: Optional[str] = None,
    date_end: Optional[str] = None,
    bases: Optional[List[str]] = None,
    csv_path: Optional[str] = None,
    min_legs: int = 30,
    max_legs: int = 2000,
    sort: bool = True,
) -> Dict:
    """
    Load BTS flights for one carrier / date range as a Tahir instance dict.

    Args:
        carrier     : IATA marketing carrier code.  Default: "DL" (Delta).
        date        : Single date "YYYY-MM-DD".  Cannot be combined with
                      date_start / date_end.
        date_start  : Start date "YYYY-MM-DD" (inclusive).
        date_end    : End date   "YYYY-MM-DD" (inclusive).
        bases       : Hub airports for this carrier.  If None, uses
                      CARRIER_HUBS defaults; if the carrier is unknown or
                      no hub flights are present, falls back to the four
                      most frequent origin airports.
        csv_path    : Override path to T_ONTIME_MARKETING.csv.
        min_legs    : Raise ValueError if fewer flights are loaded.
        max_legs    : Stop after this many flights (truncates; flights are
                      processed in CSV order before sorting).
        sort        : Sort legs by departure time (default True).

    Returns:
        A dict with keys matching the Tahir CPP/CPPSC instance format:
          aircraft_type, instance_id, source,
          airports, bases, legs, availability
        Plus BTS-specific metadata:
          carrier, date_start, date_end
    """
    path = Path(csv_path) if csv_path else BTS_CSV
    if not path.exists():
        raise FileNotFoundError(
            f"BTS CSV not found: {path}\n"
            "Expected file: data/T_ONTIME_MARKETING.csv"
        )

    # ── Date filter resolution ────────────────────────────────────────────────
    if date is not None and (date_start is not None or date_end is not None):
        raise ValueError("Use either 'date' or 'date_start'/'date_end', not both.")

    if date is not None:
        d_start = d_end = datetime.strptime(date, "%Y-%m-%d")
    elif date_start is not None and date_end is not None:
        d_start = datetime.strptime(date_start, "%Y-%m-%d")
        d_end   = datetime.strptime(date_end,   "%Y-%m-%d")
        if d_end < d_start:
            raise ValueError(f"date_end ({date_end}) < date_start ({date_start})")
    else:
        d_start = d_end = None  # will pick first date seen for this carrier

    # ── Parse CSV ─────────────────────────────────────────────────────────────
    legs: List[Dict] = []
    all_airports: set = set()

    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("MKT_UNIQUE_CARRIER", "").strip() != carrier:
                continue

            try:
                fl_date = _parse_fl_date(row["FL_DATE"])
            except (ValueError, KeyError):
                continue

            # Auto-detect first date if none specified
            if d_start is None:
                d_start = d_end = fl_date

            if not (d_start <= fl_date <= d_end):
                continue

            origin = row.get("ORIGIN", "").strip()
            dest   = row.get("DEST",   "").strip()
            if not origin or not dest:
                continue

            try:
                elapsed = float(row.get("CRS_ELAPSED_TIME", "") or 0)
            except ValueError:
                elapsed = 0.0
            if elapsed <= 0:
                continue

            dep_h, dep_m = _parse_hhmm(row.get("CRS_DEP_TIME", "0"))

            # Build departure datetime (handle hour overflow, e.g. 2400)
            dep_dt = fl_date.replace(
                hour=dep_h % 24,
                minute=dep_m,
                second=0, microsecond=0,
            )
            if dep_h >= 24:
                dep_dt += timedelta(days=1)

            arr_dt = dep_dt + timedelta(minutes=int(elapsed))

            dep_abs  = int((dep_dt - EPOCH).total_seconds() // 60)
            arr_abs  = int((arr_dt - EPOCH).total_seconds() // 60)
            duration = arr_abs - dep_abs

            all_airports.add(origin)
            all_airports.add(dest)

            legs.append({
                "leg_name":      f"LEG_{len(legs):06d}",
                "aircraft_type": carrier,
                "origin":        origin,
                "dest":          dest,
                "dep_dt":        dep_dt,
                "arr_dt":        arr_dt,
                "dep_abs":       dep_abs,
                "arr_abs":       arr_abs,
                "dep_day":       int((dep_dt - EPOCH).days),
                "dep_min":       dep_dt.hour * 60 + dep_dt.minute,
                "arr_day":       int((arr_dt - EPOCH).days),
                "arr_min":       arr_dt.hour * 60 + arr_dt.minute,
                "duration":      duration,
            })

            if len(legs) >= max_legs:
                break

    # ── Validation ────────────────────────────────────────────────────────────
    if d_start is None:
        raise ValueError(
            f"No flights found for carrier={carrier!r} in {path}. "
            "Check the MKT_UNIQUE_CARRIER column."
        )

    if len(legs) < min_legs:
        raise ValueError(
            f"Only {len(legs)} leg(s) loaded for carrier={carrier!r}, "
            f"dates={d_start.date()} – {d_end.date()}. "
            f"Minimum required: {min_legs}. "
            "Try widening the date range or check the CSV."
        )

    # ── Sort and assign flight IDs ─────────────────────────────────────────────
    if sort:
        legs.sort(key=lambda x: (x["dep_abs"], x["origin"], x["dest"]))
    for i, leg in enumerate(legs):
        leg["flight_id"] = i + 1  # 1-based: avoids -0 == 0 deadhead encoding bug

    # ── Base airport resolution ────────────────────────────────────────────────
    if bases is not None:
        active_bases = [b for b in bases if b in all_airports]
    else:
        default_hubs = CARRIER_HUBS.get(carrier, [])
        active_bases = [b for b in default_hubs if b in all_airports]

    if not active_bases:
        # Fallback: pick the four most frequent origin airports
        freq = Counter(leg["origin"] for leg in legs)
        active_bases = [a for a, _ in freq.most_common(4)]

    # ── Instance ID label ─────────────────────────────────────────────────────
    if d_start == d_end:
        instance_label = d_start.strftime("%Y%m%d")
    else:
        instance_label = f"{d_start.strftime('%Y%m%d')}_{d_end.strftime('%Y%m%d')}"

    sorted_airports = sorted(all_airports)

    return {
        # Tahir-compatible keys
        "aircraft_type": carrier,
        "instance_id":   instance_label,
        "source":        "BTS",
        "airports":      sorted_airports,
        "bases":         active_bases,
        "legs":          legs,
        "availability":  {},
        # BTS-specific metadata
        "carrier":       carrier,
        "date_start":    d_start,
        "date_end":      d_end,
    }


def discover_bts_instances(
    carrier: str = "DL",
    csv_path: Optional[str] = None,
    step_days: int = 7,
) -> List[Dict]:
    """
    Scan the BTS CSV and return a list of date windows for a given carrier.

    Each window spans `step_days` calendar days and reports the total number
    of flights within it.

    Args:
        carrier   : IATA carrier code.
        csv_path  : Override path to CSV.
        step_days : Width of each window in days.

    Returns:
        List of dicts: [{"carrier", "date_start", "date_end", "n_legs"}, ...]
    """
    path = Path(csv_path) if csv_path else BTS_CSV
    if not path.exists():
        return []

    date_counts: Dict[datetime, int] = defaultdict(int)

    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("MKT_UNIQUE_CARRIER", "").strip() != carrier:
                continue
            try:
                fl_date = _parse_fl_date(row["FL_DATE"])
                date_counts[fl_date] += 1
            except ValueError:
                continue

    if not date_counts:
        return []

    sorted_dates = sorted(date_counts.keys())
    windows: List[Dict] = []
    i = 0
    while i < len(sorted_dates):
        w_start = sorted_dates[i]
        w_end   = w_start + timedelta(days=step_days - 1)
        n = sum(v for d, v in date_counts.items() if w_start <= d <= w_end)
        windows.append({
            "carrier":    carrier,
            "date_start": w_start.strftime("%Y-%m-%d"),
            "date_end":   w_end.strftime("%Y-%m-%d"),
            "n_legs":     n,
        })
        # Advance to the first date past this window
        while i < len(sorted_dates) and sorted_dates[i] <= w_end:
            i += 1

    return windows


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Inspect BTS T_ONTIME_MARKETING.csv and preview an instance."
    )
    parser.add_argument("--carrier",    default="DL",
                        help="IATA carrier code (default: DL)")
    parser.add_argument("--date",       default=None,
                        help="Single date YYYY-MM-DD")
    parser.add_argument("--date_start", default=None,
                        help="Start date YYYY-MM-DD")
    parser.add_argument("--date_end",   default=None,
                        help="End date YYYY-MM-DD")
    parser.add_argument("--discover",   action="store_true",
                        help="List available weekly windows and exit")
    parser.add_argument("--step_days",  type=int, default=7,
                        help="Window width for --discover (default: 7)")
    parser.add_argument("--csv",        default=None,
                        help="Path to T_ONTIME_MARKETING.csv")
    args = parser.parse_args()

    if args.discover:
        windows = discover_bts_instances(
            carrier=args.carrier,
            csv_path=args.csv,
            step_days=args.step_days,
        )
        if not windows:
            print(f"No data found for carrier={args.carrier!r}")
        else:
            print(f"Available {args.step_days}-day windows for {args.carrier}:")
            print(f"  {'Start':12s} {'End':12s} {'Flights':>8}")
            print(f"  {'-'*12} {'-'*12} {'-'*8}")
            for w in windows:
                print(f"  {w['date_start']:12s} {w['date_end']:12s} {w['n_legs']:8d}")
    else:
        inst = load_bts_instance(
            carrier=args.carrier,
            date=args.date,
            date_start=args.date_start,
            date_end=args.date_end,
            csv_path=args.csv,
        )
        print(f"Instance loaded:")
        print(f"  carrier    : {inst['carrier']}")
        print(f"  dates      : {inst['date_start'].date()} – {inst['date_end'].date()}")
        print(f"  legs       : {len(inst['legs'])}")
        print(f"  airports   : {len(inst['airports'])}")
        print(f"  bases      : {inst['bases']}")
        print(f"  instance_id: {inst['instance_id']}")
        if inst["legs"]:
            first = inst["legs"][0]
            last  = inst["legs"][-1]
            print(f"  first leg  : {first['origin']}→{first['dest']} "
                  f"dep={first['dep_dt']}")
            print(f"  last  leg  : {last['origin']}→{last['dest']} "
                  f"dep={last['dep_dt']}")
