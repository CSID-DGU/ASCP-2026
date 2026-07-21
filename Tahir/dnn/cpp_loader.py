"""
CPP Instance Loader
Supports two dataset formats:

1. CPP_instances (original, per-instance subdirs):
   data/cpp_instances/CPP_instances/AT_320/instance_0/legs
   Format: `name is_base;` / `LEG_... BASE1 2000-01-01 21:50 AIR1 2000-01-02 00:32;`

2. CPPSC_Instances (full 7-type dataset, one legs file per type):
   data/cppsc_instances/CPPSC_Instances/320/legs
   Format: `airport is_base` / `LEG_06_0 AIR15 2000-01-06 21:05 BASE1 2000-01-07 00:21`
   + availability_constraints_1 ~ availability_constraints_5

References:
  Kasirzadeh, Saddoune, Soumis 2017 (CPP_instances)
  Tahir, Desaulniers, El Hallaoui 2021 (CPPSC_Instances)
"""

import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional


CPP_DIR   = Path(__file__).parent.parent / "data" / "cpp_instances"   / "CPP_instances"
CPPSC_DIR = Path(__file__).parent.parent / "data" / "cppsc_instances" / "CPPSC_Instances"

# All 7 aircraft types in the paper
ALL_TYPES = ["727", "09", "94", "95", "757", "319", "320"]


# ── Shared parsers (handle both semicolon and non-semicolon formats) ──────────

def _parse_airports(path: str) -> Tuple[List[str], List[str]]:
    """Returns (all_airports, base_airports). Handles both formats."""
    all_airports, bases = [], []
    with open(path) as f:
        for line in f:
            line = line.strip().rstrip(";")
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            # Skip header lines
            if parts[0].lower() in ("name", "airport"):
                continue
            try:
                name, is_base = parts[0], int(parts[1])
            except ValueError:
                continue
            all_airports.append(name)
            if is_base:
                bases.append(name)
    return all_airports, bases


def _parse_legs(path: str, aircraft_type: str) -> List[Dict]:
    """Parse legs file into list of flight dicts. Handles both formats."""
    legs  = []
    epoch = datetime(2000, 1, 1)

    with open(path) as f:
        for line in f:
            line = line.strip().rstrip(";")
            if not line or line.startswith("leg_name"):
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            try:
                leg_name = parts[0]
                origin   = parts[1]
                dep_dt   = datetime.strptime(parts[2] + " " + parts[3], "%Y-%m-%d %H:%M")
                dest     = parts[4]
                arr_dt   = datetime.strptime(parts[5] + " " + parts[6], "%Y-%m-%d %H:%M")
            except ValueError:
                continue

            dep_abs  = int((dep_dt - epoch).total_seconds() // 60)
            arr_abs  = int((arr_dt - epoch).total_seconds() // 60)
            duration = arr_abs - dep_abs

            legs.append({
                "leg_name":      leg_name,
                "aircraft_type": aircraft_type,
                "origin":        origin,
                "dest":          dest,
                "dep_dt":        dep_dt,
                "arr_dt":        arr_dt,
                "dep_abs":       dep_abs,
                "arr_abs":       arr_abs,
                "dep_day":       int((dep_dt - epoch).days),
                "dep_min":       dep_dt.hour * 60 + dep_dt.minute,
                "arr_day":       int((arr_dt - epoch).days),
                "arr_min":       arr_dt.hour * 60 + arr_dt.minute,
                "duration":      duration,
            })
    return legs


def _parse_availability(path: str) -> Dict[str, Dict[int, int]]:
    """Returns {base: {day: avail_count}}. Handles both formats."""
    avail: Dict[str, Dict[int, int]] = {}
    with open(path) as f:
        for line in f:
            line = line.strip().rstrip(";")
            if not line or line.startswith("base"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                base, day, count = parts[0], int(parts[1]), int(parts[2])
            except ValueError:
                continue
            avail.setdefault(base, {})[day] = count
    return avail


# ── CPP_instances loader (multi-instance per type) ───────────────────────────

def load_instance(aircraft_type: str, instance_id: int) -> Dict:
    """Load one instance from CPP_instances dataset."""
    at_dir   = CPP_DIR / f"AT_{aircraft_type}"
    inst_dir = at_dir / f"instance_{instance_id}"

    if not inst_dir.exists():
        raise FileNotFoundError(f"{inst_dir}")

    airports, bases = _parse_airports(str(at_dir / "airports"))
    legs = _parse_legs(str(inst_dir / "legs"), aircraft_type=aircraft_type)
    avail_path = inst_dir / "availability_constraints"
    availability = _parse_availability(str(avail_path)) if avail_path.exists() else {}

    for i, leg in enumerate(legs):
        leg["flight_id"] = i + 1  # 1-based: avoids -0 == 0 deadhead encoding bug

    return {
        "aircraft_type": aircraft_type,
        "instance_id":   instance_id,
        "source":        "CPP",
        "airports":      airports,
        "bases":         bases,
        "legs":          legs,
        "availability":  availability,
    }


def discover_instances() -> List[Tuple[str, int]]:
    """Return all (aircraft_type, instance_id) from CPP_instances."""
    result = []
    if not CPP_DIR.exists():
        return result
    for at_dir in sorted(CPP_DIR.iterdir()):
        if not at_dir.is_dir():
            continue
        at = at_dir.name.replace("AT_", "")
        for inst_dir in sorted(at_dir.iterdir()):
            if not inst_dir.is_dir():
                continue
            legs_file = inst_dir / "legs"
            if legs_file.exists() and legs_file.stat().st_size > 0:
                inst_id = int(inst_dir.name.replace("instance_", ""))
                result.append((at, inst_id))
    return result


# ── CPPSC_Instances loader (one large instance per type) ─────────────────────

def load_cppsc_instance(
    aircraft_type: str,
    tightness: int = 1,
) -> Dict:
    """
    Load one instance from CPPSC_Instances dataset.

    Args:
        aircraft_type: one of '727','09','94','95','757','319','320'
        tightness:     availability constraint tightness level (1-5)
    """
    at_dir = CPPSC_DIR / aircraft_type
    if not at_dir.exists():
        raise FileNotFoundError(f"{at_dir}")

    airports, bases = _parse_airports(str(at_dir / "airports"))
    legs = _parse_legs(str(at_dir / "legs"), aircraft_type=aircraft_type)

    avail_path = at_dir / f"availability_constraints_{tightness}"
    availability = _parse_availability(str(avail_path)) if avail_path.exists() else {}

    for i, leg in enumerate(legs):
        leg["flight_id"] = i + 1  # 1-based: avoids -0 == 0 deadhead encoding bug

    return {
        "aircraft_type": aircraft_type,
        "instance_id":   tightness,        # tightness level used as instance_id
        "source":        "CPPSC",
        "airports":      airports,
        "bases":         bases,
        "legs":          legs,
        "availability":  availability,
    }


def discover_cppsc_instances() -> List[Tuple[str, int]]:
    """Return all (aircraft_type, tightness) from CPPSC_Instances."""
    result = []
    if not CPPSC_DIR.exists():
        return result
    for at_dir in sorted(CPPSC_DIR.iterdir()):
        if not at_dir.is_dir():
            continue
        at = at_dir.name   # e.g. "320" (no "AT_" prefix in CPPSC)
        legs_file = at_dir / "legs"
        if legs_file.exists() and legs_file.stat().st_size > 0:
            for t in range(1, 6):
                avail = at_dir / f"availability_constraints_{t}"
                if avail.exists():
                    result.append((at, t))
    return result


def discover_all_instances() -> List[Tuple[str, int, str]]:
    """
    Return all instances from both datasets.
    Returns list of (aircraft_type, instance_id, source) tuples.
    source: 'CPP' or 'CPPSC'
    """
    result = []
    for at, iid in discover_instances():
        result.append((at, iid, "CPP"))
    for at, t in discover_cppsc_instances():
        result.append((at, t, "CPPSC"))
    return result
