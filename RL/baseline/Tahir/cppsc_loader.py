"""
CPPSC → ASCP-2026 flight format converter.

Tahir legs: dep_abs / arr_abs in minutes since 2000-01-01.
ASCP-2026:  dep_time / arr_time in hours (float).
"""

import sys
from pathlib import Path
from typing import List, Dict, Tuple

# Tahir repo lives alongside ASCP-2026
TAHIR_DIR = Path(__file__).parent.parent.parent / "Tahir"
sys.path.insert(0, str(TAHIR_DIR))

try:
    from dnn.cpp_loader import load_cppsc_instance, CPPSC_DIR
except ImportError as e:
    raise ImportError(
        f"Cannot import Tahir cpp_loader. Make sure {TAHIR_DIR} exists.\n{e}"
    )


def load_cppsc_flights(
    aircraft_type: str,
    tightness: int = 1,
) -> Tuple[List[Dict], Dict[str, int], List[int]]:
    """
    Load one CPPSC instance as ASCP-2026 flight dicts.

    Args:
        aircraft_type: one of '727','09','94','95','757','319','320'
        tightness:     availability constraint level 1-5

    Returns:
        flights:     list of {id, origin(int), dest(int), dep_time(h), arr_time(h)}
        airport_map: airport_name -> integer id
        base_ids:    list of base airport integer ids
    """
    instance = load_cppsc_instance(aircraft_type, tightness)
    legs = instance["legs"]
    airports = instance["airports"]
    bases = instance["bases"]

    airport_map: Dict[str, int] = {name: i for i, name in enumerate(sorted(set(airports)))}
    base_ids = [airport_map[b] for b in bases if b in airport_map]

    flights = []
    for leg in legs:
        flights.append({
            "id":       leg["flight_id"],
            "origin":   airport_map[leg["origin"]],
            "dest":     airport_map[leg["dest"]],
            "dep_time": leg["dep_abs"] / 60.0,
            "arr_time": leg["arr_abs"] / 60.0,
        })

    return flights, airport_map, base_ids


def get_cppsc_constraints(base_airport_id: int = 0) -> Dict:
    """
    CPPSC-compatible constraints (Tahir Table 5).
      T_BAR_D = 720 min = 12.0 h  (max duty elapsed time)
      T_C_MIN =  30 min =  0.5 h  (min connection)
      T_C_MAX = 240 min =  4.0 h  (max connection)
      F_MAX   =   5                (max legs per duty)

    CPPSC 벤치마크는 multi-day pairing 개념이 없어 min_rest/max_duty_periods/
    max_pairing_days를 정의하지 않지만, FILM_CONSTRAINT_KEYS(7개)는 이 3개를
    포함하므로 constraint_to_tensor()가 KeyError 없이 돌도록 config.DEFAULT_CONSTRAINTS
    기본값을 명시적으로 채워 넣는다 (environment.py의 get_mask/step은 이미 이 기본값으로
    fallback하므로 동작 자체는 바뀌지 않음).
    """
    return {
        "max_duty":          12.0,
        "min_conn":           0.5,
        "max_conn":           4.0,
        "max_legs":             5,
        "base_airport":  base_airport_id,
        "min_rest":          10.0,
        "max_duty_periods":     2,
        "max_pairing_days":     5,
    }
