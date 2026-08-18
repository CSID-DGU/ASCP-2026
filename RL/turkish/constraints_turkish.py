# constraints_turkish.py -- Turkish-specific constraint definitions (allows asymmetric HB1/HB2 termination)
#
# Why this file exists instead of using RL/constraints.py's get_turkish_constraints() directly:
# HB1/HB2 are both in the same city (Istanbul) and are effectively the same home, so a pairing
# that starts at one and ends at the other should incur no penalty (see field 1 description in
# RL/data/timetables/ttfields.txt). For delta and others the bases are in different cities, so
# this logic must not apply there -- hence the original RL/constraints.py is left untouched, and
# the base_ids (set-based) logic is kept separate here for turkish only.

from airline_constraints.turkish import TURKISH_CONSTRAINTS


def get_turkish_constraints(base_airport: int, base_ids=None):
    """Return the Turkish Airlines (THY) constraint dict.

    base_airport: the base airport ID where the pairing starts for this episode (either HB1 or HB2)
    base_ids: full list of HB1/HB2 IDs. When given, environment_turkish.py accepts any base in
        base_ids for pairing end/restart (allows HB1<->HB2 asymmetry).
        If None, only base_airport itself is treated as a valid base, as before.
    base_airport/base_ids are excluded from the FiLM input (categorical -- not in FILM_CONSTRAINT_KEYS)
    """
    c = {**TURKISH_CONSTRAINTS, "base_airport": base_airport}
    if base_ids is not None:
        c["base_ids"] = list(base_ids)
    return c


# FiLM input constraint key order -- must match FILM_CONSTRAINT_KEYS in RL/constraints.py
# (kept for checkpoint compatibility -- not redefined here, imported directly from the original)
from constraints import FILM_CONSTRAINT_KEYS  # noqa: E402,F401
