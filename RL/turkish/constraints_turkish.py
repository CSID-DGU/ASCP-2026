# constraints_turkish.py -- Turkish-specific constraint definitions
# base_ids는 episode base 후보 집합이며 각 pairing은 선택된 동일 base로 복귀함.

from airline_constraints.turkish import TURKISH_CONSTRAINTS


def get_turkish_constraints(base_airport: int, base_ids=None):
    """Return the Turkish Airlines (THY) constraint dict.

    base_airport: the base airport ID where the pairing starts for this episode (either HB1 or HB2)
    base_ids: episode별 base 선택에 사용하는 HB1/HB2 후보 ID 목록.
        pairing 시작과 종료는 선택된 base_airport로 고정됨.
    base_airport/base_ids are excluded from the FiLM input (categorical -- not in FILM_CONSTRAINT_KEYS)
    """
    c = {**TURKISH_CONSTRAINTS, "base_airport": base_airport}
    if base_ids is not None:
        c["base_ids"] = list(base_ids)
    return c


# FiLM input constraint key order -- must match FILM_CONSTRAINT_KEYS in RL/constraints.py
# (kept for checkpoint compatibility -- not redefined here, imported directly from the original)
from constraints import FILM_CONSTRAINT_KEYS  # noqa: E402,F401
