# constraints_turkish.py -- Turkish-specific constraint definitions
# HB1/HB2는 상호 대체 가능한 Turkish home-base 집합으로 처리함.

from airline_constraints.turkish import TURKISH_CONSTRAINTS


def get_turkish_constraints(base_airport: int, base_ids=None):
    """Return the Turkish Airlines (THY) constraint dict.

    base_airport: the base airport ID where the pairing starts for this episode (either HB1 or HB2)
    base_ids: pairing이 시작하거나 종료할 수 있는 HB1/HB2 home-base ID 목록.
        HB1에서 시작해 HB2로 복귀하거나 그 반대인 pairing도 유효함.
    base_airport/base_ids are excluded from the FiLM input (categorical -- not in FILM_CONSTRAINT_KEYS)
    """
    c = {**TURKISH_CONSTRAINTS, "base_airport": base_airport}
    if base_ids is not None:
        c["base_ids"] = list(base_ids)
        c["allow_cross_base_return"] = True
    return c


# FiLM input constraint key order -- must match FILM_CONSTRAINT_KEYS in RL/constraints.py
# (kept for checkpoint compatibility -- not redefined here, imported directly from the original)
from constraints import FILM_CONSTRAINT_KEYS  # noqa: E402,F401
