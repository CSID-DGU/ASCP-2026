"""전체 운항편을 명시적으로 다루는 V2 full-flight master.

기존 ``set_partition.solve_set_covering``은 AAAI restricted-master 재현용으로
유지하고, 이 모듈은 candidate가 없는 운항편도 문제에서 제외하지 않는 별도 경로를 제공함.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence, Tuple


LEGAL_SOURCE_TYPES = frozenset({"policy", "salvage", "rescue"})
OPERATIONAL_SOURCE_TYPES = frozenset({"reposition", "reserve"})
SUPPORTED_SOURCE_TYPES = LEGAL_SOURCE_TYPES | OPERATIONAL_SOURCE_TYPES


class FullFlightInputError(ValueError):
    """Full-flight master 입력 계약 위반."""


def validate_master_inputs(
    columns: Sequence[Dict],
    all_flight_ids: Iterable[int],
) -> Tuple[List[Dict], Tuple[int, ...]]:
    """입력 universe와 column을 검증하고 결정적인 순서로 정규화함."""
    universe = tuple(all_flight_ids)
    if len(set(universe)) != len(universe):
        raise FullFlightInputError("all_flight_ids에 중복 ID가 있음")

    universe_set = set(universe)
    normalized: List[Dict] = []
    seen_column_ids = set()

    for index, raw in enumerate(columns):
        column = dict(raw)
        column_id = column.get("column_id", f"column_{index}")
        if column_id in seen_column_ids:
            raise FullFlightInputError(f"중복 column_id: {column_id}")
        seen_column_ids.add(column_id)

        legs = list(column.get("legs", []))
        if not legs:
            raise FullFlightInputError(f"{column_id}: legs가 비어 있음")
        if len(set(legs)) != len(legs):
            raise FullFlightInputError(f"{column_id}: column 내부 중복 flight가 있음")
        unknown = sorted(set(legs) - universe_set)
        if unknown:
            raise FullFlightInputError(f"{column_id}: universe 밖 flight ID {unknown}")

        source_type = column.get("source_type")
        if source_type not in SUPPORTED_SOURCE_TYPES:
            raise FullFlightInputError(f"{column_id}: 지원하지 않는 source_type {source_type!r}")
        if source_type in LEGAL_SOURCE_TYPES and column.get("is_legal") is not True:
            raise FullFlightInputError(f"{column_id}: legal column은 is_legal=True여야 함")

        try:
            cost = float(column["cost"])
        except (KeyError, TypeError, ValueError) as exc:
            raise FullFlightInputError(f"{column_id}: 유효한 cost가 필요함") from exc
        if not math.isfinite(cost) or cost < 0:
            raise FullFlightInputError(f"{column_id}: cost는 0 이상의 유한값이어야 함")

        column.update(column_id=column_id, legs=legs, cost=cost)
        normalized.append(column)

    return normalized, universe

