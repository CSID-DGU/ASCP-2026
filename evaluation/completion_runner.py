"""Policy부터 artificial까지 누적 실행하는 V2 completion runner."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

from evaluation.full_flight_master import solve_full_flight_master


STAGES = (
    ("policy", {"policy"}, False, False, False),
    ("salvage", {"policy", "salvage"}, False, False, False),
    ("rescue", {"policy", "salvage", "rescue"}, False, False, False),
    ("operational", {"policy", "salvage", "rescue", "reposition", "reserve"}, True, True, False),
    ("artificial", {"policy", "salvage", "rescue", "reposition", "reserve"}, True, True, True),
)


def solve_completion_stages(
    columns: Sequence[Dict],
    all_flight_ids: Iterable[int],
    **master_options,
) -> List[Dict]:
    """동일 universe에서 허용 source와 completion 수단을 누적하며 실행함."""
    universe = tuple(all_flight_ids)
    universe_set = set(universe)
    results = []
    previous_candidate_covered = set()

    for stage_name, sources, allow_reposition, allow_reserve, allow_artificial in STAGES:
        stage_columns = [column for column in columns if column.get("source_type") in sources]
        candidate_covered = {
            flight_id for column in stage_columns for flight_id in column.get("legs", [])
            if flight_id in universe_set
        }
        if not previous_candidate_covered.issubset(candidate_covered):
            raise RuntimeError("누적 stage candidate coverage가 감소함")
        previous_candidate_covered = candidate_covered

        result = solve_full_flight_master(
            stage_columns,
            universe,
            allow_reposition=allow_reposition,
            allow_reserve=allow_reserve,
            allow_artificial=allow_artificial,
            **master_options,
        )
        result["stage"] = stage_name
        result["allowed_sources"] = sorted(sources)
        result["candidate_covered_flight_ids"] = sorted(candidate_covered)
        result["candidate_uncovered_flight_ids"] = sorted(universe_set - candidate_covered)
        result["candidate_coverage"] = (
            len(candidate_covered) / len(universe) if universe else 1.0
        )
        results.append(result)

    return results
