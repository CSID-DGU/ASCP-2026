"""Policy부터 artificial까지 누적 실행하는 V2 completion runner."""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, Iterable, List, Sequence

from evaluation.full_flight_master import FullFlightInputError, calibrate_completion_penalties, solve_full_flight_master, validate_master_inputs



def merge_rescue_columns(
    columns: Sequence[Dict], rescue_columns: Sequence[Dict], all_flight_ids: Iterable[int]
) -> List[Dict]:
    """찬주 generator의 rescue column을 계약 검증 후 기존 pool에 병합함."""
    universe = tuple(all_flight_ids)
    merged = [dict(column) for column in columns]
    seen_legs = {tuple(column.get("legs", [])) for column in merged}
    for index, raw in enumerate(rescue_columns):
        rescue = dict(raw)
        rescue.setdefault("column_id", f"rescue-{index}")
        if rescue.get("source_type") != "rescue":
            raise FullFlightInputError(f"{rescue['column_id']}: rescue source_type이 필요함")
        targets = set(rescue.get("repair_target_flights", []))
        if not targets:
            raise FullFlightInputError(f"{rescue['column_id']}: repair_target_flights가 비어 있음")
        if not targets.issubset(set(rescue.get("legs", []))):
            raise FullFlightInputError(f"{rescue['column_id']}: target flight가 legs에 없음")
        for field in ("validator_version", "constraint_hash"):
            if not rescue.get(field):
                raise FullFlightInputError(f"{rescue['column_id']}: {field}가 필요함")
        key = tuple(rescue.get("legs", []))
        if key in seen_legs:
            continue
        merged.append(rescue)
        seen_legs.add(key)
    validate_master_inputs(merged, universe)
    return merged

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
    calibrated = calibrate_completion_penalties(columns)
    options = dict(master_options)
    # 실제 운항 eligibility 자료가 없는 실행에서 모든 미커버 flight를 임의로
    # reposition/reserve 가능하다고 간주하지 않음. 명시적 opt-in일 때만 proxy 사용.
    auto_operational = bool(options.pop("auto_operational", False))
    for key, value in calibrated.items():
        if options.get(key) is None:
            options[key] = value
    results = []
    previous_candidate_covered = set()
    previous_solve_signature = None
    previous_solve_result = None

    for stage_name, sources, allow_reposition, allow_reserve, allow_artificial in STAGES:
        stage_columns = [column for column in columns if column.get("source_type") in sources]
        candidate_covered = {
            flight_id for column in stage_columns for flight_id in column.get("legs", [])
            if flight_id in universe_set
        }
        if not previous_candidate_covered.issubset(candidate_covered):
            raise RuntimeError("누적 stage candidate coverage가 감소함")
        previous_candidate_covered = candidate_covered
        candidate_uncovered = universe_set - candidate_covered

        # 실제 eligibility 입력이 있거나 proxy 사용을 명시한 경우에만 operational
        # 변수를 엶. 기본 실행에서는 남은 flight를 artificial로 투명하게 보고함.
        stage_options = dict(options)
        if auto_operational and allow_reposition and stage_options.get("reposition_flight_ids") is None:
            stage_options["reposition_flight_ids"] = sorted(candidate_uncovered)
        if auto_operational and allow_reserve and stage_options.get("reserve_flight_ids") is None:
            stage_options["reserve_flight_ids"] = sorted(candidate_uncovered)

        reposition_targets = tuple(sorted(stage_options.get("reposition_flight_ids") or ()))
        reserve_targets = tuple(sorted(stage_options.get("reserve_flight_ids") or ()))
        solve_signature = (
            tuple(column.get("column_id") for column in stage_columns),
            reposition_targets if allow_reposition else (),
            reserve_targets if allow_reserve else (),
            bool(allow_artificial),
        )
        if solve_signature == previous_solve_signature:
            result = deepcopy(previous_solve_result)
            result["solve_reused"] = True
            result["reused_from_stage"] = results[-1]["stage"]
        else:
            result = solve_full_flight_master(
                stage_columns,
                universe,
                allow_reposition=allow_reposition,
                allow_reserve=allow_reserve,
                allow_artificial=allow_artificial,
                **stage_options,
            )
            result["solve_reused"] = False
            result["reused_from_stage"] = None
            previous_solve_signature = solve_signature
            previous_solve_result = deepcopy(result)
        result["stage"] = stage_name
        result["allowed_sources"] = sorted(sources)
        result["candidate_covered_flight_ids"] = sorted(candidate_covered)
        result["candidate_uncovered_flight_ids"] = sorted(candidate_uncovered)
        result["candidate_coverage"] = (
            len(candidate_covered) / len(universe) if universe else 1.0
        )
        result["operational_stage_has_inputs"] = bool(
            stage_name == "operational"
            and (
                any(c.get("source_type") in {"reposition", "reserve"} for c in stage_columns)
                or stage_options.get("reposition_flight_ids")
                or stage_options.get("reserve_flight_ids")
            )
        )
        result["auto_operational"] = auto_operational
        results.append(result)

    return results
