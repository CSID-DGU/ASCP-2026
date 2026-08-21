"""V2 completion 결과를 JSON과 사람이 읽는 표로 변환함."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List


EXPECTED_STAGES = ("policy", "salvage", "rescue", "operational", "artificial")


def build_completion_report(stage_results: Iterable[Dict], all_flight_ids: Iterable[int]) -> Dict:
    stages = list(stage_results)
    names = tuple(stage.get("stage") for stage in stages)
    if names != EXPECTED_STAGES:
        raise ValueError(f"completion stage 순서가 잘못됨: {names}")
    universe = tuple(all_flight_ids)
    candidate_coverages = [stage["candidate_coverage"] for stage in stages]
    if any(right + 1e-12 < left for left, right in zip(candidate_coverages, candidate_coverages[1:])):
        raise ValueError("candidate coverage가 stage 진행 중 감소함")

    compact_stages: List[Dict] = []
    for stage in stages:
        compact_stages.append({
            "stage": stage["stage"],
            "status": stage["status"],
            "is_feasible": stage["is_feasible"],
            "candidate_coverage": stage["candidate_coverage"],
            "legal_coverage": stage["coverage"],
            "operational_completion_coverage": stage["operational_completion_coverage"],
            "completion_coverage": stage["completion_coverage"],
            "candidate_uncovered_flight_ids": stage["candidate_uncovered_flight_ids"],
            "uncovered_flight_ids": stage["uncovered_flight_ids"],
            "selected_count_by_source": stage["selected_count_by_source"],
            "selected_cost_by_source": stage["selected_cost_by_source"],
            "objective_breakdown": stage["objective_breakdown"],
            "mip_objective": stage["mip_objective"],
            "reposition_flight_ids": stage["reposition_flight_ids"],
            "reserve_flight_ids": stage["reserve_flight_ids"],
            "artificial_flight_ids": stage["artificial_flight_ids"],
        })

    final = stages[-1]
    return {
        "schema_version": "v2-completion-1.0",
        "n_flights": len(universe),
        "all_flight_ids": list(universe),
        "direct_candidate_coverage": stages[0]["candidate_coverage"],
        "salvage_assisted_candidate_coverage": stages[1]["candidate_coverage"],
        "post_rescue_candidate_coverage": stages[2]["candidate_coverage"],
        "operational_completion_coverage": stages[3]["operational_completion_coverage"],
        "completion_coverage": final["completion_coverage"],
        "artificial_count": final["artificial_count"],
        "artificial_flight_ids": final["artificial_flight_ids"],
        "final_uncovered_flight_ids": final["uncovered_flight_ids"],
        "stages": compact_stages,
    }


def render_completion_table(report: Dict) -> str:
    header = "stage | status | candidate% | legal% | operational% | completion% | artificial"
    divider = "--- | --- | ---: | ---: | ---: | ---: | ---:"
    rows = [header, divider]
    for stage in report["stages"]:
        rows.append(
            "{stage} | {status} | {candidate:.2f} | {legal:.2f} | {operational:.2f} | {completion:.2f} | {artificial}".format(
                stage=stage["stage"], status=stage["status"],
                candidate=100 * stage["candidate_coverage"],
                legal=100 * stage["legal_coverage"],
                operational=100 * stage["operational_completion_coverage"],
                completion=100 * stage["completion_coverage"],
                artificial=len(stage["artificial_flight_ids"]),
            )
        )
    return "\n".join(rows)


def save_completion_report(report: Dict, output_path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
