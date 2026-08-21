"""V2 full-flight master의 단계별 동작을 재현하는 소규모 pilot."""

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from evaluation.completion_report import render_completion_table
from evaluation.evaluate_ip import solve_pool_completion


def column(column_id, legs, cost, source_type, is_legal=True, **extra):
    value = {
        "column_id": column_id, "legs": legs, "cost": cost,
        "source_type": source_type, "is_legal": is_legal,
    }
    value.update(extra)
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="result/v2_full_flight_master/pilot.json")
    args = parser.parse_args()

    pool = [
        column("policy-0", [0], 2, "policy"),
        column("salvage-1", [1], 3, "salvage"),
        column("reposition-3", [3], 20, "reposition", is_legal=False),
    ]
    rescue = [
        column("rescue-2", [2], 5, "rescue", repair_target_flights=[2], validator_version="0.1.0", constraint_hash="pilot"),
    ]
    result = solve_pool_completion(
        pool, 6, lambda_excess=1,
        rescue_columns=rescue, report_path=args.output,
    )
    report = result["completion_report"]
    assert report["direct_candidate_coverage"] == 1 / 6
    assert report["salvage_assisted_candidate_coverage"] == 2 / 6
    assert report["post_rescue_candidate_coverage"] == 3 / 6
    assert report["operational_completion_coverage"] == 4 / 6
    assert report["completion_coverage"] == 1.0
    assert report["artificial_flight_ids"] == [4, 5]
    print(render_completion_table(report))
    print(json.dumps({
        "output": args.output,
        "artificial_flight_ids": report["artificial_flight_ids"],
        "completion_coverage": report["completion_coverage"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
