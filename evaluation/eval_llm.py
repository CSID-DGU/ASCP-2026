"""LLM이 제출한 crew-pairing 파일을 수정 없이 평가하는 CLI."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict

import pandas as pd


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "RL"))

import config  # noqa: E402
from constraints import (  # noqa: E402
    get_alaska_constraints,
    get_delta_constraints,
    get_jetblue_constraints,
)
from evaluation.llm_adapter import parse_llm_solution  # noqa: E402
from evaluation.llm_direct import (  # noqa: E402
    build_legacy_forced_100,
    evaluate_direct_solution,
)
from loader import (  # noqa: E402
    bases_to_ids,
    build_airport_map,
    convert_time,
    scheduled_local_datetime,
    utc_offset_hours,
)
from turkish.constraints_turkish import get_turkish_constraints  # noqa: E402
from turkish.loader_turkish import (  # noqa: E402
    ZEREN_FEB_FILE,
    ZEREN_FEB_WINDOW,
    build_airport_map_turkish,
    parse_legs_dir,
)


AIRLINE_PREFIX = {
    "delta": "DL",
    "alaska": "AS",
    "jetblue": "B6",
    "turkish": "TK",
}
_STANDARD_CONSTRAINT_FACTORY = {
    "delta": get_delta_constraints,
    "alaska": get_alaska_constraints,
    "jetblue": get_jetblue_constraints,
}


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def infer_airline(data_path: str) -> str:
    """기존 evaluator처럼 데이터 경로에서 항공사를 판별함."""
    if os.path.isdir(data_path) or data_path.lower().endswith(".legs"):
        return "turkish"
    name = os.path.basename(data_path).lower()
    for airline in ("jetblue", "alaska", "delta"):
        if airline in name:
            return airline
    return "delta"


def replace_symbolic_flight_ids(text: str, prefix: str) -> str:
    """DL00001 형식의 ID를 보존된 1-based 원본 행 ID로 변환함."""
    pattern = re.compile(rf"\b{re.escape(prefix)}(?P<number>\d{{5}})\b", re.IGNORECASE)
    return pattern.sub(lambda match: str(int(match.group("number"))), text)


def load_bts_instance(csv_path: str) -> Dict:
    """BTS 원본 행 ID를 보존하며 공통 UTC 시간 기준을 적용함."""
    raw = pd.read_csv(csv_path).reset_index(drop=True)
    raw["__flight_id"] = raw.index + 1
    required = [
        "ORIGIN",
        "DEST",
        "CRS_DEP_TIME",
        "CRS_ARR_TIME",
        "CRS_ELAPSED_TIME",
        "FL_DATE",
    ]
    frame = raw[required + ["__flight_id"]].dropna().copy()
    frame["FL_DATE"] = pd.to_datetime(frame["FL_DATE"], format="mixed")
    base_date = frame["FL_DATE"].min()
    airport_map = build_airport_map(csv_path)

    flights = {}
    for _, row in frame.iterrows():
        flight_id = int(row["__flight_id"])
        day_offset = (row["FL_DATE"] - base_date).days
        departure = convert_time(row["CRS_DEP_TIME"]) + day_offset * 24.0
        departure -= utc_offset_hours(
            row["ORIGIN"],
            scheduled_local_datetime(row["FL_DATE"], row["CRS_DEP_TIME"]),
        )
        arrival = departure + float(row["CRS_ELAPSED_TIME"]) / 60.0
        flights[flight_id] = {
            "id": flight_id,
            "origin": int(airport_map[row["ORIGIN"]]),
            "dest": int(airport_map[row["DEST"]]),
            "dep_time": float(departure),
            "arr_time": float(arrival),
        }
    return {
        "flights": flights,
        "airport_map": airport_map,
        "metadata": {
            "data_path": csv_path,
            "data_sha256": _sha256_file(csv_path),
            "raw_rows": len(raw),
            "evaluated_rows": len(frame),
            "time_basis": "utc",
            "flight_id_basis": "raw-row-1-based",
        },
    }


def load_turkish_instance(data_path: str) -> Dict:
    """Turkish 파일은 지정 파일 또는 공통 Zeren February 범위를 읽음."""
    if os.path.isfile(data_path):
        directory = os.path.dirname(data_path) or "."
        selected_files = [os.path.basename(data_path)]
        date_range = None
    else:
        directory = data_path
        selected_files = [ZEREN_FEB_FILE]
        date_range = ZEREN_FEB_WINDOW

    frame = parse_legs_dir(
        directory,
        files=selected_files,
        date_range=date_range,
    ).sort_values("dep_utc").reset_index(drop=True)
    airport_map = build_airport_map_turkish(df=frame)
    base_time = frame["dep_utc"].min()
    flights = {}
    for index, row in frame.iterrows():
        flight_id = index + 1
        flights[flight_id] = {
            "id": flight_id,
            "origin": int(airport_map[row["ORIGIN"]]),
            "dest": int(airport_map[row["DEST"]]),
            "dep_time": float((row["dep_utc"] - base_time).total_seconds() / 3600.0),
            "arr_time": float((row["arr_utc"] - base_time).total_seconds() / 3600.0),
        }
    hashed_files = [os.path.join(directory, filename) for filename in selected_files]
    return {
        "flights": flights,
        "airport_map": airport_map,
        "metadata": {
            "data_path": data_path,
            "files": selected_files,
            "file_sha256": {path: _sha256_file(path) for path in hashed_files},
            "date_range": list(date_range) if date_range else None,
            "evaluated_rows": len(frame),
            "time_basis": "turkish-native-utc",
            "flight_id_basis": "sorted-1-based",
        },
    }


def load_evaluation_instance(airline: str, data_path: str) -> Dict:
    loaded = (
        load_turkish_instance(data_path)
        if airline == "turkish"
        else load_bts_instance(data_path)
    )
    airport_map = loaded["airport_map"]
    base_ids = bases_to_ids(config.AIRLINE_BASES[airline], airport_map)
    if airline == "turkish":
        constraint = get_turkish_constraints(base_ids[0], base_ids=base_ids)
    else:
        constraint = _STANDARD_CONSTRAINT_FACTORY[airline](base_ids[0])
    loaded.update(base_ids=base_ids, constraint=constraint)
    return loaded


def evaluate_llm_text(text: str, instance: Dict, *, airline: str) -> Dict:
    """LLM pairing을 direct 평가하고 forced-100은 항상 별도 블록으로 계산함."""
    converted = replace_symbolic_flight_ids(text, AIRLINE_PREFIX[airline])
    pairing_records, declared_uncovered = parse_llm_solution(converted)
    direct = evaluate_direct_solution(
        pairing_records,
        declared_uncovered,
        instance["flights"],
        instance["constraint"],
        instance["base_ids"],
        instance["airport_map"],
    )
    return {
        **direct,
        "airline": airline,
        "input": instance["metadata"],
        "llm_output_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "legacy_forced_100": build_legacy_forced_100(direct, instance["flights"]),
    }


def render_summary(result: Dict) -> str:
    forced = result["legacy_forced_100"]
    return "\n".join([
        "LLM direct evaluation",
        f"solution_feasible: {result['solution_feasible']}",
        f"legal_union_coverage: {result['legal_union_coverage'] * 100:.2f}%",
        f"conflict_free_legal_coverage: {result['conflict_free_legal_coverage'] * 100:.2f}%",
        f"submitted/legal/invalid pairings: {result['n_submitted_pairings']}/"
        f"{result['n_legal_pairings']}/{result['n_invalid_pairings']}",
        f"declared_uncovered: {len(result['declared_uncovered_flight_ids'])}",
        f"legally_uncovered: {len(result['legally_uncovered_flight_ids'])}",
        f"duplicates: {len(result['duplicate_flight_ids'])}",
        f"legacy_forced_100: {forced['synthetic_completion_coverage'] * 100:.2f}% "
        f"({forced['n_forced_pairings']} forced)",
    ])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="LLM pairing 파일을 선택·보완하지 않고 직접 평가"
    )
    parser.add_argument("llm_output")
    parser.add_argument("data_path")
    args = parser.parse_args()

    airline = infer_airline(args.data_path)
    instance = load_evaluation_instance(airline, args.data_path)
    text = Path(args.llm_output).read_text(encoding="utf-8")
    result = evaluate_llm_text(text, instance, airline=airline)
    print(render_summary(result))
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
