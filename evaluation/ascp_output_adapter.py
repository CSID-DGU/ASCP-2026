"""
evaluation/ascp_output_adapter.py -- 저장된 ASCP 평가 결과(JSON, evaluate_ip.py::save_result_json())를
다시 읽어서 pairing_record 리스트로 변환하는 adapter (F1/V1 산출물 5번, evaluation/llm_adapter.py와 짝을 이룸)
"""

import json
from typing import Dict, List, Union


def load_ascp_output(path: str) -> Dict:
    """저장된 JSON 파일 전체를 읽음(checkpoint/airline/eval_mode/요약 지표/
    validation_report/pairings 전부 포함)."""
    with open(path) as f:
        return json.load(f)


def ascp_output_to_pairing_records(payload_or_path: Union[str, Dict]) -> List[Dict]:
    """저장된 ASCP 결과에서 pairing_record 리스트만 뽑아 validate_pairing()/
    aggregate_by_source()에 바로 넣을 수 있는 형태로 반환

    payload_or_path: load_ascp_output()가 반환한 dict, 또는 그 파일 경로(str) 둘 다 받음.
    save_result_json()이 이미 pairing_record 스키마(legs/source_type/
    duty_break_indices)에 맞춰 저장하므로, 이 함수는 얇은 로더 역할만 한다.
    """
    payload = payload_or_path
    if isinstance(payload_or_path, str):
        payload = load_ascp_output(payload_or_path)
    return payload["pairings"]
