"""
completion/rescue_pool_io.py -- rescue candidate pool을 --rescue-pool-path 형식으로 저장

evaluation/evaluate_ip.py의 로더(evaluate_full() 안, --rescue-pool-path 처리 부분)는
JSON을 읽어서 dict면 "columns" 키, 없으면 "rescue_columns" 키를 순서대로 찾는다
(evaluation/*.py는 혜린 담당이라 직접 수정하지 않고, 그 계약에 맞춰 저장만 한다).
여기서는 "rescue_columns" 키를 쓴다 -- rescue 전용 candidate만 담기 때문에 이름을
명확히 하는 게 낫다고 판단함.
"""

import json
import os
from typing import Dict, List, Optional


def save_rescue_pool_json(
    path: str,
    candidates: List[Dict],
    failures: Optional[Dict[int, str]] = None,
) -> None:
    """generate_rescue_candidates()의 결과를 evaluate_ip.py --rescue-pool-path가
    읽을 수 있는 형식으로 저장한다. failures는 evaluate_ip.py가 쓰지 않는 필드지만,
    "실패 target 누락 0"(v2-chanju.md §7)을 나중에 감사하기 위해 함께 남겨둔다.
    """
    payload = {
        "rescue_columns": candidates,
        "n_candidates": len(candidates),
        "failures": failures or {},
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)


def load_rescue_pool_json(path: str) -> List[Dict]:
    """저장된 rescue pool을 다시 읽는다 -- evaluate_ip.py와 동일한 dict/list 겸용 규칙
    (columns 우선, 그다음 rescue_columns)을 그대로 따라서 왕복(save->load) 일관성을 보장.
    """
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        return payload.get("columns", payload.get("rescue_columns", []))
    return payload
