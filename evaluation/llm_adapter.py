"""
evaluation/llm_adapter.py -- LLM(GPT/Claude 등) 원시 출력을 validator가 먹을 수 있는
pairing_record로 변환하는 adapter (F1/V1 C4)

기존 eval_llm.py의 parser/metric 계산 로직을 그대로 갖다 쓰지 않는다 -- 그쪽의
"valid only" 판정이 우리 strict validator보다 느슨한 걸 이미 확인했기 때문에(base
복귀·duty 간 연속성을 안 걸러냄, duplicate는 기록만 함). 그래서 여기서는 파싱만
독립적으로 다시 구현하고, validity 판정은 전부 evaluation/validator.py를 거치게 한다.

LLM 출력 형식 (기존 eval_llm.py와 동일하게 파싱):
    Pairing 1 (base=ATL): [1, 23, 45]
    Pairing 2 (base=SLC): [7, 88]
    ...
    Uncovered: [12, 34, ...]
"""

import re
from typing import Dict, List, Tuple

_PAIRING_PATTERN = re.compile(
    r"Pairing\s+\d+\s*(?:\(base=\w+\))?\s*:\s*\[([^\]]*)\]",
    re.IGNORECASE,
)
_UNCOVERED_PATTERN = re.compile(r"Uncovered\s*:\s*\[([^\]]*)\]", re.IGNORECASE)


def parse_llm_output(text: str) -> Tuple[List[List[int]], List[int]]:
    """LLM 원시 텍스트에서 pairing별 flight ID 리스트와 uncovered 리스트를 뽑아냄.

    파싱 형식은 기존 eval_llm.py와 동일 -- 다만 validity 판정은 여기서 전혀 안 함,
    순수 파싱만. 판정은 evaluation/validator.py::validate_pairing()에 맡긴다.
    """
    pairings: List[List[int]] = []
    for m in _PAIRING_PATTERN.finditer(text):
        ids_str = m.group(1).strip()
        if not ids_str:
            continue
        try:
            ids = [int(x.strip()) for x in ids_str.split(",") if x.strip()]
        except ValueError:
            continue
        if ids:
            pairings.append(ids)

    uncovered: List[int] = []
    m = _UNCOVERED_PATTERN.search(text)
    if m:
        ids_str = m.group(1).strip()
        if ids_str:
            try:
                uncovered = [int(x.strip()) for x in ids_str.split(",") if x.strip()]
            except ValueError:
                pass

    return pairings, uncovered


def to_pairing_records(pairings: List[List[int]]) -> List[Dict]:
    """LLM이 직접 제시한 pairing들을 policy_direct와 동등하게 취급 -- LLM이 "자기가
    직접 고른 것"이라는 점에서 우리 모델의 policy 출력과 같은 역할이므로
    source_type="policy"로 태깅해서 aggregate_by_source()의 policy_direct
    bucket에 들어가게 한다.
    """
    return [{"legs": legs, "source_type": "policy"} for legs in pairings]


def forced_singleton_records(uncovered_flight_ids: List[int]) -> List[Dict]:
    """LLM이 못 커버한 flight마다 1-leg "forced" pairing을 만듦(기존 baseline의
    "forced 100" 완성 방식과 동일 구성). source_type="forced"로 명시 태깅해서
    legal direct coverage에 섞이지 않게 한다(v1.md C4 "forced 100은 source_type=
    forced로만 기록하고 legal direct coverage에 포함 금지"). 이 pairing들은 대부분
    base 미복귀라 validate_pairing()으로 검증하면 invalid로 나올 텐데, 그건 의도된
    결과다 -- 실제 legal한 pairing이 아니라 "억지로 채운 것"이므로.
    """
    return [{"legs": [fid], "source_type": "forced"} for fid in uncovered_flight_ids]


def llm_output_to_pairing_records(text: str, include_forced_completion: bool = False) -> List[Dict]:
    """LLM 원시 텍스트 -> pairing_record 리스트 (validator/validation_report에 바로
    넣을 수 있는 형태).

    include_forced_completion=True면 uncovered flight의 forced singleton도 같이
    포함(source_type="forced"로 구분되니 aggregate_by_source()에서 자동으로 policy_direct
    와 분리됨).
    """
    pairings, uncovered = parse_llm_output(text)
    records = to_pairing_records(pairings)
    if include_forced_completion:
        records += forced_singleton_records(uncovered)
    return records
