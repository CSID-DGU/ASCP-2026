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


class LLMOutputParseError(ValueError):
    """LLM 출력이 합의된 pairing 형식을 충족하지 못함."""


_PAIRING_LINE_PATTERN = re.compile(
    r"^\s*Pairing\s+(?P<number>\d+)\s*"
    r"(?:\(base=(?P<base>[A-Za-z0-9]+)\))?\s*:\s*"
    r"\[(?P<ids>[^\]]*)\]\s*$",
    re.IGNORECASE,
)
_UNCOVERED_LINE_PATTERN = re.compile(
    r"^\s*Uncovered\s*:\s*\[(?P<ids>[^\]]*)\]\s*$",
    re.IGNORECASE,
)


def _parse_integer_ids(raw_ids: str, *, line_number: int, label: str) -> List[int]:
    raw_ids = raw_ids.strip()
    if not raw_ids:
        return []
    values = []
    for token in raw_ids.split(","):
        token = token.strip()
        if not token:
            raise LLMOutputParseError(
                f"line {line_number}: {label} ID 목록에 빈 항목이 있음"
            )
        try:
            values.append(int(token))
        except ValueError as exc:
            raise LLMOutputParseError(
                f"line {line_number}: {label} flight ID {token!r}는 정수가 아님"
            ) from exc
    return values


def parse_llm_solution(text: str) -> Tuple[List[Dict], List[int]]:
    """base를 보존하고 잘못된 pairing을 누락하지 않는 엄격 parser."""
    records: List[Dict] = []
    uncovered: List[int] | None = None
    seen_pairing_numbers = set()
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        if re.match(r"^Pairing\b", stripped, re.IGNORECASE):
            match = _PAIRING_LINE_PATTERN.fullmatch(line)
            if match is None:
                raise LLMOutputParseError(
                    f"line {line_number}: Pairing 형식이 잘못됨: {stripped!r}"
                )
            pairing_number = int(match.group("number"))
            if pairing_number in seen_pairing_numbers:
                raise LLMOutputParseError(
                    f"line {line_number}: Pairing 번호 {pairing_number}가 중복됨"
                )
            seen_pairing_numbers.add(pairing_number)
            legs = _parse_integer_ids(
                match.group("ids"), line_number=line_number, label="Pairing"
            )
            if not legs:
                raise LLMOutputParseError(
                    f"line {line_number}: Pairing {pairing_number}의 legs가 비어 있음"
                )
            declared_base = match.group("base")
            records.append({
                "pairing_number": pairing_number,
                "declared_base": declared_base.upper() if declared_base else None,
                "legs": legs,
                "source_type": "policy",
            })
            continue
        if re.match(r"^Uncovered\b", stripped, re.IGNORECASE):
            match = _UNCOVERED_LINE_PATTERN.fullmatch(line)
            if match is None:
                raise LLMOutputParseError(
                    f"line {line_number}: Uncovered 형식이 잘못됨: {stripped!r}"
                )
            if uncovered is not None:
                raise LLMOutputParseError(
                    f"line {line_number}: Uncovered 줄이 두 번 이상 나타남"
                )
            uncovered = _parse_integer_ids(
                match.group("ids"), line_number=line_number, label="Uncovered"
            )
    if not records and uncovered is None:
        raise LLMOutputParseError("Pairing 또는 Uncovered 결과를 찾지 못함")
    return records, uncovered or []


def parse_llm_output(text: str) -> Tuple[List[List[int]], List[int]]:
    """기존 호출부용 legs-only 반환 wrapper."""
    records, uncovered = parse_llm_solution(text)
    return [record["legs"] for record in records], uncovered


def to_pairing_records(pairings: List[List[int]]) -> List[Dict]:
    """LLM이 직접 제시한 pairing들을 policy_direct와 동등하게 취급 -- LLM이 "자기가
    직접 고른 것"이라는 점에서 우리 모델의 policy 출력과 같은 역할이므로
    source_type="policy"로 태깅해서 aggregate_by_source()의 policy_direct
    bucket에 들어가게 한다.
    """
    return [{"legs": legs, "source_type": "policy"} for legs in pairings]


def llm_output_to_pairing_records(text: str) -> List[Dict]:
    """LLM 원시 출력에서 base 정보가 보존된 pairing record를 반환함."""
    records, _ = parse_llm_solution(text)
    return records
