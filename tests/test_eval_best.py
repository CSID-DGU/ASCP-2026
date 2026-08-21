"""
tests/test_eval_best.py -- evaluation/eval_best.py::parse_metrics()/dominates() 테스트 (F1 C3)

eval_best.py는 evaluate_ip.py를 서브프로세스로 부르는 wrapper라서 별도 validator를
갖고 있지 않다 -- 대신 evaluate_ip.py의 stdout에서 independent validator가 찾은
invalid selected pairing 수를 파싱해서, dominates() 판단에 correctness를 성능보다
우선하도록 반영한다.
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from evaluation.eval_best import parse_metrics, dominates  # noqa: E402


SAMPLE_OUTPUT = """
  dead time (within-duty gaps only, excl. overnight): 12.50h
  deadhead:          3 legs
  ManDays:           42
  FTC:               8.25%

[validator] independent re-check: 0 invalid selected pairing(s) (validator_version=0.1.0)
"""

SAMPLE_OUTPUT_WITH_INVALID = SAMPLE_OUTPUT.replace(
    "independent re-check: 0 invalid", "independent re-check: 2 invalid"
)


def test_parse_metrics_extracts_invalid_selected_count():
    m = parse_metrics(SAMPLE_OUTPUT)
    assert m["dead_time"] == 12.50
    assert m["deadhead"] == 3
    assert m["mandays"] == 42
    assert m["ftc"] == 8.25
    assert m["invalid_selected"] == 0


def test_parse_metrics_extracts_nonzero_invalid_selected_count():
    m = parse_metrics(SAMPLE_OUTPUT_WITH_INVALID)
    assert m["invalid_selected"] == 2


def test_parse_metrics_missing_validator_line_is_none():
    m = parse_metrics("dead time (within-duty gaps only, excl. overnight): 1.0h\ndeadhead: 0 legs")
    assert m["invalid_selected"] is None


def test_dominates_ignores_invalid_selected_when_both_zero():
    a = dict(dead_time=1.0, deadhead=0, invalid_selected=0)
    b = dict(dead_time=2.0, deadhead=1, invalid_selected=0)
    assert dominates(a, b)


def test_dominates_never_prefers_side_with_invalid_selected():
    # a는 dead_time/deadhead 둘 다 b보다 좋지만, invalid selected pairing이 있음 --
    # b는 invalid 없음 -- a가 b를 dominate한다고 하면 안 됨(correctness 우선)
    a = dict(dead_time=1.0, deadhead=0, invalid_selected=2)
    b = dict(dead_time=5.0, deadhead=3, invalid_selected=0)
    assert not dominates(a, b)


if __name__ == "__main__":
    test_parse_metrics_extracts_invalid_selected_count()
    test_parse_metrics_extracts_nonzero_invalid_selected_count()
    test_parse_metrics_missing_validator_line_is_none()
    test_dominates_ignores_invalid_selected_when_both_zero()
    test_dominates_never_prefers_side_with_invalid_selected()
    print("OK: 5개 테스트 통과")
