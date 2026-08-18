"""
find_zeren_date_window.py — Zeren 논문(log/0629/Zeren.pdf) Table 3의 "Feb: 15,738편" 목표에
맞는 tt201402.legs 날짜 경계를 탐색.

결론(중요): 모든 (시작일, 길이) 조합을 전수 탐색해보면 target±50 이내로 맞는 조합은 전부
정확히 35일짜리 윈도우다. 이건 Zeren이 실제로 이 날짜 범위를 썼다는 증거가 아니라, 일평균
편수가 거의 일정(~450편/일)하기 때문에 15,738÷450≈35일이 나오는 산술적 결과일 뿐이다 —
어떤 35일을 골라도 비슷하게 맞아서 "어느 35일인지"는 전혀 특정하지 못한다. 논문이 문자
그대로 서술한 범위("2월 + 다음달 첫 10일" ≈ 39일)로 계산하면 17,075로 전혀 안 맞는다.
→ 날짜 경계 문제는 여전히 미해결.
"""
import sys
from pathlib import Path
from collections import Counter

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from RL.turkish.loader_turkish import parse_legs_dir

TARGET = 15738
DATA_DIR = REPO_ROOT / "RL" / "data" / "timetables"
FILE = "tt201402.legs"


def main():
    df = parse_legs_dir(DATA_DIR, fleet_prefix="3", files=[FILE])
    dates = sorted(df["dep_date_utc"].unique())
    print(f"{FILE} fleet_prefix='3' 필터 후 전체: {len(df)}편, 날짜 범위: {dates[0].date()} ~ {dates[-1].date()}")

    # 1. 논문이 문자 그대로 서술한 후보들
    print("\n=== 논문 서술 기반 후보 ===")
    candidates = [
        ("2/1~2/28 (순수 2월)", "2014-02-01", "2014-03-01"),
        ("2/1~3/10 (2월+다음달10일, 논문 서술)", "2014-02-01", "2014-03-11"),
        ("1/20~2/28 (파일 시작~2월말)", "2014-01-20", "2014-03-01"),
        ("1/20~3/10 (파일 전체 범위)", "2014-01-20", "2014-03-11"),
    ]
    for label, start, end in candidates:
        n = len(df[(df["dep_date_utc"] >= start) & (df["dep_date_utc"] < end)])
        print(f"  {label:38s} n={n:6d}  diff={n - TARGET:+6d}")

    # 2. 전수 탐색 — 모든 (시작일, 길이) 조합
    print("\n=== 전수 탐색 (모든 시작일 x 길이 조합) ===")
    results = []
    for i, start in enumerate(dates):
        for j in range(i + 1, len(dates) + 1):
            end = dates[j - 1] + pd.Timedelta(days=1)
            n = len(df[(df["dep_date_utc"] >= start) & (df["dep_date_utc"] < end)])
            results.append((n, j - i, start, dates[j - 1]))

    results.sort(key=lambda r: abs(r[0] - TARGET))
    print(f"전체 조합 수: {len(results)}")
    print("target에 가장 가까운 10개:")
    for n, ndays, start, end in results[:10]:
        print(f"  n={n:6d} diff={n - TARGET:+6d}  {start.date()} ~ {end.date()}  ({ndays}일)")

    close = [r for r in results if abs(r[0] - TARGET) <= 50]
    length_dist = Counter(r[1] for r in close)
    print(f"\ntarget±50 이내 조합 수: {len(close)}, 윈도우 길이 분포: {length_dist.most_common()}")
    print("→ 전부 같은 길이라면 이는 '일평균 편수가 일정하다'는 산술적 결과이지,")
    print("  Zeren이 실제로 쓴 날짜 범위를 특정한 것이 아님에 유의.")


if __name__ == "__main__":
    main()
