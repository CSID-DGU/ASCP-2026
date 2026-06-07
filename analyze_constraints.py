"""
BTS 데이터에서 항공사별 운항 constraint 추정치 도출.

분석 방법:
  - TAIL_NUM(항공기 등록번호)을 crew 이동 proxy로 사용
  - 동일 항공기의 연속 편에서 실제 운항 패턴 역산
  - crew가 항공기와 함께 이동한다는 가정 (domestic 운항 기준 타당)

추정 대상:
  - min_conn : 동일 duty 내 최소 연결 시간 (5th percentile)
  - max_conn : 동일 duty 내 최대 연결 시간 (95th percentile)
  - max_legs : duty당 최대 비행 편수 (95th percentile)
  - max_duty_periods : pairing당 최대 duty 수 (95th percentile)
  - max_pairing_days : 최대 연속 운항 일수 (95th percentile)
"""

import pandas as pd
import numpy as np

DATA_PATH = "RL/data/T_ONTIME_MARKETING.csv"

AIRLINES = {
    "delta":     "DL",
    "alaska":    "AS",
    "jetblue":   "B6",
    "southwest": "WN",
}

# FAA Part 117 기준: 이 시간 이상이면 overnight rest → duty 경계
REST_THRESHOLD_H = 9.0   # hours — 이 이상 gap이면 새 duty로 간주


def hhmm_to_hours(hhmm):
    hhmm = int(hhmm)
    return (hhmm // 100) + (hhmm % 100) / 60


def load_airline(carrier_code):
    df = pd.read_csv(DATA_PATH, usecols=[
        "FL_DATE", "OP_UNIQUE_CARRIER", "TAIL_NUM",
        "ORIGIN", "DEST", "CRS_DEP_TIME", "CRS_ARR_TIME", "CRS_ELAPSED_TIME",
    ])
    df = df[df["OP_UNIQUE_CARRIER"] == carrier_code].copy()
    df = df.dropna(subset=["TAIL_NUM", "CRS_DEP_TIME", "CRS_ARR_TIME", "CRS_ELAPSED_TIME"])
    df["FL_DATE"] = pd.to_datetime(df["FL_DATE"], format="mixed")

    base_date = df["FL_DATE"].min()
    df["day_offset"] = (df["FL_DATE"] - base_date).dt.days

    # 절대 시간 (hours, base_date 기준)
    df["dep_h"] = df["CRS_DEP_TIME"].apply(hhmm_to_hours) + df["day_offset"] * 24
    df["elapsed_h"] = df["CRS_ELAPSED_TIME"] / 60.0
    df["arr_h"] = df["dep_h"] + df["elapsed_h"]

    return df.sort_values(["TAIL_NUM", "dep_h"]).reset_index(drop=True)


def analyze(df, airline_name):
    print(f"\n{'='*55}")
    print(f"  {airline_name.upper()}")
    print(f"  총 {len(df):,}편  |  항공기 {df['TAIL_NUM'].nunique():,}대")
    print(f"{'='*55}")

    conn_times   = []   # 동일 duty 내 연결 시간 (h)
    legs_per_day = []   # 항공기 1대 하루 비행 편수
    duty_counts  = []   # pairing당 duty 수
    pairing_days = []   # 연속 운항 일수 (pairing 길이)

    for tail, grp in df.groupby("TAIL_NUM"):
        grp = grp.sort_values("dep_h").reset_index(drop=True)
        if len(grp) < 2:
            continue

        # ── 연결 시간 & duty/pairing 구조 ────────────────────────
        current_duty_legs   = 1
        current_duty_start  = grp.loc[0, "dep_h"]
        current_duty_legs_list = [1]
        duties_in_pairing   = 1
        pairing_start_day   = int(grp.loc[0, "dep_h"] // 24)
        prev_airport        = grp.loc[0, "DEST"]
        prev_arr            = grp.loc[0, "arr_h"]

        for i in range(1, len(grp)):
            row  = grp.loc[i]
            gap  = row["dep_h"] - prev_arr          # 연결 시간 (h)
            same_airport = (row["ORIGIN"] == prev_airport)

            if gap >= REST_THRESHOLD_H:
                # duty 경계
                current_duty_legs_list.append(current_duty_legs)
                current_duty_legs = 1

                if gap >= 24.0:
                    # 24h 이상 → pairing 경계 (항공기 비번)
                    duty_counts.append(duties_in_pairing)
                    pairing_end_day = int(prev_arr // 24)
                    pairing_days.append(max(pairing_end_day - pairing_start_day + 1, 1))
                    duties_in_pairing  = 1
                    pairing_start_day  = int(row["dep_h"] // 24)
                else:
                    # overnight rest → 새 duty, 같은 pairing
                    duties_in_pairing += 1
                    if same_airport and 0 < gap < REST_THRESHOLD_H * 2:
                        conn_times.append(gap)

            else:
                # 같은 duty 내 연결
                current_duty_legs += 1
                if same_airport and gap > 0:
                    conn_times.append(gap)

            prev_arr      = row["arr_h"]
            prev_airport  = row["DEST"]

        # 마지막 duty/pairing flush
        current_duty_legs_list.append(current_duty_legs)
        duty_counts.append(duties_in_pairing)
        pairing_end_day = int(prev_arr // 24)
        pairing_days.append(max(pairing_end_day - pairing_start_day + 1, 1))

        # legs per day: 항공기별 날짜별 편수
        for date, dgrp in grp.groupby(grp["dep_h"].apply(lambda x: int(x // 24))):
            legs_per_day.append(len(dgrp))

    # ── 결과 출력 ─────────────────────────────────────────────
    def fmt(arr, name, unit="h"):
        arr = np.array(arr)
        print(f"  {name:20s}  "
              f"p5={np.percentile(arr,5):.2f}{unit}  "
              f"p50={np.percentile(arr,50):.2f}{unit}  "
              f"p95={np.percentile(arr,95):.2f}{unit}  "
              f"mean={arr.mean():.2f}{unit}  "
              f"max={arr.max():.2f}{unit}  "
              f"n={len(arr):,}")

    if conn_times:
        fmt(conn_times,  "connection time", "h")
    if legs_per_day:
        fmt(legs_per_day, "legs / day",     " legs")
    if duty_counts:
        fmt(duty_counts,  "duties / pairing"," duties")
    if pairing_days:
        fmt(pairing_days, "pairing days",   " days")

    # ── 추천 constraint 값 ────────────────────────────────────
    print(f"\n  [ 추천 설정값 (논문 표기용 추정치) ]")
    if conn_times:
        ct = np.array(conn_times)
        print(f"  min_conn         ≈ {np.percentile(ct, 5):.1f}h  (p5)")
        print(f"  max_conn         ≈ {np.percentile(ct, 95):.1f}h  (p95, duty 내 연결만)")
    if legs_per_day:
        lp = np.array(legs_per_day)
        print(f"  max_legs         ≈ {int(np.percentile(lp, 95))} legs/day  (p95)")
    if duty_counts:
        dc = np.array(duty_counts)
        print(f"  max_duty_periods ≈ {int(np.percentile(dc, 95))} duties  (p95)")
    if pairing_days:
        pd_ = np.array(pairing_days)
        print(f"  max_pairing_days ≈ {int(np.percentile(pd_, 95))} days  (p95)")


if __name__ == "__main__":
    for name, code in AIRLINES.items():
        df = load_airline(code)
        if df.empty:
            print(f"\n{name}: 데이터 없음 (carrier code: {code})")
            continue
        analyze(df, name)

    print("\n\n분석 완료.")
    print("출처: BTS On-Time Performance (T_ONTIME_MARKETING.csv)")
    print("방법론: TAIL_NUM 기반 항공기 이동 추적 → crew 패턴 역산")
    print("       gap >= 9h → duty 경계, gap >= 24h → pairing 경계 (FAA Part 117 기준)")
