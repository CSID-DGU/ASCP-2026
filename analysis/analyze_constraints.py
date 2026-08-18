"""
항공사별 운항 constraint 추정치 도출.

지원 데이터:
  - BTS On-Time Marketing (delta/alaska/jetblue/southwest)
  - Turkish Airlines .legs 파일 (turkishData/)

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

import glob
import os
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


TURKISH_HOMEBASES = {"HB1", "HB2"}


def analyze_turkish(df):
    """Turkish 전용 분석: homebase 복귀를 pairing 경계로 사용.

    BTS용 analyze()는 gap >= 24h를 pairing 경계로 쓰지만,
    Turkish 항공기는 매일 운항해 24h gap이 없어 pairing 감지가 불가능함.
    대신 HB1/HB2 출발 → HB1/HB2 도착을 1개 pairing으로 정의한다.
    """
    print(f"\n{'='*55}")
    print(f"  TURKISH (homebase-return 기반 pairing 분석)")
    print(f"  총 {len(df):,}편  |  항공기 {df['TAIL_NUM'].nunique():,}대")
    print(f"{'='*55}")

    conn_times   = []
    legs_per_duty = []
    duty_counts  = []
    pairing_days = []

    for tail, grp in df.groupby("TAIL_NUM"):
        grp = grp.sort_values("dep_h").reset_index(drop=True)
        if len(grp) < 2:
            continue

        in_pairing      = False
        pairing_flights = []

        for _, row in grp.iterrows():
            origin = row["ORIGIN"]
            dest   = row["DEST"]

            if not in_pairing:
                if origin in TURKISH_HOMEBASES:
                    in_pairing = True
                    pairing_flights = [row]
            else:
                pairing_flights.append(row)
                if dest in TURKISH_HOMEBASES:
                    # ── pairing 완성 → 분석 ──────────────────────────
                    _analyze_one_pairing(pairing_flights, conn_times,
                                         legs_per_duty, duty_counts, pairing_days)
                    in_pairing = False
                    pairing_flights = []

    # ── 결과 출력 ──────────────────────────────────────────────
    def fmt(arr, name, unit="h"):
        arr = np.array(arr)
        print(f"  {name:22s}  "
              f"p5={np.percentile(arr,5):.2f}{unit}  "
              f"p50={np.percentile(arr,50):.2f}{unit}  "
              f"p95={np.percentile(arr,95):.2f}{unit}  "
              f"mean={arr.mean():.2f}{unit}  "
              f"max={arr.max():.2f}{unit}  "
              f"n={len(arr):,}")

    if conn_times:
        fmt(conn_times,    "connection time",  "h")
    if legs_per_duty:
        fmt(legs_per_duty, "legs / duty",      " legs")
    if duty_counts:
        fmt(duty_counts,   "duties / pairing", " duties")
    if pairing_days:
        fmt(pairing_days,  "pairing days",     " days")

    print(f"\n  [ 추천 설정값 (homebase-return 기반 추정치) ]")
    if conn_times:
        ct = np.array(conn_times)
        print(f"  min_conn         ≈ {np.percentile(ct,  5):.1f}h  (p5)")
        print(f"  max_conn         ≈ {np.percentile(ct, 95):.1f}h  (p95)")
    if legs_per_duty:
        lp = np.array(legs_per_duty)
        print(f"  max_legs         ≈ {int(np.percentile(lp, 95))} legs/duty  (p95)")
    if duty_counts:
        dc = np.array(duty_counts)
        print(f"  max_duty_periods ≈ {int(np.percentile(dc, 95))} duties  (p95)")
    if pairing_days:
        pd_ = np.array(pairing_days)
        print(f"  max_pairing_days ≈ {int(np.percentile(pd_, 95))} days  (p95)")


def _analyze_one_pairing(flights, conn_times, legs_per_duty, duty_counts, pairing_days):
    """완성된 pairing(flight 리스트) 에서 지표를 추출해 리스트에 누적."""
    if len(flights) < 1:
        return

    current_duty_legs = 1
    prev_arr  = flights[0]["arr_h"]
    prev_dest = flights[0]["DEST"]

    duties = 1
    for row in flights[1:]:
        gap = row["dep_h"] - prev_arr
        if gap >= REST_THRESHOLD_H:
            legs_per_duty.append(current_duty_legs)
            duties += 1
            current_duty_legs = 1
        else:
            current_duty_legs += 1
            if row["ORIGIN"] == prev_dest and gap > 0:
                conn_times.append(gap)
        prev_arr  = row["arr_h"]
        prev_dest = row["DEST"]

    legs_per_duty.append(current_duty_legs)
    duty_counts.append(duties)

    span = (flights[-1]["arr_h"] - flights[0]["dep_h"]) / 24.0
    pairing_days.append(max(int(span) + 1, 1))


def load_turkish(data_dir="turkishData", fleet_prefix="3"):
    """Turkish Airlines .legs 파일 파싱 → analyze()와 호환되는 DataFrame 반환.

    포맷 (pipe 구분):
        origin|dest|fleet|aircraft_num|dep_local|arr_local|dep_tz_offset|arr_tz_offset

    fleet_prefix="3": Airbus narrow body (3xxx) 만 포함 (솔루션 생성 대상 기체).
    tz_offset 단위: 분 (양수 = UTC+). UTC = local - tz_offset.
    HB1/HB2: Istanbul homebase (같은 도시, 동일 취급).
    """
    records = []
    for path in sorted(glob.glob(os.path.join(data_dir, "*.legs"))):
        with open(path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("|")
                if len(parts) < 8:
                    continue
                origin, dest, fleet, aircraft = parts[0], parts[1], parts[2], parts[3]
                if not fleet.startswith(fleet_prefix):
                    continue
                try:
                    dep_local = pd.to_datetime(parts[4])
                    arr_local = pd.to_datetime(parts[5])
                    dep_tz    = int(parts[6])   # 분 단위 UTC 오프셋
                    arr_tz    = int(parts[7])
                except (ValueError, IndexError):
                    continue
                records.append({
                    "TAIL_NUM":  f"{fleet}_{aircraft}",  # fleet+번호 복합 키로 고유성 보장
                    "ORIGIN":    origin,
                    "DEST":      dest,
                    "dep_local": dep_local,
                    "arr_local": arr_local,
                    "dep_tz":    dep_tz,
                    "arr_tz":    arr_tz,
                })

    df = pd.DataFrame(records)
    if df.empty:
        return df

    # 로컬 시간 → UTC (local - tz_offset분)
    df["dep_utc"] = df["dep_local"] - pd.to_timedelta(df["dep_tz"], unit="m")
    df["arr_utc"] = df["arr_local"] - pd.to_timedelta(df["arr_tz"], unit="m")

    # UTC 절대 시간 (hours, 데이터 시작일 기준)
    base_utc = df["dep_utc"].min().normalize()
    df["dep_h"] = (df["dep_utc"] - base_utc).dt.total_seconds() / 3600
    df["arr_h"] = (df["arr_utc"] - base_utc).dt.total_seconds() / 3600

    return df.sort_values(["TAIL_NUM", "dep_h"]).reset_index(drop=True)


if __name__ == "__main__":
    # ── BTS 항공사 ────────────────────────────────────────────────────────
    for name, code in AIRLINES.items():
        df = load_airline(code)
        if df.empty:
            print(f"\n{name}: 데이터 없음 (carrier code: {code})")
            continue
        analyze(df, name)

    # ── Turkish Airlines ──────────────────────────────────────────────────
    df_tk = load_turkish()
    if df_tk.empty:
        print("\nturkish: 데이터 없음 (turkishData/*.legs 확인)")
    else:
        analyze_turkish(df_tk)

    print("\n\n분석 완료.")
    print("출처: BTS On-Time Performance (T_ONTIME_MARKETING.csv)")
    print("      Turkish Airlines .legs (turkishData/)")
    print("방법론: TAIL_NUM 기반 항공기 이동 추적 → crew 패턴 역산")
    print("       gap >= 9h → duty 경계, gap >= 24h → pairing 경계 (FAA Part 117 기준)")
