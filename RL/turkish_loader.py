"""
Turkish Airlines .legs 파일 로더.

파일 형식: ORIGIN|DEST|FLEET|AC_NUM|DEP_DT_LOCAL|ARR_DT_LOCAL|DEP_TZ_MIN|ARR_TZ_MIN
UTC 변환: dep_utc = dep_local - DEP_TZ_MIN분

load_flights_rolling()과 동일한 flight dict 형식을 반환하므로
train.py / evaluation/evaluate_ip.py 등 기존 코드를 그대로 활용 가능.
"""
import os
import glob
import random
import pandas as pd
from collections import Counter


def _parse_legs_file(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) != 8:
                continue
            try:
                rows.append({
                    "ORIGIN": parts[0],
                    "DEST":   parts[1],
                    "FLEET":  parts[2],
                    "DEP_DT": parts[4],
                    "ARR_DT": parts[5],
                    "DEP_TZ": int(parts[6]),
                    "ARR_TZ": int(parts[7]),
                })
            except (ValueError, IndexError):
                continue
    return pd.DataFrame(rows)


def parse_legs_dir(dir_path, fleet_prefix="3", files=None):
    """
    디렉토리 내 .legs 파일을 파싱 → UTC 기준 시간 포함 DataFrame 반환.

    반환 컬럼:
        ORIGIN, DEST          공항 코드
        dep_utc, arr_utc      UTC datetime
        dep_date_utc          dep_utc의 날짜 (rolling window 기준)

    Args:
        dir_path:     .legs 파일이 있는 디렉토리
        fleet_prefix: 포함할 fleet 코드 prefix. None이면 전체. 기본 "3" = Airbus narrow body.
        files:        사용할 파일 이름 목록 (예: ["tt201401.legs"]). None이면 디렉토리 내 전체.
    """
    if files is not None:
        legs_files = [os.path.join(dir_path, f) for f in files]
        missing = [p for p in legs_files if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(f".legs 파일 없음: {missing}")
    else:
        legs_files = sorted(glob.glob(os.path.join(dir_path, "*.legs")))
    if not legs_files:
        raise FileNotFoundError(f".legs 파일 없음: {dir_path}")

    dfs = [_parse_legs_file(p) for p in legs_files]
    df  = pd.concat(dfs, ignore_index=True)

    if fleet_prefix is not None:
        df = df[df["FLEET"].str.startswith(fleet_prefix)].copy()

    # 로컬 시간 → UTC: dep_utc = dep_local - DEP_TZ분
    df["dep_local"] = pd.to_datetime(df["DEP_DT"])
    df["arr_local"] = pd.to_datetime(df["ARR_DT"])
    df["dep_utc"]   = df["dep_local"] - pd.to_timedelta(df["DEP_TZ"], unit="min")
    df["arr_utc"]   = df["arr_local"] - pd.to_timedelta(df["ARR_TZ"], unit="min")
    df["dep_date_utc"] = df["dep_utc"].dt.normalize()

    # 같은 origin-dest 및 arr <= dep (동향 단거리 시간대 역전) 필터
    df = df[(df["ORIGIN"] != df["DEST"]) & (df["arr_utc"] > df["dep_utc"])].copy()

    return df[["ORIGIN", "DEST", "dep_utc", "arr_utc", "dep_date_utc"]].reset_index(drop=True)


def build_airport_map_turkish(dir_path=None, fleet_prefix="3", df=None):
    """
    공항→int ID 맵 생성.

    빈도 내림차순 정렬 (index 0 = 가장 빈도 높은 공항 = HB1/HB2 homebase).

    Args:
        dir_path:     parse_legs_dir에 넘길 디렉토리 경로. df가 있으면 생략 가능.
        fleet_prefix: parse_legs_dir에 넘길 fleet 필터. df가 있으면 무시.
        df:           이미 parse_legs_dir()로 로드한 DataFrame. 제공 시 재파싱 생략.
    """
    if df is None:
        if dir_path is None:
            raise ValueError("dir_path 또는 df 중 하나를 제공해야 합니다.")
        df = parse_legs_dir(dir_path, fleet_prefix=fleet_prefix)
    counts = Counter(list(df["ORIGIN"]) + list(df["DEST"]))
    airports_sorted = sorted(counts.keys(), key=lambda a: -counts[a])
    return {a: i for i, a in enumerate(airports_sorted)}


def load_flights_rolling_turkish(
    window_days=5,
    offset_days=0,
    airport_map=None,
    base_airport=None,
    df=None,
    n_max=None,
):
    """
    슬라이딩 윈도우 방식으로 Turkish .legs 데이터에서 flight 로드.

    load_flights_rolling()과 동일한 flight dict 형식 반환.
    dep_time / arr_time: UTC 기준 윈도우 시작 시점으로부터의 상대 시간 (hours).

    Args:
        window_days:  윈도우 크기 (일), 기본 5
        offset_days:  전체 날짜 목록 기준 시작 인덱스 (에피소드마다 랜덤 지정)
        airport_map:  build_airport_map_turkish()로 생성한 공항→int 맵
        base_airport: 에피소드 base 공항 ID; base-first sampling 기준
        df:           parse_legs_dir()로 사전 로드된 DataFrame (필수)
        n_max:        에피소드 최대 flight 수; 초과 시 base-first sampling 적용

    Returns:
        flight dict 리스트 (dep_time 오름차순)
    """
    if df is None:
        raise ValueError("df는 parse_legs_dir()로 사전 로드해 전달해야 합니다.")
    if airport_map is None:
        raise ValueError("airport_map은 build_airport_map_turkish()로 생성해 전달해야 합니다.")

    dates = sorted(df["dep_date_utc"].unique())
    window_dates = dates[offset_days: offset_days + window_days]
    if not window_dates:
        return []

    df_win = df[df["dep_date_utc"].isin(window_dates)].copy()

    # 윈도우 시작 UTC 기준 상대 시간 (hours)
    base_dt = pd.Timestamp(min(window_dates))
    df_win["dep_time"] = (df_win["dep_utc"] - base_dt).dt.total_seconds() / 3600.0
    df_win["arr_time"] = (df_win["arr_utc"] - base_dt).dt.total_seconds() / 3600.0

    df_win = df_win.sort_values("dep_time").reset_index(drop=True)

    df_win["origin"] = df_win["ORIGIN"].map(airport_map)
    df_win["dest"]   = df_win["DEST"].map(airport_map)
    df_win = df_win.dropna(subset=["origin", "dest"])

    flights = []
    for _, row in df_win.iterrows():
        flights.append({
            "id":       len(flights),
            "origin":   int(row["origin"]),
            "dest":     int(row["dest"]),
            "dep_time": float(row["dep_time"]),
            "arr_time": float(row["arr_time"]),
        })

    # base-first sampling: base 관련 편 우선, 나머지 랜덤 샘플링
    if n_max is not None and len(flights) > n_max and base_airport is not None:
        base_fs = [f for f in flights if f["origin"] == base_airport or f["dest"] == base_airport]
        mid_fs  = [f for f in flights if f["origin"] != base_airport and f["dest"] != base_airport]
        if len(base_fs) >= n_max:
            random.shuffle(base_fs)
            flights = sorted(base_fs[:n_max], key=lambda f: f["dep_time"])
        else:
            remaining = n_max - len(base_fs)
            random.shuffle(mid_fs)
            flights = sorted(base_fs + mid_fs[:remaining], key=lambda f: f["dep_time"])
        for i, f in enumerate(flights):
            f["id"] = i

    return flights
