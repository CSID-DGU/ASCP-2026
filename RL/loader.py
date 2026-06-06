import random
import pandas as pd
from collections import Counter


def convert_time(hhmm):
    hhmm = int(hhmm)
    h = hhmm // 100
    m = hhmm % 100
    return h + m / 60


def build_airport_map(path):
    """전체 BTS CSV 기준으로 공항→int 맵을 생성한다.

    빈도 내림차순 정렬: index 0 = 가장 빈도 높은 공항(허브/base 후보).
    에피소드마다 다른 rolling window를 써도 ID가 일관된다.
    """
    df = pd.read_csv(path, usecols=["ORIGIN", "DEST"]).dropna()
    counts = Counter(list(df["ORIGIN"]) + list(df["DEST"]))
    airports_sorted = sorted(counts.keys(), key=lambda a: -counts[a])
    return {a: i for i, a in enumerate(airports_sorted)}


def bases_to_ids(bases, airport_map):
    """문자열 base 리스트를 정수 ID로 변환한다.

    Args:
        bases:       공항 코드 문자열 리스트 (예: ["ATL", "DTW", "MSP"])
        airport_map: build_airport_map()으로 생성한 공항→int 맵

    Returns:
        정수 ID 리스트. airport_map에 없는 코드는 무시한다.
    """
    ids = [airport_map[b] for b in bases if b in airport_map]
    if len(ids) < len(bases):
        missing = [b for b in bases if b not in airport_map]
        raise ValueError(f"airport_map에 없는 base: {missing}")
    return ids


def get_bases(flights, n_bases=3):
    """flight dict 리스트에서 빈도 상위 n_bases개 공항 ID를 반환한다.

    airport_map이 빈도 내림차순이므로 결과는 통상 [0, 1, ..., n_bases-1].
    """
    counts = Counter()
    for f in flights:
        counts[f["origin"]] += 1
        counts[f["dest"]] += 1
    return [a for a, _ in counts.most_common(n_bases)]


def load_flights(path, limit=50, seed=42, n_days_max=None):
    """BTS 데이터에서 flight 로드.

    공항 인덱스: 전체 CSV 기준 빈도 내림차순 → index 0 = 허브.
    limit과 무관하게 항상 동일한 airport_map이 사용된다.
    """
    df = pd.read_csv(path)
    df = df[[
        "ORIGIN", "DEST", "CRS_DEP_TIME", "CRS_ARR_TIME", "FL_DATE"
    ]].dropna()
    df["FL_DATE"] = pd.to_datetime(df["FL_DATE"], format="mixed")

    # 전체 데이터 기준으로 airport_map 구축 (subset 잘라내기 전에 계산)
    airport_counts = Counter(list(df["ORIGIN"]) + list(df["DEST"]))
    airports_sorted = sorted(airport_counts.keys(), key=lambda a: -airport_counts[a])
    airport_map = {a: i for i, a in enumerate(airports_sorted)}

    if n_days_max is not None:
        dates = sorted(df["FL_DATE"].unique())[:n_days_max]
        n_per_day = max(1, limit // len(dates))
        pieces = [
            day_df.sample(min(n_per_day, len(day_df)), random_state=seed)
            for date in dates
            for day_df in [df[df["FL_DATE"] == date]]
        ]
        df = pd.concat(pieces).reset_index(drop=True).head(limit)
    else:
        df = df.head(limit)

    df["dep_time"] = df["CRS_DEP_TIME"].apply(convert_time)
    df["arr_time"] = df["CRS_ARR_TIME"].apply(convert_time)
    # 자정을 넘는 항공편: FL_DATE는 출발일 기준 → arr_time이 dep_time보다 작으면 +24
    df.loc[df["arr_time"] < df["dep_time"], "arr_time"] += 24

    base_date = df["FL_DATE"].min()
    df["day_offset"] = (df["FL_DATE"] - base_date).dt.days
    df["dep_time"] += df["day_offset"] * 24
    df["arr_time"] += df["day_offset"] * 24

    df = df.sort_values("dep_time").reset_index(drop=True)

    df["origin"] = df["ORIGIN"].map(airport_map)
    df["dest"]   = df["DEST"].map(airport_map)

    flights = []
    for _, row in df.iterrows():
        flights.append({
            "id":       len(flights),
            "origin":   int(row["origin"]),
            "dest":     int(row["dest"]),
            "dep_time": float(row["dep_time"]),
            "arr_time": float(row["arr_time"]),
        })

    return flights



def load_flights_rolling(
    path,
    window_days=5,
    offset_days=0,
    airport_map=None,
    base_airport=None,
    n_max=None,
    df=None,
):
    """슬라이딩 윈도우 방식으로 실제 날짜 데이터 로드.

    에피소드마다 offset_days를 달리하면 서로 다른 flight 구성으로 훈련 가능.
    hub_only 없이 모든 노선(spoke-spoke 포함)을 그대로 사용한다.

    Base-first sampling (base_airport + n_max 지정 시):
        base_flights = origin=base 또는 dest=base → 전부 포함
        mid_flights  = 나머지 spoke-spoke         → 남은 슬롯만큼 랜덤 샘플링
    n_max 미지정 시 전체 flight 반환.

    Args:
        path:          BTS CSV 경로
        window_days:   윈도우 크기 (일), 기본 5
        offset_days:   전체 날짜 목록 기준 시작 인덱스 (에피소드마다 랜덤 지정)
        airport_map:   전체-데이터 기준 공항 ID 맵; None이면 전체 CSV에서 재계산(느림).
        base_airport:  에피소드 base 공항 ID; base-first sampling 기준
        n_max:         에피소드 최대 flight 수; 초과 시 base-first sampling 적용
        df:            사전 로드된 DataFrame; 제공 시 CSV 재로딩 생략 (에피소드 반복 호출 최적화)

    Returns:
        flight dict 리스트 (dep_time 오름차순 정렬)
    """
    if df is None:
        df = pd.read_csv(path)
        df = df[[
            "ORIGIN", "DEST", "CRS_DEP_TIME", "CRS_ARR_TIME", "FL_DATE"
        ]].dropna()
        df["FL_DATE"] = pd.to_datetime(df["FL_DATE"], format="mixed")

    # airport_map이 없으면 전체 데이터 기준으로 구축 (에피소드 간 ID 일관성 보장)
    if airport_map is None:
        counts = Counter(list(df["ORIGIN"]) + list(df["DEST"]))
        airports_sorted = sorted(counts.keys(), key=lambda a: -counts[a])
        airport_map = {a: i for i, a in enumerate(airports_sorted)}

    # 윈도우 날짜 추출
    dates = sorted(df["FL_DATE"].unique())
    window_dates = dates[offset_days: offset_days + window_days]
    if not window_dates:
        return []

    df = df[df["FL_DATE"].isin(window_dates)].copy()

    # 시간 변환 + 윈도우 시작일 기준 day offset
    df["dep_time"] = df["CRS_DEP_TIME"].apply(convert_time)
    df["arr_time"] = df["CRS_ARR_TIME"].apply(convert_time)
    df.loc[df["arr_time"] < df["dep_time"], "arr_time"] += 24
    base_date = min(window_dates)
    df["day_offset"] = (df["FL_DATE"] - base_date).dt.days
    df["dep_time"] += df["day_offset"] * 24
    df["arr_time"] += df["day_offset"] * 24

    df = df.sort_values("dep_time").reset_index(drop=True)

    df["origin"] = df["ORIGIN"].map(airport_map)
    df["dest"]   = df["DEST"].map(airport_map)
    df = df.dropna(subset=["origin", "dest"])

    flights = []
    for _, row in df.iterrows():
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
            # base 관련 편만으로도 n_max 초과 → base 편에서 랜덤 샘플링
            random.shuffle(base_fs)
            flights = sorted(base_fs[:n_max], key=lambda f: f["dep_time"])
        else:
            remaining = n_max - len(base_fs)
            random.shuffle(mid_fs)
            flights = sorted(base_fs + mid_fs[:remaining], key=lambda f: f["dep_time"])
        for i, f in enumerate(flights):
            f["id"] = i

    return flights
