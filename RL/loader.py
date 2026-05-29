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


def load_flights_multiday(path, limit=200, n_days=4, seed=42):
    """같은 flight set을 n_days일치로 복제하여 multi-day 데이터 생성.

    동일한 flights를 매일 반복 → overnight connection이 자연 생성됨.
    결과: limit × n_days 개의 flight (ID = day * limit + original_id)

    NOTE: 다양성이 필요한 학습에는 load_flights_rolling() 사용 권장.
    """
    base = load_flights(path, limit=limit, seed=seed)

    all_flights = []
    for day in range(n_days):
        for f in base:
            all_flights.append({
                "id":       day * limit + f["id"],
                "origin":   f["origin"],
                "dest":     f["dest"],
                "dep_time": f["dep_time"] + day * 24.0,
                "arr_time": f["arr_time"] + day * 24.0,
            })

    return all_flights


def load_flights_rolling(
    path,
    window_days=5,
    offset_days=0,
    airport_map=None,
):
    """슬라이딩 윈도우 방식으로 실제 날짜 데이터 로드.

    에피소드마다 offset_days를 달리하면 서로 다른 flight 구성으로 훈련 가능.
    hub_only 없이 모든 노선(spoke-spoke 포함)을 그대로 사용한다.
    flight 수는 window_days로만 제어한다 (개수 제한 없음).

    Args:
        path:         BTS CSV 경로
        window_days:  윈도우 크기 (일), 기본 5
        offset_days:  전체 날짜 목록 기준 시작 인덱스 (에피소드마다 랜덤 지정)
        airport_map:  전체-데이터 기준 공항 ID 맵; None이면 전체 CSV에서 재계산(느림).
                      train 루프에서는 build_airport_map()으로 사전 계산 후 전달 권장.

    Returns:
        flight dict 리스트 (dep_time 오름차순 정렬)
    """
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

    return flights
