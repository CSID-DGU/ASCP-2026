import random
import pandas as pd
from collections import Counter


def convert_time(hhmm):
    hhmm = int(hhmm)
    h = hhmm // 100
    m = hhmm % 100
    return h + m / 60


# 공항별 UTC 오프셋(분, 표준시 기준). BTS CRS_DEP_TIME/CRS_ARR_TIME은 각 공항의 현지
# 로컬시각이라, 서로 다른 타임존의 공항을 잇는 연결편은 dep_time을 그대로 빼면 gap이
# 음수/이상값으로 나와 실제로는 연결 가능한 편이 마스크에서 차단된다 (예: ATL(ET)→LAX(PT)
# 착륙 후 다음 편 dep_time이 PT 기준이라 ET 기준 current_time과 clock이 안 맞음).
# dep_time을 UTC 절대시간으로 앵커링하면 arr_time = dep_time + elapsed(block time,
# 타임존 무관)만으로 도착 시각도 자동으로 올바른 UTC가 된다.
# (analysis/flight_time_distribution.py의 _UTC 테이블과 동일 출처)
_UTC_OFFSET_MIN = {
    **{ap: -600 for ap in ['HNL', 'KOA', 'LIH', 'OGG']},
    **{ap: -540 for ap in ['ANC', 'FAI']},
    **{ap: -480 for ap in ['GEG', 'LAS', 'LAX', 'OAK', 'ONT', 'PDX', 'PSP', 'RNO', 'SAN', 'SEA',
                            'SFO', 'SJC', 'SMF', 'SNA']},
    **{ap: -420 for ap in ['ABQ', 'BIL', 'BOI', 'BZN', 'COS', 'DEN', 'EGE', 'ELP', 'FCA', 'HDN',
                            'JAC', 'MSO', 'MTJ', 'PHX', 'SLC', 'TUS']},
    **{ap: -360 for ap in ['ATW', 'AUS', 'BHM', 'BIS', 'BNA', 'BTR', 'CID', 'DAL', 'DFW', 'DSM',
                            'ECP', 'FAR', 'FSD', 'GPT', 'GRB', 'HOU', 'HSV', 'IAH', 'ICT', 'JAN',
                            'LFT', 'LIT', 'MCI', 'MDW', 'MEM', 'MKE', 'MOB', 'MSN', 'MSP', 'MSY',
                            'OKC', 'OMA', 'ORD', 'PNS', 'SAT', 'STL', 'TUL', 'VPS', 'XNA']},
    **{ap: -300 for ap in ['ABE', 'AGS', 'ALB', 'ATL', 'AVL', 'AVP', 'BDL', 'BOS', 'BUF', 'BWI',
                            'CAE', 'CAK', 'CHA', 'CHO', 'CHS', 'CLE', 'CLT', 'CMH', 'CRW', 'CVG',
                            'DAB', 'DAY', 'DCA', 'DTW', 'EWR', 'EYW', 'FAY', 'FLL', 'FNT', 'GNV',
                            'GRR', 'GSO', 'GSP', 'HPN', 'IAD', 'ILM', 'IND', 'JAX', 'JFK', 'LEX',
                            'LGA', 'MCO', 'MDT', 'MHT', 'MIA', 'MLB', 'MYR', 'ORF', 'PBI', 'PHF',
                            'PHL', 'PIT', 'PVD', 'PWM', 'RDU', 'RIC', 'ROA', 'ROC', 'RSW', 'SAV',
                            'SDF', 'SRQ', 'SYR', 'TLH', 'TPA', 'TRI', 'TYS']},
    **{ap: -240 for ap in ['SJU', 'STT', 'STX']},
}


def utc_offset_hours(airport_code):
    """공항의 UTC 오프셋(시간). 목록에 없으면 Eastern(-5h)을 기본값으로 사용."""
    return _UTC_OFFSET_MIN.get(airport_code, -300) / 60.0


def build_airport_map(path):
    """전체 BTS CSV 기준으로 공항→int 맵을 생성한다.

    path: str 또는 str 리스트. 여러 항공사 CSV를 합쳐 통합 공항 ID 공간 구성 가능.
    빈도 내림차순 정렬: index 0 = 가장 빈도 높은 공항(허브/base 후보).
    에피소드마다 다른 rolling window를 써도 ID가 일관된다.
    """
    paths = [path] if isinstance(path, str) else path
    counts = Counter()
    for p in paths:
        df = pd.read_csv(p, usecols=["ORIGIN", "DEST"]).dropna()
        counts.update(list(df["ORIGIN"]) + list(df["DEST"]))
    airports_sorted = sorted(counts.keys(), key=lambda a: -counts[a])
    return {a: i for i, a in enumerate(airports_sorted)}


def bases_to_ids(bases, airport_map):
    """문자열 base 리스트를 정수 ID로 변환한다.

    Args:
        bases:       공항 코드 문자열 리스트 (예: ["ATL", "DTW", "MSP"])
        airport_map: build_airport_map()으로 생성한 공항→int 맵

    Returns:
        정수 ID 리스트. airport_map에 없는 코드는 경고 후 무시한다.
    """
    ids = [airport_map[b] for b in bases if b in airport_map]
    missing = [b for b in bases if b not in airport_map]
    if missing:
        print(f"[bases_to_ids] 경고: airport_map에 없는 base 제외됨: {missing}")
    if not ids:
        raise ValueError(f"유효한 base가 없음. bases={bases}")
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
        "ORIGIN", "DEST", "CRS_DEP_TIME", "CRS_ARR_TIME", "CRS_ELAPSED_TIME", "FL_DATE"
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

    base_date = df["FL_DATE"].min()
    df["day_offset"] = (df["FL_DATE"] - base_date).dt.days
    # dep_time을 UTC 절대시간으로 앵커링 (출발 공항 로컬시각 - UTC 오프셋)
    # → 서로 다른 타임존 공항 간 연결편 gap 계산이 정확해짐 (log/0703/공유받은문제.md 참고)
    df["dep_time"] = (
        df["CRS_DEP_TIME"].apply(convert_time)
        + df["day_offset"] * 24
        - df["ORIGIN"].map(utc_offset_hours)
    )
    # CRS_ELAPSED_TIME(분, block time)은 타임존 무관 → dep_time(UTC)에 더하면 arr_time도 UTC
    df["arr_time"] = df["dep_time"] + df["CRS_ELAPSED_TIME"] / 60.0

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



def sample_connected_subnet(flights_window, base_id, n_max):
    """공항(스테이션) 집합 기반 서브넷 샘플링 — star graph 문제 해결
      1. base를 반드시 포함
      2. 교통량 상위 spoke를 하나씩 추가
      3. chosen 공항 집합 '내부' 간선(origin∈chosen AND dest∈chosen)만 포함
         → spoke-spoke 간선이 살아남아 ATL→MSP→SLC→ATL 같은 긴 chain 가능
      4. n_max 초과 시 시간순으로 잘라냄
    """
    from collections import Counter
    deg = Counter()
    for f in flights_window:
        deg[f["origin"]] += 1
        deg[f["dest"]] += 1
    spokes = [a for a, _ in deg.most_common() if a != base_id]
    chosen = {base_id}
    out = []
    for s in spokes:
        chosen.add(s)
        candidate = [f for f in flights_window
                     if f["origin"] in chosen and f["dest"] in chosen]
        if len(candidate) >= n_max:
            out = candidate
            break
        out = candidate
    out = sorted(out, key=lambda f: f["dep_time"])[:n_max]
    for i, f in enumerate(out):
        f["id"] = i
    return out


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
            "ORIGIN", "DEST", "CRS_DEP_TIME", "CRS_ARR_TIME", "CRS_ELAPSED_TIME", "FL_DATE"
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
    base_date = min(window_dates)
    df["day_offset"] = (df["FL_DATE"] - base_date).dt.days
    # dep_time을 UTC 절대시간으로 앵커링 — load_flights()와 동일 이유 (log/0703/공유받은문제.md)
    df["dep_time"] = (
        df["CRS_DEP_TIME"].apply(convert_time)
        + df["day_offset"] * 24
        - df["ORIGIN"].map(utc_offset_hours)
    )
    df["arr_time"] = df["dep_time"] + df["CRS_ELAPSED_TIME"] / 60.0

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

    # connected subnet sampling: star graph 문제 해결 (base-first 랜덤 샘플링 → 공항 집합 기반)
    if n_max is not None and len(flights) > n_max and base_airport is not None:
        flights = sample_connected_subnet(flights, base_airport, n_max)

    return flights
