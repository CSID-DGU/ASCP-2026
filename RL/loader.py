import pandas as pd
from collections import Counter


def convert_time(hhmm):
    hhmm = int(hhmm)
    h = hhmm // 100
    m = hhmm % 100
    return h + m / 60


def load_flights(path, limit=50, seed=42, n_days_max=None, hub_only=False):
    """
    BTS 데이터에서 flight 로드.

    hub_only=True: 가장 빈도 높은 공항(허브)에 출발 또는 도착하는 flight만 포함.
        → 모든 flight이 허브를 경유하므로 base-to-base pairing 항상 가능 → dummy 0개.
        → 모든 flight이 허브를 경유하므로 base-to-base pairing 항상 가능.

    공항 인덱스: 빈도 내림차순 정렬 → index 0 = 허브 = base_airport.
    """
    df = pd.read_csv(path)

    df = df[[
        "ORIGIN",
        "DEST",
        "CRS_DEP_TIME",
        "CRS_ARR_TIME",
        "FL_DATE"
    ]].dropna()

    df["FL_DATE"] = pd.to_datetime(df["FL_DATE"], format="mixed")

    # 허브 결정: 전체 데이터에서 가장 빈도 높은 공항 (필터링 전 기준)
    all_airports = list(df["ORIGIN"]) + list(df["DEST"])
    hub = Counter(all_airports).most_common(1)[0][0]

    # hub_only: round-trip 가능한 스포크 도시만 포함
    #   전체 데이터 기준 ATL→X AND X→ATL 둘 다 존재하는 도시만 필터
    #   샘플링 후 단방향만 들어간 도시 반복 제거 → 양방향 보장
    if hub_only:
        hub_to_spoke = set(df[df["ORIGIN"] == hub]["DEST"].unique()) - {hub}
        spoke_to_hub = set(df[df["DEST"] == hub]["ORIGIN"].unique()) - {hub}
        round_trip_cities = hub_to_spoke & spoke_to_hub
        df = df[
            ((df["ORIGIN"] == hub) & (df["DEST"].isin(round_trip_cities))) |
            ((df["DEST"] == hub) & (df["ORIGIN"].isin(round_trip_cities)))
        ].copy()

    # 기본: head(limit)으로 앞에서부터 자름 (같은 날 데이터)
    # n_days_max 지정 시: 날짜별 골고루 샘플링
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

    # hub_only 후처리:
    # 1) 단방향 spoke city 제거 (샘플에서 한쪽 방향이 빠진 경우)
    # 2) overnight timing 호환성 체크: 선행 duty가 overnight window 내 도착 불가한 flight 제거
    #    (X→ATL flight인데 ATL→X 도착이 너무 늦어 min_rest 부족한 경우)
    if hub_only:
        df_pool = df.copy()  # 현재 head(limit) 결과
        df_full_hub = df  # 이미 round_trip filter 적용된 전체

        # 전체 round-trip pool (limit보다 훨씬 많음)
        import pandas as _pd
        df_all = _pd.read_csv(path)
        df_all = df_all[["ORIGIN", "DEST", "CRS_DEP_TIME", "CRS_ARR_TIME", "FL_DATE"]].dropna()
        df_all["FL_DATE"] = _pd.to_datetime(df_all["FL_DATE"], format="mixed")
        hub_to_spoke_full = set(df_all[df_all["ORIGIN"] == hub]["DEST"].unique()) - {hub}
        spoke_to_hub_full = set(df_all[df_all["DEST"] == hub]["ORIGIN"].unique()) - {hub}
        rt_full = hub_to_spoke_full & spoke_to_hub_full
        df_pool_full = df_all[
            ((df_all["ORIGIN"] == hub) & (df_all["DEST"].isin(rt_full))) |
            ((df_all["DEST"] == hub) & (df_all["ORIGIN"].isin(rt_full)))
        ].copy()

        # 현재 샘플에서 단방향 city 제거 → 부족분 보충 반복
        for _ in range(10):
            outbound = set(df[df["ORIGIN"] == hub]["DEST"]) - {hub}
            inbound  = set(df[df["DEST"] == hub]["ORIGIN"]) - {hub}
            connected = outbound & inbound
            df = df[
                ((df["ORIGIN"] == hub) & (df["DEST"].isin(connected))) |
                ((df["DEST"] == hub) & (df["ORIGIN"].isin(connected)))
            ]
            if len(df) >= limit:
                df = df.head(limit)
                break
            # 부족하면 pool에서 추가 rows 보충
            used_idx = set(df.index)
            extra = df_pool_full[~df_pool_full.index.isin(used_idx)].head(limit - len(df) + 20)
            df = pd.concat([df, extra]).drop_duplicates()

    df["dep_time"] = df["CRS_DEP_TIME"].apply(convert_time)
    df["arr_time"] = df["CRS_ARR_TIME"].apply(convert_time)

    base_date = df["FL_DATE"].min()
    df["day_offset"] = (df["FL_DATE"] - base_date).dt.days

    df["dep_time"] += df["day_offset"] * 24
    df["arr_time"] += df["day_offset"] * 24

    df = df.sort_values("dep_time").reset_index(drop=True)

    # hub_only: overnight timing 호환성 체크 (dep_time 변환 후)
    # X→ATL flight: ATL→X flights이 24h expansion 후 올바른 overnight window로 도착 가능해야 함
    # 4-day expansion 기준: day k의 X→ATL(dep=T)에 대해
    #   preceding ATL→X arr_time이 [T-24-max_rest, T-24-min_rest] ∪ [T-max_rest, T-min_rest] 내에 있어야 함
    # 간단 체크: ATL→X 최소 arrival(= min arr_time in df for DEST==X) + min_rest < X→ATL dep + 24
    if hub_only:
        MIN_REST = 8.0
        MAX_REST = 24.0
        # ATL→X: 각 spoke 공항 X에 대한 최소 도착 시간 (당일 가장 이른 ATL 출발 편)
        outbound_min_arr = df[df["ORIGIN"] == hub].groupby("DEST")["arr_time"].min()
        # X→ATL: 각 spoke 공항 X에 대한 최소 출발 시간
        inbound_min_dep = df[df["DEST"] == hub].groupby("ORIGIN")["dep_time"].min()

        def is_pairable(row):
            """overnight [min_rest, max_rest] 내 연결 가능한 flight인지 체크"""
            if row["ORIGIN"] == hub:
                # ATL→X: day0 arr + min_rest < day1 X→ATL earliest dep (= dep+24)
                x = row["DEST"]
                if x not in inbound_min_dep:
                    return False
                # day0 ATL→X arr + min_rest ≤ day1 X→ATL dep (= inbound_dep + 24)
                return row["arr_time"] + MIN_REST <= inbound_min_dep[x] + 24
            else:
                # X→ATL: day0 ATL→X earliest arr + min_rest ≤ day1 this flight dep
                x = row["ORIGIN"]
                if x not in outbound_min_arr:
                    return False
                # day1 dep = row["dep_time"] + 24
                return outbound_min_arr[x] + MIN_REST <= row["dep_time"] + 24

        mask = df.apply(is_pairable, axis=1)
        df = df[mask].reset_index(drop=True)

    # 공항 인덱스: 빈도 내림차순 → index 0 = 허브 = base_airport
    airport_counts = Counter(list(df["ORIGIN"]) + list(df["DEST"]))
    airports_sorted = sorted(airport_counts.keys(), key=lambda a: -airport_counts[a])
    airport_map = {a: i for i, a in enumerate(airports_sorted)}

    df["origin"] = df["ORIGIN"].map(airport_map)
    df["dest"] = df["DEST"].map(airport_map)

    flights = []
    for _, row in df.iterrows():
        flights.append({
            "id": len(flights),
            "origin": int(row["origin"]),
            "dest": int(row["dest"]),
            "dep_time": float(row["dep_time"]),
            "arr_time": float(row["arr_time"])
        })

    return flights


def load_flights_multiday(path, limit=200, n_days=4, seed=42, hub_only=False):
    """같은 flight set을 n_days일치로 복제하여 multi-day 데이터 생성.

    동일한 flights를 매일 반복 → overnight connection이 자연 생성됨.
    결과: limit × n_days 개의 flight (ID는 day * limit + original_id)
    """
    base = load_flights(path, limit=limit, seed=seed, hub_only=hub_only)

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
