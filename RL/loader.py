import pandas as pd


def convert_time(hhmm):
    hhmm = int(hhmm)
    h = hhmm // 100
    m = hhmm % 100
    return h + m / 60


def load_flights(path, limit=50):
    df = pd.read_csv(path)

    df = df[[
        "ORIGIN",
        "DEST",
        "CRS_DEP_TIME",
        "CRS_ARR_TIME",
        "FL_DATE"
    ]].dropna()

    # ✔ 먼저 limit 적용 (핵심)
    df = df.head(limit)

    df["dep_time"] = df["CRS_DEP_TIME"].apply(convert_time)
    df["arr_time"] = df["CRS_ARR_TIME"].apply(convert_time)

    df["FL_DATE"] = pd.to_datetime(df["FL_DATE"], format="mixed")

    base_date = df["FL_DATE"].min()
    df["day_offset"] = (df["FL_DATE"] - base_date).dt.days

    df["dep_time"] += df["day_offset"] * 24
    df["arr_time"] += df["day_offset"] * 24

    df = df.sort_values("dep_time")

    airports = pd.concat([df["ORIGIN"], df["DEST"]]).unique()
    airport_map = {a: i for i, a in enumerate(airports)}

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