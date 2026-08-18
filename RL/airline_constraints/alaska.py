# Alaska Airlines operating constraint values -- as of 2019 (referencing Alaska 2022 TA)
# base_airport is not included in this file -- train.py injects the actual ID per episode

ALASKA_CONSTRAINTS = {
    "max_duty":        12.5,  # max duty elapsed time (h) -- per Alaska 2022 TA
    "min_conn":        0.65,  # min connection time (h, ~39 min) -- BTS p5 estimate
    "max_conn":         8.8,  # max connection time (h) -- BTS p95 estimate
    "max_legs":           6,  # max flights per duty -- BTS p95 estimate
    "min_rest":        10.0,  # min rest between duties (h) -- FAR 117 SS117.25
    "max_duty_periods":   2,  # max overnight rests per pairing (i.e. number of overnights)
    "max_pairing_days":   5,  # max pairing length (days) -- BTS p95 estimate
    "min_pairing_legs":   3,  # min legs required for END_PAIRING -- blocks 2-leg Nash
    # base_airport excluded -- injected per episode
}
