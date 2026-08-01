# JetBlue Airways operating constraint values -- as of 2019 (JetBlue 2018 CBA / FAR 117 applied)
# base_airport is not included in this file -- train.py injects the actual ID per episode

JETBLUE_CONSTRAINTS = {
    "max_duty":        13.0,  # max duty elapsed time (h) -- per FAR 117 Table B
    "min_conn":        0.58,  # min connection time (h, ~35 min) -- BTS p5 estimate
    "max_conn":         8.0,  # max connection time (h) -- BTS p95 estimate
    "max_legs":           7,  # max flights per duty -- BTS p95 estimate
    "min_rest":        10.0,  # min rest between duties (h) -- FAR 117 SS117.25
    "max_duty_periods":   3,  # max overnight rests per pairing (i.e. number of overnights, upper bound)
    "max_pairing_days":   7,  # max pairing length (days) -- BTS p95 estimate (upper bound)
    "min_pairing_legs":   3,  # min legs required for END_PAIRING -- blocks 2-leg Nash
    # base_airport excluded -- injected per episode
}
