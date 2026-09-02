# Delta Air Lines operating constraint values -- as of 2019
# base_airport is not included in this file -- train.py injects the actual ID per episode

DELTA_CONSTRAINTS = {
    "max_duty":        13.0,  # max duty elapsed time (h) -- Delta CBA SS12.D.2 / FAR 117 Table B
    "min_conn":        0.65,  # min connection time (h, ~39 min) -- BTS p5 estimate
    "max_conn":        12.0,  # max connection time (h) -- widened from 9.0 to 12.0 to improve coverage
    "max_legs":           8,  # max flights per duty -- Delta CBA SS12.F (8 landings)
    "min_rest":        10.0,  # min rest between duties (h) -- FAR 117 SS117.25
    "max_duty_periods":   2,  # max overnight rests per pairing (i.e. number of overnights)
    "max_pairing_days":   5,  # max pairing length (days) -- BTS p95 estimate
}
