# Turkish Airlines (THY) operating constraint values -- based on 2014 timetable data
# Estimated values based on EASA FTL Subpart Q / SHGM regulations
# Airbus narrow body fleet only (fleet 3xxx)
# base_airport is not included -- train.py injects the actual ID per episode

TURKISH_CONSTRAINTS = {
    "max_duty":        13.0,  # max FDP per EASA FTL SS ORO.FTL.205 (h)
    "min_conn":         0.8, # min connection time (h, ~45 min) -- Istanbul hub estimate
    "max_conn":         4.0,  # max connection time (h)
    "max_legs":           4,  # max sectors per duty per EASA FTL
    "min_rest":        10.0,  # min rest per EASA FTL SS ORO.FTL.235 (h)
    "max_duty_periods":   2,  # max overnight rests per pairing
    "max_pairing_days":   3,  # max pairing length (days)
}
