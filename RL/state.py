def init_state(flights, constraint):
    """Build the initial episode state.

    flights: list of flight dicts returned by loader.py (sorted ascending by dep_time)
    constraint: per-episode constraint dict -- used to initialize base_airport etc.
    """
    base = constraint["base_airport"]
    base_flights = [f for f in flights if f["origin"] == base]
    # Fall back to flights[0] if there is no base-departing leg.
    # load_flights_rolling() includes all legs, so a base-departing leg is
    # not guaranteed; windows with none must be skipped as an episode in train.py.
    first = min(base_flights, key=lambda f: f["dep_time"]) if base_flights else flights[0]

    return {
        # Current position / time
        "current_airport":    base,                   # current crew location airport ID (episode base)
        "current_time":       first["dep_time"],      # current time (absolute, hours)

        # Within-duty tracking
        # duty_time: cumulative flight time (h) used as a decoder input feature
        # FAA duty-window limit checking is done separately in get_mask() using duty_start_time
        "duty_time":          0.0,
        "duty_start_time":    first["dep_time"],      # current duty start time (FAA window reference point)
        "legs":               0,                      # number of legs selected in the current duty
        "total_legs":         0,                      # cumulative legs over the whole pairing (used for the EndPairing bonus)

        # Episode-wide tracking
        "remaining":          len(flights),           # number of unassigned flights

        # Pairing state
        "pairing_start":      True,                   # right after starting a new pairing -- if True, skip airport/time checks
        "duty_period":        0,                      # number of duties completed in the current pairing
        "pairing_start_time": first["dep_time"],      # reference point for pairing-duration calculations

        # Rest state
        "is_resting":         False,                  # whether currently resting after EndDuty
        "rest_end_time":      None,                   # rest end time (valid when is_resting=True)
    }
