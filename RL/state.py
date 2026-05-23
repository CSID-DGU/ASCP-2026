def init_state(flights, constraint):
    """에피소드 초기 state 생성

    flights: loader.py가 반환한 flight dict 리스트 (dep_time 기준 오름차순 정렬)
    constraint: 미사용 — 시그니처 통일 목적으로만 받음. 실제 제약 적용은 environment.py에서 수행.
    """
    first = flights[0]

    return {
        # 현재 위치 / 시각
        "current_airport":    first["origin"],       # 현재 승무원 위치 공항 ID
        "current_time":       first["dep_time"],      # 현재 시각 (절대시간, hours)

        # duty 내부 추적
        # duty_time: decoder 입력 feature용 누적 비행 시간 (h)
        # FAA duty window 한도 체크는 duty_start_time 기준으로 get_mask()에서 별도 계산
        "duty_time":          0.0,
        "duty_start_time":    first["dep_time"],      # 현재 duty 시작 시각 (FAA window 기준점)
        "legs":               0,                      # 현재 duty 내 선택된 flight 수

        # 에피소드 전체 추적
        "remaining":          len(flights),           # 미배정 flight 수

        # pairing 상태
        "pairing_start":      True,                   # 새 pairing 시작 직후 — True면 공항/시간 체크 스킵
        "duty_period":        0,                      # 현재 pairing 내 완료된 duty 수
        "pairing_start_time": first["dep_time"],      # pairing 전체 기간 계산 기준점

        # rest 상태
        "is_resting":         False,                  # END_DUTY 후 rest 중 여부
        "rest_end_time":      None,                   # rest 종료 시각 (is_resting=True일 때 유효)
    }
