"""CPPSC 공항 identity를 ASCP checkpoint embedding에 맞추는 함수."""


def remap_cppsc_airports(flights, cppsc_airport_map, base_ids, checkpoint_airport_map):
    """CPPSC의 자체 숫자 ID를 checkpoint가 학습한 동일 공항 코드 ID로 변환함."""
    id_to_code = {idx: code for code, idx in cppsc_airport_map.items()}
    used_codes = {
        id_to_code[airport_id]
        for flight in flights
        for airport_id in (flight["origin"], flight["dest"])
    }
    unknown = sorted(used_codes - set(checkpoint_airport_map))
    if unknown:
        raise ValueError(
            "CPPSC 공항을 checkpoint embedding 의미에 맞게 변환할 수 없음: "
            f"학습 당시 없던 공항 {unknown}"
        )
    remapped = [
        {
            **flight,
            "origin": checkpoint_airport_map[id_to_code[flight["origin"]]],
            "dest": checkpoint_airport_map[id_to_code[flight["dest"]]],
        }
        for flight in flights
    ]
    remapped_bases = [
        checkpoint_airport_map[id_to_code[base_id]]
        for base_id in base_ids
        if id_to_code[base_id] in checkpoint_airport_map
    ]
    if not remapped_bases:
        raise ValueError("CPPSC base가 checkpoint airport_map에 없음")
    return remapped, remapped_bases
