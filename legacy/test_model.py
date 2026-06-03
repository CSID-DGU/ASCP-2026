"""목요일(0402) 완료 기준 테스트:
- FiLM: dummy input으로 constraint 바꾸면 출력 바뀜
- Transformer: flight 벡터 입출력 shape 정상
- Pointer: dummy query로 probs 출력, 합 = 1
"""

import torch
from model import FlightEncoder, PointerDecoder

# 설정
N_FLIGHTS = 10
N_AIRPORTS = 20
D_MODEL = 128
AIRPORT_EMB_DIM = 32
CONSTRAINT_DIM = 1


def test_encoder():
    print("=" * 50)
    print("Encoder 테스트")
    print("=" * 50)

    encoder = FlightEncoder(
        n_airports=N_AIRPORTS,
        constraint_dim=CONSTRAINT_DIM,
        airport_emb_dim=AIRPORT_EMB_DIM,
        d_model=D_MODEL,
    )

    # dummy flights
    origins = torch.randint(0, N_AIRPORTS, (N_FLIGHTS,))
    dests = torch.randint(0, N_AIRPORTS, (N_FLIGHTS,))
    dep_times = torch.rand(N_FLIGHTS)
    arr_times = dep_times + torch.rand(N_FLIGHTS) * 0.3

    # 테스트 1: shape 확인
    constraint = torch.tensor([10.0])
    encoded = encoder(origins, dests, dep_times, arr_times, constraint)
    assert encoded.shape == (N_FLIGHTS, D_MODEL), \
        f"[FAIL] encoded shape: {encoded.shape} (expected: ({N_FLIGHTS}, {D_MODEL}))"
    print(f"[PASS] encoded shape: {encoded.shape}")

    # 테스트 2: FiLM — constraint 바꾸면 출력 바뀜
    constraint_strict = torch.tensor([8.0])
    constraint_relaxed = torch.tensor([14.0])
    enc_strict = encoder(origins, dests, dep_times, arr_times, constraint_strict)
    enc_relaxed = encoder(origins, dests, dep_times, arr_times, constraint_relaxed)

    diff = (enc_strict - enc_relaxed).abs().mean().item()
    assert diff > 0, \
        f"[FAIL] constraint 8h vs 14h 출력이 동일함 (diff={diff})"
    print(f"[PASS] constraint 8h vs 14h 출력 차이: {diff:.4f}")

    # 테스트 3: cosine similarity (참고용, assert 없음)
    cos_sim = torch.nn.functional.cosine_similarity(
        enc_strict.flatten().unsqueeze(0),
        enc_relaxed.flatten().unsqueeze(0),
    ).item()
    print(f"[INFO] cosine similarity: {cos_sim:.4f} (학습 전이라 높을 수 있음, 목표: < 0.9)")

    return encoder, encoded


def test_decoder(encoder, encoded):
    print()
    print("=" * 50)
    print("Decoder 테스트")
    print("=" * 50)

    decoder = PointerDecoder(d_model=D_MODEL, airport_emb_dim=AIRPORT_EMB_DIM)

    # dummy state: airport_emb(32) + current_time(1) + duty_time(1)
    state_vec = torch.randn(AIRPORT_EMB_DIM + 2)

    # mask: 처음 5개만 valid + END_PAIRING
    mask = torch.zeros(N_FLIGHTS + 1)
    mask[:5] = 1    # flight 0~4 valid
    mask[-1] = 1    # END_PAIRING valid

    probs = decoder(encoded, state_vec, mask)

    # 테스트 1: shape
    assert probs.shape == (N_FLIGHTS + 1,), \
        f"[FAIL] probs shape: {probs.shape} (expected: ({N_FLIGHTS + 1},))"
    print(f"[PASS] probs shape: {probs.shape}")

    # 테스트 2: 합 = 1
    prob_sum = probs.sum().item()
    assert abs(prob_sum - 1.0) < 1e-5, \
        f"[FAIL] probs sum: {prob_sum:.6f} (expected: 1.0)"
    print(f"[PASS] probs sum: {prob_sum:.6f}")

    # 테스트 3: invalid flight prob = 0
    invalid_probs = probs[5:N_FLIGHTS]  # flight 5~9는 masked
    max_invalid = invalid_probs.max().item()
    assert max_invalid < 1e-6, \
        f"[FAIL] invalid flight max prob: {max_invalid:.8f} (expected: ~0)"
    print(f"[PASS] invalid flight max prob: {max_invalid:.8f}")

    # 테스트 4: valid flight prob > 0
    valid_probs = probs[mask == 1]
    min_valid = valid_probs.min().item()
    assert min_valid > 0, \
        f"[FAIL] valid flight min prob: {min_valid:.6f} (expected: > 0)"
    print(f"[PASS] valid flight min prob: {min_valid:.6f}")

    print()
    print(f"probs 전체: {[round(p, 4) for p in probs.detach().tolist()]}")


if __name__ == "__main__":
    encoder, encoded = test_encoder()
    test_decoder(encoder, encoded)
    print()
    print("=" * 50)
    print("모든 테스트 통과함")
    print("=" * 50)
