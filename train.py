"""
찬주 model/ + 혜린 RL/environment 통합 학습 스크립트
- encoder: 찬주 (FlightEncoder — Embedding + FiLM + Transformer)
- decoder: 찬주 (PointerDecoder — Pointer Attention + hard masking)
- environment: 혜린 (mask + step + reward)
- loader: 혜린 (BTS CSV → flight dict)
"""

import sys
import torch
import torch.optim as optim
from torch.distributions import Categorical

# 찬주 모델 (model/ 폴더 먼저 잡히도록)
from model import FlightEncoder, PointerDecoder

# 혜린 코드 (RL/ 폴더)
sys.path.insert(0, "RL")
from loader import load_flights
from environment import get_mask, step, final_reward
from RL.constraints import get_delta_constraints, FILM_CONSTRAINT_KEYS
from RL.state import init_state

PAIRING_PENALTY = -1.0
BASE_PENALTY = -2.0


def constraint_to_tensor(constraint):
    """constraint dict → FiLM 입력 tensor"""
    return torch.tensor([constraint[k] for k in FILM_CONSTRAINT_KEYS], dtype=torch.float32)

def flights_to_tensors(flights):
    """혜린 flight dict → 찬주 encoder 입력 tensor 변환"""
    origins = torch.tensor([f["origin"] for f in flights])
    dests = torch.tensor([f["dest"] for f in flights])
    dep_times = torch.tensor([f["dep_time"] for f in flights], dtype=torch.float32)
    arr_times = torch.tensor([f["arr_time"] for f in flights], dtype=torch.float32)
    return origins, dests, dep_times, arr_times


def state_to_vec(state, encoder, constraint):
    """혜린 state dict → 찬주 decoder 입력 tensor 변환"""
    airport_emb = encoder.airport_emb(torch.tensor(state["current_airport"]))
    
    return torch.cat([
        airport_emb,
        torch.tensor([
            state["current_time"] / 24.0,
            state["duty_time"] / constraint["max_duty"],
            state["legs"] / constraint["max_legs"],
        ], dtype=torch.float32)
    ])


def run_episode(flights, constraint, encoder, decoder, encoded, greedy=False):
    """
    혜린 environment + 찬주 model로 에피소드 진행

    Returns:
        total_reward, log_probs, entropies, n_pairings
    """
    assigned = {f["id"]: False for f in flights}
    state = init_state(flights, constraint)

    log_probs = []
    entropies = []
    total_reward = 0
    n_pairings = 0

    while True:
        # 혜린 mask
        mask_list = get_mask(state, flights, assigned, constraint)
        mask = torch.tensor(mask_list, dtype=torch.float32)

        # valid한 flight가 없으면 (END만 남으면) 종료
        if sum(mask_list[:-1]) == 0:
            n_pairings += 1
            total_reward += PAIRING_PENALTY

            # 미배정 flight 남아있으면 새 pairing 시작
            unassigned = [f for f in flights if not assigned[f["id"]]]
            if len(unassigned) == 0:
                break

            # 가장 이른 미배정 flight로 새 pairing 시작
            start = sorted(unassigned, key=lambda x: x["dep_time"])[0] # dep_time 기준 정렬 필요
            assigned[start["id"]] = True

            state = {
                "current_airport": start["dest"],
                "current_time": start["arr_time"],
                "duty_time": start["arr_time"] - start["dep_time"],
                "duty_start_time": start["dep_time"],
                "legs": 1,
                "remaining": sum(1 for v in assigned.values() if not v),
            }
            continue

        # 찬주 decoder
        state_vec = state_to_vec(state, encoder, constraint)
        probs = decoder(encoded, state_vec, mask)

        if greedy:
            action = probs.argmax().item()
        else:
            dist = Categorical(probs)
            a = dist.sample()
            log_probs.append(dist.log_prob(a))
            entropies.append(dist.entropy())
            action = a.item()

        # END action
        if action == len(flights):
            n_pairings += 1
            
            if state["current_airport"] != constraint["base_airport"]:
                total_reward += BASE_PENALTY

            total_reward += PAIRING_PENALTY

            # 미배정 flight 남아있으면 새 pairing 시작
            unassigned = [f for f in flights if not assigned[f["id"]]]
            if len(unassigned) == 0:
                break

            start = sorted(unassigned, key=lambda x: x["dep_time"])[0]
            assigned[start["id"]] = True
            state = {
                "current_airport": start["dest"],
                "current_time": start["arr_time"],
                "duty_time": start["arr_time"] - start["dep_time"],
                "duty_start_time": start["dep_time"],
                "legs": 1,
                "remaining": sum(1 for v in assigned.values() if not v),
            }
            continue

        # 혜린 step (assigned도 내부에서 업데이트)
        state, r, done = step(state, action, flights, assigned, constraint)
        total_reward += r

        if done:
            break

    # 혜린 final reward
    total_reward += final_reward(assigned)

    return total_reward, log_probs, entropies, n_pairings


def train():
    # 데이터 로드 (혜린 loader)
    flights = load_flights("RL/data/T_ONTIME_MARKETING.csv", limit=50)
    n_airports = max(max(f["origin"], f["dest"]) for f in flights) + 1

    print(f"flights: {len(flights)}개, airports: {n_airports}개")
    print()

    # 찬주 모델 생성
    encoder = FlightEncoder(
        n_airports=n_airports,
        constraint_dim=len(FILM_CONSTRAINT_KEYS),  # 4개: max_duty, min_conn, max_conn, max_legs
        airport_emb_dim=32,
        d_model=128,
    )
    decoder = PointerDecoder(d_model=128, airport_emb_dim=32)

    # 전체 파라미터 합쳐서 optimizer
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(params, lr=1e-4)  # lr 낮춤 (안정성)

    # flight → tensor (1번만)
    origins, dests, dep_times, arr_times = flights_to_tensors(flights)

    for episode in range(1000):
        # constraint 번갈아 (FiLM 검증용)
        constraint = get_delta_constraints()
        c_tensor = constraint_to_tensor(constraint)

        # encode (에피소드당 1번)
        encoded = encoder(origins, dests, dep_times, arr_times, c_tensor)

        # ── sample rollout (학습용) ──
        reward_s, log_probs, entropies, pairings_s = run_episode(
            flights, constraint, encoder, decoder, encoded, greedy=False
        )

        if len(log_probs) == 0:
            continue

        # ── greedy rollout (baseline) ──
        with torch.no_grad():
            encoded_g = encoder(origins, dests, dep_times, arr_times, c_tensor)
            reward_g, _, _, pairings_g = run_episode(
                flights, constraint, encoder, decoder, encoded_g, greedy=True
            )

        # scaling (reward 범위에 따라 advantage가 너무 커지는 문제 방지)
        # advantage = sample - greedy
        advantage = (reward_s - reward_g) / (abs(reward_g) + 1e-6)

        # REINFORCE loss + entropy bonus
        loss = torch.stack([
            -lp * advantage - 0.005 * ent
            for lp, ent in zip(log_probs, entropies)
        ]).sum()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)  # gradient clipping
        optimizer.step()

        if episode % 20 == 0:
            print(
                f"Ep {episode:4d} | "
                f"sample: {reward_s:7.1f} (p={pairings_s:2d}) | "
                f"greedy: {reward_g:7.1f} (p={pairings_g:2d}) | "
                f"adv: {advantage:6.2f} | "
                f"duty: {constraint['max_duty']:2.0f}h | "
                f"loss: {loss.item():8.3f}"
            )

    # ── FiLM 검증: constraint별 greedy 결과 비교 ──
    print()
    print("=" * 60)
    print("FiLM 검증: 같은 flights, 다른 constraint")
    print("=" * 60)

    for duty in [6.0, 8.0, 10.0, 12.0, 14.0]:
        with torch.no_grad():
            test_constraint = get_delta_constraints()
            test_constraint["max_duty"] = duty
            c = constraint_to_tensor(test_constraint)

            enc = encoder(origins, dests, dep_times, arr_times, c)
            reward, _, _, n_pair = run_episode(
                flights,
                test_constraint,
                encoder,
                decoder,
                enc,
                greedy=True
            )
        print(f"  max_duty={duty:4.0f}h → pairings: {n_pair:3d}, reward: {reward:7.1f}")


if __name__ == "__main__":
    train()
