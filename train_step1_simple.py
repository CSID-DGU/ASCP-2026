"""
Curriculum Step 1: 50 flights + constraint 고정 + 단순 reward
reward = -len(pairings) 만 사용 (혜린 복합 reward와 비교용)
"""

import sys
import torch
import torch.optim as optim
from torch.distributions import Categorical

from model import FlightEncoder, PointerDecoder

sys.path.insert(0, "RL")
from loader import load_flights
from environment import get_mask, init_state


def flights_to_tensors(flights):
    origins = torch.tensor([f["origin"] for f in flights])
    dests = torch.tensor([f["dest"] for f in flights])
    dep_times = torch.tensor([f["dep_time"] for f in flights], dtype=torch.float32)
    arr_times = torch.tensor([f["arr_time"] for f in flights], dtype=torch.float32)
    return origins, dests, dep_times, arr_times


def state_to_vec(state, encoder):
    airport_emb = encoder.airport_emb(torch.tensor(state["current_airport"]))
    return torch.cat([
        airport_emb,
        torch.tensor([state["current_time"]], dtype=torch.float32),
        torch.tensor([state["duty_time"]], dtype=torch.float32),
    ])


def simple_step(state, action, flights, assigned):
    """단순 step: reward 없이 state만 업데이트"""
    f = flights[action]
    assigned[f["id"]] = True
    flight_time = f["arr_time"] - f["dep_time"]
    next_state = {
        "current_airport": f["dest"],
        "current_time": f["arr_time"],
        "duty_time": state["duty_time"] + flight_time,
        "remaining": state["remaining"] - 1,
    }
    return next_state


def run_episode(flights, constraint, encoder, decoder, encoded, greedy=False):
    assigned = {f["id"]: False for f in flights}
    state = init_state(flights)

    log_probs = []
    entropies = []
    n_pairings = 0

    while True:
        mask_list = get_mask(state, flights, assigned, constraint)
        mask = torch.tensor(mask_list, dtype=torch.float32)

        if sum(mask_list[:-1]) == 0:
            n_pairings += 1

            unassigned = [f for f in flights if not assigned[f["id"]]]
            if len(unassigned) == 0:
                break

            start = unassigned[0]
            assigned[start["id"]] = True
            state = {
                "current_airport": start["dest"],
                "current_time": start["arr_time"],
                "duty_time": start["arr_time"] - start["dep_time"],
                "remaining": sum(1 for v in assigned.values() if not v),
            }
            continue

        state_vec = state_to_vec(state, encoder)
        probs = decoder(encoded, state_vec, mask)

        if greedy:
            action = probs.argmax().item()
        else:
            dist = Categorical(probs)
            a = dist.sample()
            log_probs.append(dist.log_prob(a))
            entropies.append(dist.entropy())
            action = a.item()

        if action == len(flights):
            n_pairings += 1

            unassigned = [f for f in flights if not assigned[f["id"]]]
            if len(unassigned) == 0:
                break

            start = unassigned[0]
            assigned[start["id"]] = True
            state = {
                "current_airport": start["dest"],
                "current_time": start["arr_time"],
                "duty_time": start["arr_time"] - start["dep_time"],
                "remaining": sum(1 for v in assigned.values() if not v),
            }
            continue

        state = simple_step(state, action, flights, assigned)

    # 단순 reward: -len(pairings)
    reward = -n_pairings
    return reward, log_probs, entropies, n_pairings


def train():
    flights = load_flights("RL/data/T_ONTIME_MARKETING.csv", limit=50)
    n_airports = max(max(f["origin"], f["dest"]) for f in flights) + 1

    CONSTRAINT = {"max_duty": 10.0}
    c_tensor = torch.tensor([CONSTRAINT["max_duty"]], dtype=torch.float32)

    print(f"=== Step 1 — 단순 reward: -len(pairings) ===")
    print(f"flights: {len(flights)}개, airports: {n_airports}개")
    print(f"constraint: max_duty={CONSTRAINT['max_duty']}h (고정)")
    print()

    encoder = FlightEncoder(n_airports=n_airports)
    decoder = PointerDecoder()

    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(params, lr=1e-4)

    origins, dests, dep_times, arr_times = flights_to_tensors(flights)

    greedy_pairings = []

    for episode in range(1000):
        encoded = encoder(origins, dests, dep_times, arr_times, c_tensor)

        reward_s, log_probs, entropies, pairings_s = run_episode(
            flights, CONSTRAINT, encoder, decoder, encoded, greedy=False
        )

        if len(log_probs) == 0:
            continue

        with torch.no_grad():
            encoded_g = encoder(origins, dests, dep_times, arr_times, c_tensor)
            reward_g, _, _, pairings_g = run_episode(
                flights, CONSTRAINT, encoder, decoder, encoded_g, greedy=True
            )

        greedy_pairings.append(pairings_g)

        advantage = (reward_s - reward_g) / 10.0

        loss = torch.stack([
            -lp * advantage - 0.01 * ent
            for lp, ent in zip(log_probs, entropies)
        ]).sum()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        optimizer.step()

        if episode % 50 == 0:
            recent_p = greedy_pairings[-50:] if len(greedy_pairings) >= 50 else greedy_pairings
            avg_p = sum(recent_p) / len(recent_p)

            print(
                f"Ep {episode:4d} | "
                f"greedy pairings: {pairings_g:2d} (avg50: {avg_p:5.1f}) | "
                f"sample pairings: {pairings_s:2d} | "
                f"adv: {advantage:6.2f}"
            )

    print()
    print("=" * 60)
    print("Step 1 결과 (단순 reward)")
    print("=" * 60)

    first50 = greedy_pairings[:50]
    last50 = greedy_pairings[-50:]
    print(f"  처음 50ep 평균 pairings: {sum(first50)/len(first50):.1f}")
    print(f"  마지막 50ep 평균 pairings: {sum(last50)/len(last50):.1f}")

    improved = sum(first50)/len(first50) > sum(last50)/len(last50)
    print(f"  수렴 여부: {'줄어듦 (수렴 중)' if improved else '안 줄어듦 (수렴 안 함)'}")

    print()
    print("FiLM 검증:")
    for duty in [6.0, 8.0, 10.0, 12.0, 14.0]:
        c = torch.tensor([duty], dtype=torch.float32)
        with torch.no_grad():
            enc = encoder(origins, dests, dep_times, arr_times, c)
            _, _, _, n_pair = run_episode(
                flights, {"max_duty": duty}, encoder, decoder, enc, greedy=True
            )
        print(f"  max_duty={duty:4.0f}h → pairings: {n_pair:3d}")


if __name__ == "__main__":
    train()
