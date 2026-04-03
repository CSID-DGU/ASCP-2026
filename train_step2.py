"""
Curriculum Step 2: 300 flights + constraint 고정 (max_duty=10h)
목표: 스케일업해도 수렴하는지 확인
"""

import sys
import random
import torch
import torch.optim as optim
from torch.distributions import Categorical

# 시드 고정 — 매번 같은 결과 나오도록
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

from model import FlightEncoder, PointerDecoder

sys.path.insert(0, "RL")
from loader import load_flights
from environment import get_mask, step, init_state, final_reward


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


def run_episode(flights, constraint, encoder, decoder, encoded, greedy=False):
    assigned = {f["id"]: False for f in flights}
    state = init_state(flights)

    log_probs = []
    entropies = []
    total_reward = 0
    n_pairings = 0

    while True:
        mask_list = get_mask(state, flights, assigned, constraint)
        mask = torch.tensor(mask_list, dtype=torch.float32)

        if sum(mask_list[:-1]) == 0:
            n_pairings += 1
            total_reward += -1.0

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
            total_reward += -1.0

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

        state, r, done = step(state, action, flights, assigned, constraint)
        total_reward += r

        if done:
            break

    total_reward += final_reward(assigned)
    return total_reward, log_probs, entropies, n_pairings


def train():
    # ★ 300 flights로 스케일업
    flights = load_flights("RL/data/T_ONTIME_MARKETING.csv", limit=300)
    n_airports = max(max(f["origin"], f["dest"]) for f in flights) + 1

    CONSTRAINT = {"max_duty": 10.0}
    c_tensor = torch.tensor([CONSTRAINT["max_duty"]], dtype=torch.float32)

    print(f"=== Curriculum Step 2 ===")
    print(f"flights: {len(flights)}개, airports: {n_airports}개")
    print(f"constraint: max_duty={CONSTRAINT['max_duty']}h (고정)")
    print()

    encoder = FlightEncoder(n_airports=n_airports)
    decoder = PointerDecoder()

    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(params, lr=1e-4)

    origins, dests, dep_times, arr_times = flights_to_tensors(flights)

    greedy_rewards = []
    greedy_pairings = []

    for episode in range(500):
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

        greedy_rewards.append(reward_g)
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

        if episode % 25 == 0:
            recent_p = greedy_pairings[-25:] if len(greedy_pairings) >= 25 else greedy_pairings
            avg_p = sum(recent_p) / len(recent_p)

            print(
                f"Ep {episode:4d} | "
                f"greedy pairings: {pairings_g:3d} (avg25: {avg_p:6.1f}) | "
                f"sample pairings: {pairings_s:3d} | "
                f"adv: {advantage:6.2f}"
            )

    print()
    print("=" * 60)
    print("Step 2 결과")
    print("=" * 60)

    first50 = greedy_pairings[:50]
    last50 = greedy_pairings[-50:]
    print(f"  처음 50ep 평균 pairings: {sum(first50)/len(first50):.1f}")
    print(f"  마지막 50ep 평균 pairings: {sum(last50)/len(last50):.1f}")

    improved = sum(first50)/len(first50) > sum(last50)/len(last50)
    print(f"  수렴 여부: {'줄어듦 (수렴 중)' if improved else '안 줄어듦 (수렴 안 함)'}")


if __name__ == "__main__":
    train()
