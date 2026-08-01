"""
Convergence test: fixed seed + fixed constraint + compare first vs last
-- to confirm "does it converge or not" for the Monday meeting.
"""

import sys
import random
import torch
import torch.optim as optim
from torch.distributions import Categorical

# fix seed
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

from model import FlightEncoder, PointerDecoder

sys.path.insert(0, "RL")
from loader import load_flights
from environment import get_mask, step, final_reward
from RL.constraints import get_delta_constraints
from RL.state import init_state

PAIRING_PENALTY = -1.0
BASE_PENALTY = -2.0


def flights_to_tensors(flights):
    origins = torch.tensor([f["origin"] for f in flights])
    dests = torch.tensor([f["dest"] for f in flights])
    dep_times = torch.tensor([f["dep_time"] for f in flights], dtype=torch.float32)
    arr_times = torch.tensor([f["arr_time"] for f in flights], dtype=torch.float32)
    return origins, dests, dep_times, arr_times


def state_to_vec(state, encoder, constraint):
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
    assigned = {f["id"]: False for f in flights}
    state = init_state(flights, constraint)

    log_probs = []
    entropies = []
    total_reward = 0
    n_pairings = 0

    while True:
        mask_list = get_mask(state, flights, assigned, constraint)
        mask = torch.tensor(mask_list, dtype=torch.float32)

        if sum(mask_list[:-1]) == 0:
            n_pairings += 1
            total_reward += PAIRING_PENALTY

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

        if action == len(flights):
            n_pairings += 1

            if state["current_airport"] != constraint["base_airport"]:
                total_reward += BASE_PENALTY

            total_reward += PAIRING_PENALTY

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

        state, r, done = step(state, action, flights, assigned, constraint)
        total_reward += r

        if done:
            break

    total_reward += final_reward(assigned)
    return total_reward, log_probs, entropies, n_pairings


def test_convergence():
    flights = load_flights("RL/data/T_ONTIME_MARKETING.csv", limit=50)
    n_airports = max(max(f["origin"], f["dest"]) for f in flights) + 1

    # fix constraint (max_duty=10h)
    constraint = get_delta_constraints()
    constraint["max_duty"] = 10.0
    c_tensor = torch.tensor([constraint["max_duty"]], dtype=torch.float32)

    print("=== Convergence Test ===")
    print(f"flights: {len(flights)}, airports: {n_airports}")
    print(f"constraint: max_duty={constraint['max_duty']}h (fixed)")
    print(f"seed: {SEED}")
    print()

    encoder = FlightEncoder(n_airports=n_airports)
    decoder = PointerDecoder()

    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(params, lr=1e-4)

    origins, dests, dep_times, arr_times = flights_to_tensors(flights)

    greedy_pairings = []

    for episode in range(1500):
        encoded = encoder(origins, dests, dep_times, arr_times, c_tensor)

        reward_s, log_probs, entropies, pairings_s = run_episode(
            flights, constraint, encoder, decoder, encoded, greedy=False
        )

        if len(log_probs) == 0:
            continue

        with torch.no_grad():
            encoded_g = encoder(origins, dests, dep_times, arr_times, c_tensor)
            reward_g, _, _, pairings_g = run_episode(
                flights, constraint, encoder, decoder, encoded_g, greedy=True
            )

        greedy_pairings.append(pairings_g)

        advantage = (reward_s - reward_g) / (abs(reward_g) + 1e-6)

        loss = torch.stack([
            -lp * advantage - 0.005 * ent
            for lp, ent in zip(log_probs, entropies)
        ]).sum()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        optimizer.step()

        if episode % 100 == 0:
            recent = greedy_pairings[-50:] if len(greedy_pairings) >= 50 else greedy_pairings
            avg = sum(recent) / len(recent)
            print(
                f"Ep {episode:4d} | "
                f"greedy: {pairings_g:2d} (avg50: {avg:5.1f}) | "
                f"sample: {pairings_s:2d}"
            )

    # Convergence verdict
    print()
    print("=" * 60)
    print("Convergence Verdict")
    print("=" * 60)

    first100 = greedy_pairings[:100]
    last100 = greedy_pairings[-100:]
    mid100 = greedy_pairings[len(greedy_pairings)//2 - 50 : len(greedy_pairings)//2 + 50]

    avg_first = sum(first100) / len(first100)
    avg_mid = sum(mid100) / len(mid100)
    avg_last = sum(last100) / len(last100)
    best = min(greedy_pairings)

    print(f"  First 100 ep avg: {avg_first:.1f}")
    print(f"  Mid 100 ep avg: {avg_mid:.1f}")
    print(f"  Last 100 ep avg: {avg_last:.1f}")
    print(f"  Overall min: {best}")
    print()

    if avg_last < avg_first:
        print("  → Converging (last < first)")
    elif avg_last > avg_first * 1.1:
        print("  → Diverging (last is more than 10% worse than first)")
    else:
        print("  → Stalled (not decreasing)")

    # Also verify FiLM
    print()
    print("FiLM verification:")
    for duty in [6.0, 8.0, 10.0, 12.0, 14.0]:
        c = torch.tensor([duty], dtype=torch.float32)
        with torch.no_grad():
            enc = encoder(origins, dests, dep_times, arr_times, c)
            test_constraint = get_delta_constraints()
            test_constraint["max_duty"] = duty
            _, _, _, n_pair = run_episode(
                flights, test_constraint, encoder, decoder, enc, greedy=True
            )
        print(f"  max_duty={duty:4.0f}h → pairings: {n_pair:3d}")


if __name__ == "__main__":
    test_convergence()
