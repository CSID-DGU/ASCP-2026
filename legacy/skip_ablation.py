"""
Skip Connection Ablation experiment script

Trains each of the 5 configurations independently and saves the results to JSON.

Experiment configurations:
  baseline          : no skip (control)
  film_skip         : skip only in the FiLM block
  transformer_skip  : skip only in the Transformer block
  decoder_skip      : skip only in the Decoder state_mlp
  all_skip          : skip in all three places

Usage:
  source venv/bin/activate
  python experiments/skip_ablation.py

Output:
  experiments/results/skip_ablation_results.json
"""

import sys
import os
import json
import time
import torch
import torch.optim as optim
from torch.distributions import Categorical

# Add project root to the path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "RL"))

from model import FlightEncoder, PointerDecoder
from loader import load_flights
from environment import get_mask, step, final_reward
from RL.constraints import get_delta_constraints, FILM_CONSTRAINT_KEYS
from RL.state import init_state

# -- Training hyperparameters -----------------------------------------------
N_EPISODES   = 600      # episodes per configuration
LOG_INTERVAL = 20       # logging interval
FLIGHT_LIMIT = 50       # data size (same as train.py)
LR           = 1e-4
GRAD_CLIP    = 1.0
ENTROPY_COEF = 0.005
PAIRING_PENALTY = -1.0
BASE_PENALTY    = -2.0

# -- Experiment configuration definitions ------------------------------------
CONFIGS = {
    "baseline":         dict(skip_film=False, skip_transformer=False, skip_state_mlp=False),
    "film_skip":        dict(skip_film=True,  skip_transformer=False, skip_state_mlp=False),
    "transformer_skip": dict(skip_film=False, skip_transformer=True,  skip_state_mlp=False),
    "decoder_skip":     dict(skip_film=False, skip_transformer=False, skip_state_mlp=True),
    "all_skip":         dict(skip_film=True,  skip_transformer=True,  skip_state_mlp=True),
}


# -- Helper functions (same as train.py) -------------------------------------
def constraint_to_tensor(constraint):
    return torch.tensor([constraint[k] for k in FILM_CONSTRAINT_KEYS], dtype=torch.float32)


def flights_to_tensors(flights):
    origins   = torch.tensor([f["origin"]   for f in flights])
    dests     = torch.tensor([f["dest"]     for f in flights])
    dep_times = torch.tensor([f["dep_time"] for f in flights], dtype=torch.float32)
    arr_times = torch.tensor([f["arr_time"] for f in flights], dtype=torch.float32)
    return origins, dests, dep_times, arr_times


def state_to_vec(state, encoder, constraint):
    airport_emb = encoder.airport_emb(torch.tensor(state["current_airport"]))
    return torch.cat([
        airport_emb,
        torch.tensor([
            state["current_time"]  / 24.0,
            state["duty_time"]     / constraint["max_duty"],
            state["legs"]          / constraint["max_legs"],
        ], dtype=torch.float32),
    ])


def run_episode(flights, constraint, encoder, decoder, encoded, greedy=False):
    assigned = {f["id"]: False for f in flights}
    state    = init_state(flights, constraint)

    log_probs, entropies = [], []
    total_reward = 0
    n_pairings   = 0

    while True:
        mask_list = get_mask(state, flights, assigned, constraint)
        mask      = torch.tensor(mask_list, dtype=torch.float32)

        if sum(mask_list[:-1]) == 0:
            n_pairings   += 1
            total_reward += PAIRING_PENALTY
            unassigned = [f for f in flights if not assigned[f["id"]]]
            if not unassigned:
                break
            start = sorted(unassigned, key=lambda x: x["dep_time"])[0]
            assigned[start["id"]] = True
            state = {
                "current_airport":  start["dest"],
                "current_time":     start["arr_time"],
                "duty_time":        start["arr_time"] - start["dep_time"],
                "duty_start_time":  start["dep_time"],
                "legs":             1,
                "remaining":        sum(1 for v in assigned.values() if not v),
            }
            continue

        state_vec = state_to_vec(state, encoder, constraint)
        probs     = decoder(encoded, state_vec, mask)

        if greedy:
            action = probs.argmax().item()
        else:
            dist   = Categorical(probs)
            a      = dist.sample()
            log_probs.append(dist.log_prob(a))
            entropies.append(dist.entropy())
            action = a.item()

        if action == len(flights):
            n_pairings += 1
            if state["current_airport"] != constraint["base_airport"]:
                total_reward += BASE_PENALTY
            total_reward += PAIRING_PENALTY
            unassigned = [f for f in flights if not assigned[f["id"]]]
            if not unassigned:
                break
            start = sorted(unassigned, key=lambda x: x["dep_time"])[0]
            assigned[start["id"]] = True
            state = {
                "current_airport":  start["dest"],
                "current_time":     start["arr_time"],
                "duty_time":        start["arr_time"] - start["dep_time"],
                "duty_start_time":  start["dep_time"],
                "legs":             1,
                "remaining":        sum(1 for v in assigned.values() if not v),
            }
            continue

        state, r, done = step(state, action, flights, assigned, constraint)
        total_reward  += r
        if done:
            break

    total_reward += final_reward(assigned)
    return total_reward, log_probs, entropies, n_pairings


# -- Train a single configuration ---------------------------------------------
def run_config(config_name, cfg, flights, n_airports):
    print(f"\n{'='*60}")
    print(f"  Experiment: {config_name}")
    print(f"  skip_film={cfg['skip_film']}  skip_transformer={cfg['skip_transformer']}  skip_state_mlp={cfg['skip_state_mlp']}")
    print(f"{'='*60}")

    torch.manual_seed(42)   # fix seed for reproducibility

    encoder = FlightEncoder(
        n_airports=n_airports,
        constraint_dim=len(FILM_CONSTRAINT_KEYS),
        airport_emb_dim=32,
        d_model=128,
        skip_film=cfg["skip_film"],
        skip_transformer=cfg["skip_transformer"],
    )
    decoder = PointerDecoder(
        d_model=128,
        airport_emb_dim=32,
        skip_state_mlp=cfg["skip_state_mlp"],
    )

    params    = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(params, lr=LR)
    origins, dests, dep_times, arr_times = flights_to_tensors(flights)

    history = []   # {episode, reward_sample, reward_greedy, pairings_sample, pairings_greedy}
    t0 = time.time()

    for episode in range(N_EPISODES):
        constraint = get_delta_constraints()
        c_tensor   = constraint_to_tensor(constraint)

        encoded = encoder(origins, dests, dep_times, arr_times, c_tensor)
        reward_s, log_probs, entropies, pairings_s = run_episode(
            flights, constraint, encoder, decoder, encoded, greedy=False
        )

        if not log_probs:
            continue

        with torch.no_grad():
            encoded_g = encoder(origins, dests, dep_times, arr_times, c_tensor)
            reward_g, _, _, pairings_g = run_episode(
                flights, constraint, encoder, decoder, encoded_g, greedy=True
            )

        advantage = (reward_s - reward_g) / (abs(reward_g) + 1e-6)
        loss = torch.stack([
            -lp * advantage - ENTROPY_COEF * ent
            for lp, ent in zip(log_probs, entropies)
        ]).sum()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
        optimizer.step()

        if episode % LOG_INTERVAL == 0:
            elapsed = time.time() - t0
            print(
                f"  Ep {episode:4d} | "
                f"sample: {reward_s:7.1f} (p={pairings_s:2d}) | "
                f"greedy: {reward_g:7.1f} (p={pairings_g:2d}) | "
                f"adv: {advantage:6.3f} | "
                f"loss: {loss.item():8.3f} | "
                f"{elapsed:.0f}s"
            )
            history.append({
                "episode":        episode,
                "reward_sample":  round(reward_s, 3),
                "reward_greedy":  round(reward_g, 3),
                "pairings_sample": pairings_s,
                "pairings_greedy": pairings_g,
                "loss":           round(loss.item(), 4),
            })

    # -- Final evaluation per constraint after training --
    final_eval = {}
    for duty in [6.0, 8.0, 10.0, 12.0, 14.0]:
        with torch.no_grad():
            tc = get_delta_constraints()
            tc["max_duty"] = duty
            c  = constraint_to_tensor(tc)
            enc = encoder(origins, dests, dep_times, arr_times, c)
            r, _, _, np_ = run_episode(flights, tc, encoder, decoder, enc, greedy=True)
        final_eval[str(duty)] = {"pairings": np_, "reward": round(r, 3)}
        print(f"  max_duty={duty:4.0f}h → pairings: {np_:3d}, reward: {r:7.1f}")

    total_time = round(time.time() - t0, 1)
    print(f"  Done: {total_time}s")

    return {
        "config":     cfg,
        "history":    history,
        "final_eval": final_eval,
        "total_time_sec": total_time,
    }


# -- Main -----------------------------------------------------------------------
def main():
    print("Loading data...")
    flights    = load_flights("RL/data/T_ONTIME_MARKETING.csv", limit=FLIGHT_LIMIT)
    n_airports = max(max(f["origin"], f["dest"]) for f in flights) + 1
    print(f"flights: {len(flights)}, airports: {n_airports}")

    all_results = {}

    for config_name, cfg in CONFIGS.items():
        result = run_config(config_name, cfg, flights, n_airports)
        all_results[config_name] = result

    # Save results
    out_path = os.path.join(ROOT, "experiments", "results", "skip_ablation_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved: {out_path}")

    # Print brief summary
    print("\n" + "="*60)
    print("Experiment summary (greedy, at max_duty=14h)")
    print("="*60)
    print(f"{'Config':<22} {'Final pairings':>14} {'Final reward':>12} {'Train time':>10}")
    print("-"*60)
    for name, res in all_results.items():
        p = res["final_eval"]["14.0"]["pairings"]
        r = res["final_eval"]["14.0"]["reward"]
        t = res["total_time_sec"]
        print(f"{name:<22} {p:>14d} {r:>12.1f} {t:>9.1f}s")


if __name__ == "__main__":
    main()
