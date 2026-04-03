import torch
import torch.optim as optim
from torch.distributions import Categorical
import random

from loader import load_flights
from environment import *
from model import PolicyNetwork


def train():
    flights = load_flights("data/T_ONTIME_MARKETING.csv", limit=50)

    flight_features = [
        [f["origin"], f["dest"], f["dep_time"], f["arr_time"]]
        for f in flights
    ]

    model = PolicyNetwork()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for episode in range(500):

        constraint = {"max_duty": 8 if episode % 2 == 0 else 14}

        encoded = model.encode(flight_features, constraint)

        log_probs = []
        entropies = []

        _, reward_sample = run_episode(
            flights, constraint, model, encoded,
            log_probs, entropies, greedy=False
        )

        # ---------------------------
        # ✔ reward scaling
        # ---------------------------
        reward_sample = reward_sample / 10.0

        # ---------------------------
        # ✔ advantage
        # ---------------------------
        advantage = reward_sample

        if len(log_probs) == 0:
            print("SKIPPED")
            continue

        # ---------------------------
        # ✔ loss (entropy 강화)
        # ---------------------------
        loss = torch.stack([
            -lp * advantage - 0.2 * ent
            for lp, ent in zip(log_probs, entropies)
        ]).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if episode % 10 == 0:
            print(
                f"Episode {episode} | reward: {reward_sample:.3f} "
                f"| A: {advantage:.3f}"
            )


def run_episode(flights, constraint, model, encoded,
                log_probs, entropies, greedy=False):

    assigned = {f["id"]: False for f in flights}

    state = init_state(flights)
    state["remaining"] = len(flights)

    total_reward = 0

    while True:

        mask = get_mask(state, flights, assigned, constraint)

        probs = model.decode(encoded, state, mask)

        # ---------------------------
        # ✔ temperature (탐색 강화)
        # ---------------------------
        temperature = 2.0
        probs = torch.softmax(torch.log(probs + 1e-8) / temperature, dim=-1)

        if greedy:
            action = torch.argmax(probs).item()
        else:
            # ---------------------------
            # ✔ epsilon-greedy
            # ---------------------------
            if random.random() < 0.2:
                action = torch.randint(0, len(probs), (1,)).item()
            else:
                dist = Categorical(probs)
                a = dist.sample()

                log_probs.append(dist.log_prob(a))
                entropies.append(dist.entropy())

                action = a.item()

        # ---------------------------
        # ✔ END action
        # ---------------------------
        if action == len(flights):
            total_reward += -1.0
            break

        state, r, done = step(state, action, flights, assigned, constraint)
        total_reward += r

        if done:
            break

    # ---------------------------
    # ✔ final reward
    # ---------------------------
    total_reward += final_reward(assigned)

    return None, total_reward


if __name__ == "__main__":
    train()