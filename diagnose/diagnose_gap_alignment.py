"""
diagnose_gap_alignment.py -- intra-duty gap diagnostics (using an existing checkpoint,
no retraining).

Execution script for Step 1 (candidate gap distribution) + Step 2 (score-gap
correlation) from log/0706/FTC_gap_diagnosis_execution_plan.md.

Step 1: at every intra-duty connection selection point, measures the fraction of
        decision points where "the chosen gap == the minimum" among the mask-passing
        candidates -- a low fraction means "a shorter candidate existed but a
        different one was picked."
Step 2: at the same decision points, measures the Spearman correlation between the
        decoder's raw score (logit) and the actual gap -- a weak correlation means the
        model doesn't distinguish gaps well (hypothesis B in
        FTC_root_cause_model_design_analysis.md).
"""
import os
import sys
import argparse

import torch
from torch.distributions import Categorical
from scipy.stats import spearmanr

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "RL"))

from model import FlightEncoder, PointerDecoder
from loader import build_airport_map, bases_to_ids, load_flights_rolling
from constraints import get_delta_constraints, FILM_CONSTRAINT_KEYS
from utils import flights_to_tensors, constraint_to_tensor, state_to_vec
import environment as env
import config


def load_model(checkpoint, device):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
    n_airports = ckpt.get("n_airports", ckpt["encoder"]["airport_emb.weight"].shape[0])
    encoder = FlightEncoder(n_airports=n_airports, constraint_dim=len(FILM_CONSTRAINT_KEYS)).to(device)
    airport_emb_dim = encoder.airport_emb.embedding_dim
    ckpt_state_dim = ckpt["decoder"]["state_mlp.0.weight"].shape[1]
    n_scalars = ckpt_state_dim - airport_emb_dim * 2 - len(FILM_CONSTRAINT_KEYS)
    decoder = PointerDecoder(constraint_dim=len(FILM_CONSTRAINT_KEYS),
                              airport_emb_dim=airport_emb_dim, n_scalars=n_scalars).to(device)
    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])
    encoder.eval()
    decoder.eval()
    return encoder, decoder


def new_pairing_state(f, episode_base, assigned):
    return {
        "current_airport": f["dest"], "current_time": f["arr_time"],
        "duty_time": f["arr_time"] - f["dep_time"], "duty_start_time": f["dep_time"],
        "legs": 1, "remaining": sum(1 for v in assigned.values() if not v),
        "pairing_start": False, "duty_period": 0,
        "pairing_start_time": f["dep_time"], "is_resting": False, "rest_end_time": None,
        "base_airport": episode_base,
    }


def rollout_with_diagnostics(flights, constraint, encoder, decoder, encoded, device, greedy=False):
    """Records (chosen gap, candidate gap list, candidate score list) for every intra-duty connection selection."""
    assigned = {f["id"]: False for f in flights}
    records = []

    episode_base = constraint.get("base_airport", 0)
    unassigned = [f for f in flights if not assigned[f["id"]]]
    base_flights = [f for f in unassigned if f["origin"] == episode_base]
    first = sorted(base_flights or unassigned, key=lambda f: f["dep_time"])[0]
    assigned[first["id"]] = True
    state = new_pairing_state(first, episode_base, assigned)
    state["pairing_start"] = False  # the first flight is treated as the flight itself, not pairing_start (no gap penalty case)

    _incl_total = decoder.state_mlp[0].weight.shape[1] > 78
    max_steps = len(flights) * 4
    steps = 0
    while steps < max_steps:
        steps += 1
        mask_list = env.get_mask(state, flights, assigned, constraint)
        mask = torch.tensor(mask_list, dtype=torch.float32).to(device)

        if sum(mask_list[:-2]) == 0 and mask_list[-2] == 0 and mask_list[-1] == 0:
            unassigned = [f for f in flights if not assigned[f["id"]]]
            if not unassigned:
                break
            nxt = sorted([f for f in unassigned if f["origin"] == episode_base] or unassigned,
                         key=lambda f: f["dep_time"])[0]
            assigned[nxt["id"]] = True
            state = new_pairing_state(nxt, episode_base, assigned)
            continue

        state_vec = state_to_vec(state, encoder, constraint, device=device, include_total_legs=_incl_total)
        logits = decoder(encoded, state_vec, mask, return_logits=True)
        probs = torch.softmax(logits, dim=-1)

        # Diagnostic target: only points that are "connections within the same duty" -- not the first flight of a pairing, not right after rest
        is_intra_context = not state.get("pairing_start", False) and not state.get("is_resting", False)
        pending = None
        if is_intra_context:
            valid_idx = [i for i, m in enumerate(mask_list[:-2]) if m == 1]
            if len(valid_idx) >= 2:
                gaps = [flights[i]["dep_time"] - state["current_time"] for i in valid_idx]
                scores = [logits[i].item() for i in valid_idx]
                pending = {"gaps": gaps, "scores": scores}

        if greedy:
            action = probs.argmax().item()
        else:
            action = Categorical(probs).sample().item()

        if action == len(flights):  # END_DUTY
            state, _, _ = env.step(state, action, flights, assigned, constraint)
            continue
        if action == len(flights) + 1:  # END_PAIRING
            unassigned = [f for f in flights if not assigned[f["id"]]]
            if not unassigned:
                break
            nxt = sorted([f for f in unassigned if f["origin"] == episode_base] or unassigned,
                         key=lambda f: f["dep_time"])[0]
            assigned[nxt["id"]] = True
            state = new_pairing_state(nxt, episode_base, assigned)
            continue

        f = flights[action]
        if pending is not None:
            pending["chosen_gap"] = f["dep_time"] - state["current_time"]
            records.append(pending)

        state, _, done = env.step(state, action, flights, assigned, constraint)
        if done:
            break

    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-rollouts", type=int, default=30)
    parser.add_argument("--subset-size", type=int, default=1200)
    args = parser.parse_args()

    device = torch.device(args.device)
    encoder, decoder = load_model(args.checkpoint, device)

    data_path = config.AIRLINE_DATA["delta"]
    airport_map = build_airport_map(data_path)
    base_ids = bases_to_ids(list(config.AIRLINE_BASES["delta"]), airport_map)
    base = base_ids[0]
    constraint = get_delta_constraints(base)

    flights = load_flights_rolling(
        data_path, window_days=5, offset_days=0, airport_map=airport_map,
        base_airport=base, n_max=args.subset_size,
    )
    max_time = 5 * 24.0
    origins, dests, dep_times, arr_times, fly_times = flights_to_tensors(flights, max_time, device=device)
    c_tensor = constraint_to_tensor(constraint, device=device)

    all_records = []
    with torch.no_grad():
        encoded = encoder(origins, dests, dep_times, arr_times, fly_times, c_tensor)
        for _ in range(args.n_rollouts):
            all_records.extend(
                rollout_with_diagnostics(flights, constraint, encoder, decoder, encoded, device, greedy=False)
            )

    n = len(all_records)
    print(f"checkpoint: {args.checkpoint}")
    print(f"diagnostic target intra-duty connection selection points: {n} ({args.n_rollouts} rollouts)")
    if n == 0:
        print("no records -- check subset/constraint settings")
        return

    # Step 1: fraction where the chosen gap == the minimum among candidates
    is_min_choice = [abs(r["chosen_gap"] - min(r["gaps"])) < 1e-6 for r in all_records]
    min_gaps = [min(r["gaps"]) for r in all_records]
    chosen_gaps = [r["chosen_gap"] for r in all_records]
    print()
    print("=== Step 1: candidate gap distribution (hypothesis A: data structure) ===")
    print(f"  fraction where chosen gap == candidate minimum: {sum(is_min_choice)/n*100:.1f}%")
    print(f"  mean of candidate minimums:   {sum(min_gaps)/n:.2f}h  (median {sorted(min_gaps)[n//2]:.2f}h)")
    print(f"  mean of actual chosen gaps: {sum(chosen_gaps)/n:.2f}h  (median {sorted(chosen_gaps)[n//2]:.2f}h)")

    # Step 2: score-gap Spearman correlation (computed per decision point, then averaged)
    corrs = []
    for r in all_records:
        if len(set(r["gaps"])) < 2:  # correlation is undefined if all candidate gaps are identical
            continue
        rho, _ = spearmanr(r["gaps"], r["scores"])
        if rho == rho:  # NaN check
            corrs.append(rho)
    print()
    print("=== Step 2: score-gap correlation (hypothesis B: model design) ===")
    print(f"  decision points where correlation is computable: {len(corrs)}/{n}")
    if corrs:
        avg_rho = sum(corrs) / len(corrs)
        print(f"  mean Spearman rho (gap vs score): {avg_rho:.3f}")
        print(f"  (negative means larger gap -> lower score, i.e. the model is aware of gap)")
        strong_neg = sum(1 for c in corrs if c <= -0.5)
        print(f"  fraction of decision points with rho <= -0.5 (strong negative correlation): {strong_neg/len(corrs)*100:.1f}%")


if __name__ == "__main__":
    main()
