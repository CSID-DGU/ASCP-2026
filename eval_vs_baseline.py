"""
RL vs Tahir I²CGp gap comparison.

Usage:
    cd /home/hyrn2/github/ASCP-2026
    source venv/bin/activate
    python eval_vs_baseline.py [--checkpoint checkpoints/model_latest.pt]
                               [--at 09]        # filter by aircraft type
                               [--tightness 1]  # filter by tightness level
                               [--results path/to/i2cgp_results.json]

Requires:
  - A trained model saved by experiments/train.py (checkpoints/model_latest.pt)
  - Tahir repo at ../Tahir with CPPSC_Instances data
  - ../Tahir/experiments/i2cgp_results.json (or specify via --results)

Gap formula:
    gap = (n_RL_pairings - n_baseline_pairings) / n_baseline_pairings * 100%
    Positive = RL worse than baseline, negative = RL better.
"""

import sys
import os
import json
import argparse
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "RL"))

from model import FlightEncoder, PointerDecoder
from RL.cppsc_loader import load_cppsc_flights, get_cppsc_constraints
from RL.constraints import FILM_CONSTRAINT_KEYS
from RL.state import init_state
from RL.environment import get_mask, step, final_reward
from torch.distributions import Categorical

TAHIR_RESULTS = os.path.join(
    os.path.dirname(__file__), "..", "Tahir", "experiments", "i2cgp_results.json"
)

ALL_TYPES = ["727", "09", "94", "95", "757", "319", "320"]


# ── helper to convert constraint dict → FiLM tensor ─────────────────────────

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
            state["current_time"] / 24.0,
            state["duty_time"]    / constraint["max_duty"],
            state.get("legs", 0)  / constraint.get("max_legs", 1),
        ], dtype=torch.float32),
    ])


# ── greedy rollout (no gradients) ────────────────────────────────────────────

def run_greedy(flights, constraint, encoder, decoder):
    encoded = encoder(*flights_to_tensors(flights), constraint_to_tensor(constraint))
    assigned = {f["id"]: False for f in flights}
    state = init_state(flights, constraint)
    n_pairings = 0

    while True:
        mask_list = get_mask(state, flights, assigned, constraint)
        mask = torch.tensor(mask_list, dtype=torch.float32)

        if sum(mask_list[:-1]) == 0:
            n_pairings += 1
            unassigned = [f for f in flights if not assigned[f["id"]]]
            if not unassigned:
                break
            earliest = sorted(unassigned, key=lambda x: x["dep_time"])[0]
            state = {
                "current_airport":  earliest["origin"],
                "current_time":     earliest["dep_time"],
                "duty_time":        0.0,
                "duty_start_time":  earliest["dep_time"],
                "legs":             0,
                "remaining":        sum(1 for v in assigned.values() if not v),
                "pairing_start":    True,
            }
            continue

        state_vec = state_to_vec(state, encoder, constraint)
        probs = decoder(encoded, state_vec, mask)
        action = probs.argmax().item()

        if action == len(flights):
            n_pairings += 1
            unassigned = [f for f in flights if not assigned[f["id"]]]
            if not unassigned:
                break
            earliest = sorted(unassigned, key=lambda x: x["dep_time"])[0]
            state = {
                "current_airport":  earliest["origin"],
                "current_time":     earliest["dep_time"],
                "duty_time":        0.0,
                "duty_start_time":  earliest["dep_time"],
                "legs":             0,
                "remaining":        sum(1 for v in assigned.values() if not v),
                "pairing_start":    True,
            }
            continue

        state, _, _ = step(state, action, flights, assigned, constraint)

    n_uncovered = sum(1 for v in assigned.values() if not v)
    coverage = (len(flights) - n_uncovered) / len(flights) * 100
    return n_pairings, n_uncovered, coverage


# ── load baseline results from Tahir ─────────────────────────────────────────

def load_baseline(results_path: str):
    """
    Returns dict keyed by (aircraft_type, instance_id) ->
      {'n_pairings': int, 'coverage': float, 'method': str}

    Prefers i2cgp over i2cg when both are present.
    """
    if not os.path.exists(results_path):
        return {}

    with open(results_path) as f:
        data = json.load(f)

    baseline = {}
    for entry in data:
        if entry.get("source") != "CPPSC":
            continue
        at   = entry["aircraft_type"]
        iid  = entry["instance_id"]
        rec  = entry.get("i2cgp") or entry.get("i2cg")
        if rec and "n_pairings" in rec:
            baseline[(at, iid)] = {
                "n_pairings": rec["n_pairings"],
                "coverage":   rec.get("coverage", float("nan")),
                "method":     "i2cgp" if "i2cgp" in entry else "i2cg",
            }
    return baseline


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/model_latest.pt")
    parser.add_argument("--at", default=None, help="Filter: aircraft type (e.g. '09')")
    parser.add_argument("--tightness", type=int, default=None,
                        help="Filter: tightness level 1-5 (default: all)")
    parser.add_argument("--results", default=TAHIR_RESULTS,
                        help="Path to Tahir i2cgp_results.json")
    args = parser.parse_args()

    # ── load model ──
    ckpt_path = os.path.join(os.path.dirname(__file__), args.checkpoint)
    if not os.path.exists(ckpt_path):
        print(f"[ERROR] No checkpoint found at {ckpt_path}")
        print("  Train first: python experiments/train.py")
        sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    n_airports     = ckpt["n_airports"]
    constraint_dim = ckpt["constraint_dim"]

    encoder = FlightEncoder(
        n_airports=n_airports,
        constraint_dim=constraint_dim,
        airport_emb_dim=32,
        d_model=128,
    )
    decoder = PointerDecoder(d_model=128, airport_emb_dim=32)
    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])
    encoder.eval()
    decoder.eval()
    print(f"모델 로드: {ckpt_path}  (n_airports={n_airports})")

    # ── load baseline ──
    baseline = load_baseline(args.results)
    if baseline:
        print(f"Tahir baseline: {len(baseline)} CPPSC entries from {args.results}")
    else:
        print(f"[WARN] No Tahir baseline found at {args.results}. Gap column will show N/A.")

    # ── determine which (type, tightness) to evaluate ──
    types      = [args.at] if args.at else ALL_TYPES
    tightnesses = [args.tightness] if args.tightness else list(range(1, 6))

    print()
    header = f"{'AT':>5}  {'T':>2}  {'Legs':>6}  {'RL pairs':>8}  {'BL pairs':>8}  {'Gap%':>7}  {'Coverage':>8}  {'Method'}"
    print(header)
    print("-" * len(header))

    rows = []
    for at in types:
        for t in tightnesses:
            try:
                flights, airport_map, base_ids = load_cppsc_flights(at, t)
            except (FileNotFoundError, Exception) as e:
                continue

            # Use first base airport as base_airport constraint
            base_airport = base_ids[0] if base_ids else 0
            constraint = get_cppsc_constraints(base_airport)

            # RL must use same n_airports as training; remap if needed
            # (CPPSC airport count may differ from training set)
            n_ap_cppsc = max(f["origin"] for f in flights) + 1
            n_ap_cppsc = max(n_ap_cppsc, max(f["dest"] for f in flights) + 1)

            if n_ap_cppsc > n_airports:
                print(f"  AT_{at} t={t}: SKIP — instance has {n_ap_cppsc} airports, "
                      f"model trained on {n_airports}")
                continue

            with torch.no_grad():
                n_rl, n_unc, cov = run_greedy(flights, constraint, encoder, decoder)

            bl = baseline.get((at, t))
            if bl:
                n_bl = bl["n_pairings"]
                gap  = (n_rl - n_bl) / max(n_bl, 1) * 100
                gap_str  = f"{gap:+.2f}%"
                meth_str = bl["method"]
            else:
                n_bl     = -1
                gap_str  = "N/A"
                meth_str = "-"

            bl_str = f"{n_bl:8d}" if n_bl >= 0 else "       -"
            print(f"  {at:>3}  {t:>2}  {len(flights):>6}  {n_rl:>8d}  {bl_str}  "
                  f"{gap_str:>7}  {cov:>7.1f}%  {meth_str}")
            rows.append((at, t, len(flights), n_rl, n_bl, gap_str))

    if rows:
        gaps = []
        for _, _, _, n_rl, n_bl, _ in rows:
            if n_bl > 0:
                gaps.append((n_rl - n_bl) / n_bl * 100)
        if gaps:
            print()
            print(f"평균 gap (baseline 있는 {len(gaps)}개): {sum(gaps)/len(gaps):+.2f}%")
            print(f"  양수 = RL이 baseline보다 pairing 더 많음 (나쁨)")
            print(f"  음수 = RL이 baseline보다 pairing 적음 (좋음)")


if __name__ == "__main__":
    main()
