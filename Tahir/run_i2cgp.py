"""
run_i2cgp.py - Full I2CGp evaluation pipeline (Tahir et al. 2021)

Runs both I2CG (baseline) and I2CGp (DNN-guided) on CPP instances
and produces a results table comparable to Table 6 in the paper.

Usage
-----
# Run both I2CG and I2CGp on all instances
python run_i2cgp.py

# Only specific aircraft type
python run_i2cgp.py --at 09

# Only I2CG baseline (no DNN)
python run_i2cgp.py --method i2cg

# Only I2CGp (DNN must be trained first: python -m dnn.train)
python run_i2cgp.py --method i2cgp

# Tuning
python run_i2cgp.py --max_iter 100 --max_fail 3 --beam 15 --max_labels 300

Output
------
experiments/i2cgp_results.json   -- per-instance results
experiments/i2cgp_summary.csv    -- Table 6 comparison
"""

import argparse
import json
import os
import sys
import time
import numpy as np
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.insert(0, str(Path(__file__).parent))

from dnn.cpp_loader import (
    load_instance, load_cppsc_instance, discover_all_instances,
)
from dnn.dataset import (
    build_successor_sets, build_encoders, build_dataset,
    filter_successors_by_pattern,
)
from dnn.reference import generate_reference_pairings
from solver.constraints import pairing_cost
from solver.icg import run_i2cg, run_i2cgp


def load_dnn_for_type(aircraft_type: str, model_dir: Path, enc: dict):
    """Load trained DNN weights for a given aircraft type."""
    import tensorflow as tf
    from dnn.model import build_model

    weights_path = model_dir / f"weights_AT_{aircraft_type}.h5"
    cfg_path     = model_dir / f"model_config_AT_{aircraft_type}.json"
    norm_path    = model_dir / f"norm_AT_{aircraft_type}.json"

    if not weights_path.exists():
        return None, None, None

    with open(cfg_path) as f:
        cfg = json.load(f)

    model = build_model(
        n_airports=cfg["n_airports"],
        n_aircraft=cfg["n_aircraft"],
        **cfg.get("hparams", {}),
    )
    dummy = np.zeros((1, 1, 27), dtype=np.float32)
    model(tf.constant(dummy))
    model.load_weights(str(weights_path))

    norm_mean, norm_std = None, None
    if norm_path.exists():
        with open(norm_path) as f:
            nd = json.load(f)
        norm_mean = np.array(nd["mean"], dtype=np.float32)
        norm_std  = np.array(nd["std"],  dtype=np.float32)

    return model, norm_mean, norm_std


def build_p_psi(inst: dict, ref_pairings, model, enc, norm_mean, norm_std):
    """
    Build DNN probability matrix P and rank matrix Psi for an instance.
    Returns (P, Psi, class_max).
    """
    import tensorflow as tf
    from dnn.dataset import build_xi_matrix
    from solver.column_gen import compute_psi

    legs   = inst["legs"]
    leg_map = {leg["flight_id"]: leg for leg in legs}
    num_cols = list(range(4, 9)) + list(range(13, 18)) + list(range(22, 27))

    succ_raw = build_successor_sets(legs)
    succ_flt = filter_successors_by_pattern(legs, succ_raw, ref_pairings)

    # Build combined P over all bases (DNN is base-agnostic for p_ij)
    # (paper uses separate P^b per base; here we aggregate since DNN
    #  takes base as a feature, probabilities implicitly encode base context)
    P_combined: dict = {}
    for base in inst["bases"]:
        for fid, succ in succ_flt.items():
            if not succ:
                P_combined[fid] = {}
                continue
            X = build_xi_matrix(leg_map[fid], succ, leg_map, enc, base)
            if norm_mean is not None:
                X[:, num_cols] = (X[:, num_cols] - norm_mean) / norm_std
            X_in = tf.constant(X[np.newaxis].astype(np.float32))
            probs = model(X_in, training=False).numpy()[0]
            existing = P_combined.get(fid, {})
            for k, jid in enumerate(succ):
                existing[jid] = max(existing.get(jid, 0.0), float(probs[k]))
            P_combined[fid] = existing

    Psi, class_max = compute_psi(P_combined)
    return P_combined, Psi, class_max


def run_instance(
    inst:       dict,
    enc:        dict,
    model_dir:  Path,
    method:     str,   # 'i2cg', 'i2cgp', or 'both'
    max_iter:   int,
    max_fail:   int,
    max_labels: int,
    max_pricing: int,
    verbose:    bool,
) -> dict:
    at  = inst["aircraft_type"]
    iid = inst["instance_id"]
    src = inst.get("source", "CPP")
    n   = len(inst["legs"])

    print(f"\n{'='*60}", flush=True)
    print(f"  {src} AT_{at} instance_{iid}  ({n} legs, bases={inst['bases']})",
          flush=True)

    # Reference pairings (used for DNN input construction; CG-based if possible)
    ref = generate_reference_pairings(inst, method="cg", verbose=False)
    print(f"  Reference pairings: {len(ref)} (method=CG)", flush=True)

    result = {"aircraft_type": at, "instance_id": iid, "source": src, "n_legs": n}

    # ── I2CG (full SP, no DNN) ────────────────────────────────────────────────
    if method in ("i2cg", "both"):
        print(f"\n  -- I2CG (full SP) --", flush=True)
        r = run_i2cg(
            inst, initial_columns=[list(p) for p in ref],
            max_fail=max_fail, max_iter=max_iter,
            time_limit_mip=300, max_labels=max_labels,
            max_pricing_cols=max_pricing, verbose=verbose,
        )
        lp = r["lp_obj"]
        gap = r.get("gap_pct",
                    abs((r["mip_obj"] - lp) / max(abs(lp), 1.0) * 100)
                    if lp < float("inf") else float("inf"))
        result["i2cg"] = {
            "mip_obj":      r["mip_obj"],
            "lp_obj":       lp,
            "gap_pct":      gap,
            "coverage":     r["coverage"],
            "n_pairings":   len(r["selected_pairings"]),
            "n_uncovered":  r.get("n_uncovered", 0),
            "n_iters":      r["n_iters"],
            "n_columns":    r["n_columns"],
            "time":         r["total_time"],
            "status":       r["status"],
        }
        gap_str = f"{gap:.4f}%" if gap < float("inf") else "N/A"
        print(f"  I2CG: obj={r['mip_obj']:.2f} lp={lp:.2f} gap={gap_str} "
              f"coverage={r['coverage']:.3f} uncovered={r.get('n_uncovered',0)} "
              f"iters={r['n_iters']} time={r['total_time']:.1f}s", flush=True)

    # ── I2CGp (DNN-guided reduced SP) ────────────────────────────────────────
    if method in ("i2cgp", "both"):
        model, norm_mean, norm_std = load_dnn_for_type(at, model_dir, enc)
        if model is None:
            print(f"  I2CGp: skipped (no trained weights for AT_{at})", flush=True)
            result["i2cgp"] = {"error": f"no weights for AT_{at}"}
        else:
            print(f"\n  -- I2CGp (DNN-guided) --", flush=True)
            P, Psi, class_max = build_p_psi(
                inst, ref, model, enc, norm_mean, norm_std
            )
            r = run_i2cgp(
                inst, P, Psi, class_max,
                initial_columns=[list(p) for p in ref],
                max_fail=max_fail, max_iter=max_iter,
                time_limit_mip=300, max_labels=max_labels,
                max_pricing_cols=max_pricing, verbose=verbose,
            )
            lp = r["lp_obj"]
            gap = r.get("gap_pct",
                        abs((r["mip_obj"] - lp) / max(abs(lp), 1.0) * 100)
                        if lp < float("inf") else float("inf"))
            result["i2cgp"] = {
                "mip_obj":      r["mip_obj"],
                "lp_obj":       lp,
                "gap_pct":      gap,
                "coverage":     r["coverage"],
                "n_pairings":   len(r["selected_pairings"]),
                "n_uncovered":  r.get("n_uncovered", 0),
                "n_iters":      r["n_iters"],
                "n_columns":    r["n_columns"],
                "time":         r["total_time"],
                "status":       r["status"],
            }
            gap_str = f"{gap:.4f}%" if gap < float("inf") else "N/A"
            print(f"  I2CGp: obj={r['mip_obj']:.2f} lp={lp:.2f} gap={gap_str} "
                  f"coverage={r['coverage']:.3f} uncovered={r.get('n_uncovered',0)} "
                  f"iters={r['n_iters']} time={r['total_time']:.1f}s", flush=True)

    return result


def print_summary(results: list):
    """Print Table-6-style summary."""
    print("\n" + "="*80)
    print(f"{'Instance':<22} {'Method':<8} {'Obj':>10} {'Cov%':>7} "
          f"{'Pairs':>6} {'Iters':>6} {'Time(s)':>8}")
    print("-"*80)
    for r in results:
        tag = f"{r['source']} AT_{r['aircraft_type']} {r['instance_id']}"
        for m in ("i2cg", "i2cgp"):
            d = r.get(m)
            if d is None:
                continue
            if "error" in d:
                print(f"  {tag:<20} {m:<8}  {'ERR':>10}")
                continue
            print(f"  {tag:<20} {m:<8} {d['mip_obj']:>10.2f} "
                  f"{d['coverage']*100:>6.1f}% {d['n_pairings']:>6} "
                  f"{d['n_iters']:>6} {d['time']:>8.1f}")
    print("="*80)


def save_csv(results: list, out_path: Path):
    import csv
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "aircraft_type", "instance_id", "n_legs",
                    "method", "mip_obj", "lp_obj", "gap_pct",
                    "coverage", "n_pairings", "n_uncovered", "n_iters", "n_columns",
                    "time_s", "status"])
        for r in results:
            for m in ("i2cg", "i2cgp"):
                d = r.get(m)
                if d is None or "error" in d:
                    continue
                w.writerow([
                    r["source"], r["aircraft_type"], r["instance_id"], r["n_legs"],
                    m, d["mip_obj"], d.get("lp_obj", ""), d.get("gap_pct", ""),
                    d["coverage"], d["n_pairings"], d.get("n_uncovered", 0),
                    d["n_iters"], d["n_columns"], d["time"], d["status"],
                ])


def main():
    parser = argparse.ArgumentParser(description="I2CGp evaluation (Tahir 2021)")
    parser.add_argument("--model_dir",   default="experiments/loto",
                        help="Directory with trained DNN weights")
    parser.add_argument("--at",          default=None,
                        help="Filter by aircraft type (e.g. 09, 320)")
    parser.add_argument("--method",      default="both",
                        choices=["i2cg", "i2cgp", "both"],
                        help="Which method to run")
    parser.add_argument("--max_iter",    type=int, default=100,
                        help="Max outer CG iterations")
    parser.add_argument("--max_fail",    type=int, default=3,
                        help="maxFail parameter (Algorithm 1)")
    parser.add_argument("--max_labels",  type=int, default=300,
                        help="Max SPPRC labels per node")
    parser.add_argument("--max_pricing", type=int, default=500,
                        help="Max new columns per pricing call")
    parser.add_argument("--verbose",     action="store_true",
                        help="Print iteration details")
    parser.add_argument("--source",      default="both",
                        choices=["CPP", "CPPSC", "both"],
                        help="Dataset to evaluate")
    parser.add_argument("--exclude",     default=[], nargs="+",
                        metavar="INST",
                        help="Instance IDs to skip, e.g. --exclude CPP_320_1 CPPSC_757_3. "
                             "Format: {SOURCE}_{AT}_{IID}")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)

    # Discover instances
    meta = discover_all_instances()
    if args.at:
        meta = [(at, iid, src) for at, iid, src in meta if at == args.at]
    if args.source != "both":
        meta = [(at, iid, src) for at, iid, src in meta if src == args.source]

    # Apply --exclude filter  (format: SOURCE_AT_IID, e.g. CPP_320_1)
    if args.exclude:
        excluded = set(args.exclude)
        before = len(meta)
        meta = [
            (at, iid, src) for at, iid, src in meta
            if f"{src}_{at}_{iid}" not in excluded
        ]
        print(f"Excluded {before - len(meta)} instance(s): {args.exclude}")

    if not meta:
        print("No instances found. Check --at / --source / --exclude arguments.")
        return

    # Build global encoders
    print("Loading instances and building encoders...", flush=True)
    all_insts = []
    for at, iid, src in meta:
        inst = (load_cppsc_instance(at, iid) if src == "CPPSC"
                else load_instance(at, iid))
        all_insts.append(inst)
    enc = build_encoders(all_insts)

    # Run
    all_results = []
    for inst in all_insts:
        r = run_instance(
            inst=inst, enc=enc, model_dir=model_dir,
            method=args.method,
            max_iter=args.max_iter, max_fail=args.max_fail,
            max_labels=args.max_labels, max_pricing=args.max_pricing,
            verbose=args.verbose,
        )
        all_results.append(r)

    print_summary(all_results)

    # Save results
    out_dir = Path("experiments")
    out_dir.mkdir(exist_ok=True)

    json_path = out_dir / "i2cgp_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {json_path}")

    csv_path = out_dir / "i2cgp_summary.csv"
    save_csv(all_results, csv_path)
    print(f"CSV summary saved to {csv_path}")


if __name__ == "__main__":
    main()
