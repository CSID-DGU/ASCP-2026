"""
Phase 2: Training script for Tahir DNN
Leave-one-type-out cross-validation:
  Train on all instances except target aircraft type,
  evaluate on target type's instances.

Usage:
  python -m dnn.train [--epochs 30] [--batch 32] [--lr 0.001]
  python -m dnn.train --ref cg --at 757 320 95   # retrain low-accuracy types only
  python -m dnn.train --ref greedy                # fast mode (legacy)
  python -m dnn.train --out_dir experiments/loto_r4   # save to different dir
"""

import argparse
import json
import os
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import List, Dict, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from dnn.cpp_loader import (
    load_instance, discover_instances,
    load_cppsc_instance, discover_all_instances,
)
from dnn.dataset import (
    build_successor_sets, build_encoders, build_dataset,
    filter_successors_by_pattern,
)
from dnn.reference import generate_reference_pairings
from dnn.model import build_model


# ── Data helpers ─────────────────────────────────────────────────────────────

def pad_batch(X_list: List[np.ndarray], y_list: List[int]) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pad variable-length X_i matrices to uniform (max_K, 27).
    Returns:
      X_pad:  (N, max_K, 27)
      y:      (N,)
      mask:   (N, max_K)  float32, 1=valid
    """
    max_K = max(x.shape[0] for x in X_list)
    N     = len(X_list)
    X_pad = np.zeros((N, max_K, 27), dtype=np.float32)
    mask  = np.zeros((N, max_K),     dtype=np.float32)
    for i, x in enumerate(X_list):
        k = x.shape[0]
        X_pad[i, :k] = x
        mask[i,  :k] = 1.0
    return X_pad, np.array(y_list, dtype=np.int32), mask


def normalise(X_pad: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score normalise numerical columns in-place (indices 4-8, 13-17, 22-26)."""
    num_cols = list(range(4, 9)) + list(range(13, 18)) + list(range(22, 27))
    mean = X_pad[:, :, num_cols].mean(axis=(0, 1))
    std  = X_pad[:, :, num_cols].std(axis=(0, 1)) + 1e-9
    X_pad[:, :, num_cols] = (X_pad[:, :, num_cols] - mean) / std
    return X_pad, mean, std


def load_all_data_from_instances(
    instances: List[Dict],
    enc: Dict,
    ref_method: str = "cg",
) -> Dict[str, Tuple[List, List]]:
    """Load (X_list, y_list) per aircraft type from pre-loaded instances.
    Avoids double-loading. Prints progress per instance.

    Args:
        ref_method: 'cg'     -> I2CG near-optimal reference (recommended, as in paper)
                    'greedy' -> fast greedy reference (lower quality)
    """
    data_by_type: Dict[str, Tuple[List, List]] = {}
    n = len(instances)
    for idx, inst in enumerate(instances):
        at = inst["aircraft_type"]
        src = inst.get("source", "CPP")
        iid = inst.get("instance_id", "?")
        print(f"  [{idx+1}/{n}] {src} AT_{at} inst_{iid} ({len(inst['legs'])} legs) "
              f"[ref={ref_method}]...", flush=True)

        ref   = generate_reference_pairings(inst, method=ref_method)
        succ  = build_successor_sets(inst["legs"])
        fsucc = filter_successors_by_pattern(inst["legs"], succ, ref)

        X_all, y_all = [], []
        for base in inst["bases"]:
            Xb, yb = build_dataset(inst, fsucc, ref, enc, base)
            X_all.extend(Xb)
            y_all.extend(yb)

        if at not in data_by_type:
            data_by_type[at] = ([], [])
        data_by_type[at][0].extend(X_all)
        data_by_type[at][1].extend(y_all)
        print(f"         → {len(X_all)} samples (total {len(data_by_type[at][0])})", flush=True)

    return data_by_type


def load_all_data(
    instances_meta: List[Tuple],
    enc: Dict,
    ref_method: str = "cg",
) -> Dict[str, Tuple[List, List]]:
    """Load (X_list, y_list) per aircraft type.
    instances_meta: List of (at, inst_id, source) or (at, inst_id)
    """
    data_by_type: Dict[str, Tuple[List, List]] = {}

    for entry in instances_meta:
        at, inst_id = entry[0], entry[1]
        source = entry[2] if len(entry) > 2 else "CPP"
        if source == "CPPSC":
            inst = load_cppsc_instance(at, inst_id)
        else:
            inst  = load_instance(at, inst_id)
        ref   = generate_reference_pairings(inst, method=ref_method)
        succ  = build_successor_sets(inst["legs"])
        fsucc = filter_successors_by_pattern(inst["legs"], succ, ref)

        X_all, y_all = [], []
        for base in inst["bases"]:
            Xb, yb = build_dataset(inst, fsucc, ref, enc, base)
            X_all.extend(Xb)
            y_all.extend(yb)

        key = at
        if key not in data_by_type:
            data_by_type[key] = ([], [])
        data_by_type[key][0].extend(X_all)
        data_by_type[key][1].extend(y_all)

    return data_by_type


# ── Training ─────────────────────────────────────────────────────────────────

def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    mask_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    mask_val: np.ndarray,
    enc: Dict,
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 0.001,
    **hparams,
) -> Tuple[tf.keras.Model, Dict]:
    """Train one model, return model + history."""
    n_airports = len(enc["airport"])
    n_aircraft = len(enc["aircraft"])

    model = build_model(
        n_airports=n_airports,
        n_aircraft=n_aircraft,
        **hparams,
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    @tf.function
    def train_step(xb, yb, mb):
        with tf.GradientTape() as tape:
            preds = model(xb, training=True)
            loss  = masked_ce(yb, preds, mb)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    @tf.function
    def val_step(xb, yb, mb):
        preds = model(xb, training=False)
        loss  = masked_ce(yb, preds, mb)
        # top-1 accuracy
        pred_class = tf.argmax(preds * mb, axis=-1, output_type=tf.int32)
        acc = tf.reduce_mean(tf.cast(pred_class == yb, tf.float32))
        return loss, acc

    def masked_ce(y_true, y_pred, mask):
        K    = tf.shape(y_pred)[1]
        oh   = tf.one_hot(tf.cast(y_true, tf.int32), K)
        mp   = y_pred * mask
        mp   = mp / (tf.reduce_sum(mp, axis=-1, keepdims=True) + 1e-9)
        loss = -tf.reduce_sum(oh * tf.math.log(mp + 1e-9), axis=-1)
        return tf.reduce_mean(loss)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    n = len(X_train)
    best_val_acc  = 0.0
    best_weights  = None
    patience      = 10
    patience_cnt  = 0

    for epoch in range(epochs):
        # Shuffle
        idx  = np.random.permutation(n)
        Xs   = X_train[idx]
        ys   = y_train[idx]
        ms   = mask_train[idx]

        epoch_loss = []
        for start in range(0, n, batch_size):
            xb = tf.constant(Xs[start:start+batch_size])
            yb = tf.constant(ys[start:start+batch_size])
            mb = tf.constant(ms[start:start+batch_size])
            l  = train_step(xb, yb, mb)
            epoch_loss.append(float(l))

        val_loss, val_acc = val_step(
            tf.constant(X_val), tf.constant(y_val), tf.constant(mask_val)
        )
        history["train_loss"].append(float(np.mean(epoch_loss)))
        history["val_loss"].append(float(val_loss))
        history["val_acc"].append(float(val_acc))

        cur_acc = float(val_acc)
        if cur_acc > best_val_acc:
            best_val_acc = cur_acc
            best_weights = [v.numpy() for v in model.trainable_variables]
            patience_cnt = 0
        else:
            patience_cnt += 1

        if (epoch + 1) % 5 == 0:
            print(f"  epoch {epoch+1:3d} | train_loss={np.mean(epoch_loss):.4f} "
                  f"val_loss={float(val_loss):.4f} val_acc={float(val_acc):.3f}  "
                  f"best={best_val_acc:.3f}", flush=True)

        if patience_cnt >= patience:
            print(f"  Early stopping at epoch {epoch+1} (best={best_val_acc:.3f})", flush=True)
            break

    # Restore best weights
    if best_weights is not None:
        for v, w in zip(model.trainable_variables, best_weights):
            v.assign(w)

    return model, history


# ── Leave-one-type-out experiment ────────────────────────────────────────────

def run_loto(
    epochs:       int  = 30,
    batch_size:   int  = 32,
    lr:           float = 0.001,
    ref_method:   str  = "cg",
    target_types: List[str] = None,
    out_dir:      str  = None,
    **hparams,
):
    """
    Leave-one-type-out training: for each aircraft type,
    train on remaining types, evaluate on held-out type.

    Args:
        ref_method:   'cg' (paper default) or 'greedy' (fast fallback)
        target_types: if set, only train/eval these types (subset of LOTO folds)
                      e.g. ['757', '320', '95'] to retrain low-accuracy types
        out_dir:      directory to save weights; defaults to experiments/loto

    Deduplication: CPPSC has 5 tightness levels sharing the SAME legs file.
    For DNN training, availability constraints are irrelevant (used only in SPPRC).
    We keep only tightness=1 per CPPSC type to avoid 5x redundant processing.
    CPP instances are kept all (different time windows = different leg sets).
    """
    meta_all = discover_all_instances()   # (at, inst_id, source)

    # Deduplicate: for CPPSC, keep only tightness=1 per type
    seen_cppsc = set()
    meta = []
    for at, inst_id, source in meta_all:
        if source == "CPPSC":
            if at not in seen_cppsc:
                seen_cppsc.add(at)
                meta.append((at, 1, "CPPSC"))   # always use tightness=1
        else:
            meta.append((at, inst_id, "CPP"))

    print(f"Instance list: {len(meta)} (after dedup; {len(meta_all)} total with tightness)")
    for at, iid, src in meta:
        print(f"  {src} AT_{at} inst_{iid}")

    # Load all instances once, build encoders, then build datasets (no double-loading)
    print("\nLoading instances and building encoders...")
    all_instances = []
    for at, inst_id, source in meta:
        if source == "CPPSC":
            all_instances.append(load_cppsc_instance(at, inst_id))
        else:
            all_instances.append(load_instance(at, inst_id))
    enc = build_encoders(all_instances)
    print(f"Global encoders: {len(enc['airport'])} airports, {len(enc['aircraft'])} types")

    print(f"\nBuilding training datasets (ref={ref_method}) ...")
    data_by_type = load_all_data_from_instances(all_instances, enc, ref_method=ref_method)
    aircraft_types = sorted(data_by_type.keys())
    print(f"Aircraft types: {aircraft_types}")
    for t, (X, y) in data_by_type.items():
        print(f"  AT_{t}: {len(X)} samples")

    # Filter to target types if specified
    eval_types = aircraft_types
    if target_types:
        eval_types = [t for t in aircraft_types if t in target_types]
        print(f"  → Evaluating only: {eval_types}")

    results = {}
    if out_dir:
        save_dir = Path(out_dir)
    else:
        save_dir = Path(__file__).parent.parent / "experiments" / "loto"
    save_dir.mkdir(parents=True, exist_ok=True)

    for test_type in eval_types:
        print(f"\n=== Leave-one-out: test type = AT_{test_type} ===", flush=True)
        # Combine all OTHER types for training
        X_train_all, y_train_all = [], []
        for t, (X, y) in data_by_type.items():
            if t != test_type:
                X_train_all.extend(X)
                y_train_all.extend(y)

        if not X_train_all:
            print(f"  Skipped (no training data)")
            continue

        X_test, y_test = data_by_type[test_type]
        if not X_test:
            print(f"  Skipped (no test data)")
            continue

        # Pad
        X_tr_pad, y_tr, m_tr = pad_batch(X_train_all, y_train_all)
        X_te_pad, y_te, m_te = pad_batch(X_test, y_test)

        # Normalise (fit on train, apply to both)
        num_cols = list(range(4, 9)) + list(range(13, 18)) + list(range(22, 27))
        mean = X_tr_pad[:, :, num_cols].mean(axis=(0, 1))
        std  = X_tr_pad[:, :, num_cols].std(axis=(0, 1)) + 1e-9
        X_tr_pad[:, :, num_cols] = (X_tr_pad[:, :, num_cols] - mean) / std
        X_te_pad[:, :, num_cols] = (X_te_pad[:, :, num_cols] - mean) / std

        print(f"  train={len(X_tr_pad)}, test={len(X_te_pad)}, max_K={X_tr_pad.shape[1]}", flush=True)

        model, history = train_model(
            X_tr_pad, y_tr, m_tr,
            X_te_pad, y_te, m_te,
            enc, epochs=epochs, batch_size=batch_size, lr=lr,
            **hparams,
        )

        best_acc = max(history["val_acc"])
        print(f"  Best val accuracy: {best_acc:.3f}", flush=True)
        results[test_type] = {"best_acc": best_acc, "history": history}

        # Save model weights (full model save fails due to Lambda layers with Ellipsis)
        model.save_weights(str(save_dir / f"weights_AT_{test_type}.h5"))
        # Save model config for reconstruction
        model_cfg_path = save_dir / f"model_config_AT_{test_type}.json"
        with open(model_cfg_path, "w") as f:
            json.dump({
                "n_airports": len(enc["airport"]),
                "n_aircraft": len(enc["aircraft"]),
                "hparams": hparams,
            }, f)
        with open(save_dir / f"norm_AT_{test_type}.json", "w") as f:
            json.dump({"mean": mean.tolist(), "std": std.tolist()}, f)

    # Summary
    print("\n=== LOTO Results ===")
    for t, r in results.items():
        print(f"  AT_{t}: best_acc = {r['best_acc']:.3f}")
    avg = np.mean([r["best_acc"] for r in results.values()])
    print(f"  Average accuracy: {avg:.3f}")

    with open(save_dir / "results.json", "w") as f:
        json.dump({k: {"best_acc": v["best_acc"]} for k, v in results.items()}, f, indent=2)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tahir DNN LOTO training")
    # Training hyperparameters
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--batch",      type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=0.001)
    parser.add_argument("--emb_dim",    type=int,   default=10)
    parser.add_argument("--conv",       type=int,   default=1)
    parser.add_argument("--filters",    type=int,   default=128)
    parser.add_argument("--fsize",      type=int,   default=3)
    parser.add_argument("--dense",      type=int,   default=2)
    parser.add_argument("--neurons",    type=int,   default=256)
    parser.add_argument("--dropout",    type=float, default=0.3)
    # Experiment control
    parser.add_argument("--ref",        default="cg", choices=["cg", "greedy"],
                        help="Reference pairing method: 'cg' (paper default, near-optimal) "
                             "or 'greedy' (fast fallback)")
    parser.add_argument("--at",         nargs="+",  default=None,
                        metavar="TYPE",
                        help="Train/eval only these aircraft types, e.g. --at 757 320 95")
    parser.add_argument("--out_dir",    default=None,
                        help="Save weights to this directory (default: experiments/loto)")
    args = parser.parse_args()

    run_loto(
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        ref_method=args.ref,
        target_types=args.at,
        out_dir=args.out_dir,
        embedding_dim=args.emb_dim,
        num_conv_layers=args.conv,
        num_filters=args.filters,
        filter_size=args.fsize,
        num_dense_layers=args.dense,
        neurons_per_layer=args.neurons,
        dropout_rate=args.dropout,
    )
