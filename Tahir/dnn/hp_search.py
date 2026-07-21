"""
Bayesian Hyperparameter Search for I²CGp DNN (Tahir 2021, Section 4.2).

Uses Optuna (TPE sampler) to search over:
  - learning rate
  - embedding_dim, num_conv_layers, num_filters, filter_size
  - num_dense_layers, neurons_per_layer, dropout_rate

Objective: mean validation loss across all LOTO folds for the target type(s).

Usage:
    # Search on AT_09 (fast, ~30 trials)
    python -m dnn.hp_search --at 09 --n_trials 30 --epochs 30

    # Full search on all types
    python -m dnn.hp_search --n_trials 100 --epochs 50

    # Resume existing study
    python -m dnn.hp_search --at 09 --n_trials 50 --storage hp_search.db
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.insert(0, str(Path(__file__).parent.parent))


def _objective_for_type(
    aircraft_type: str,
    epochs: int,
    batch_size: int,
    out_dir: Path,
    ref_method: str,
):
    """Return an Optuna objective function for one aircraft type."""
    import optuna
    import numpy as np

    def objective(trial: "optuna.Trial") -> float:
        hparams = {
            "embedding_dim":      trial.suggest_int("embedding_dim", 5, 20),
            "num_conv_layers":    trial.suggest_int("num_conv_layers", 1, 3),
            "num_filters":        trial.suggest_int("num_filters", 32, 256, step=32),
            "filter_size":        trial.suggest_int("filter_size", 1, 5),
            "num_dense_layers":   trial.suggest_int("num_dense_layers", 1, 3),
            "neurons_per_layer":  trial.suggest_int("neurons_per_layer", 64, 512, step=64),
            "dropout_rate":       trial.suggest_float("dropout_rate", 0.0, 0.5),
        }
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

        from dnn.train import run_loto

        results = run_loto(
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            ref_method=ref_method,
            target_types=[aircraft_type],
            out_dir=str(out_dir / f"hp_trial_{trial.number}"),
            **hparams,
        )
        if not results:
            return float("inf")
        # Minimize 1 - mean_val_accuracy (equivalent to maximising accuracy)
        accs = [v["best_acc"] for v in results.values()]
        return float(1.0 - np.mean(accs))

    return objective


def run_hp_search(
    aircraft_types: list,
    n_trials: int = 50,
    epochs: int = 50,
    batch_size: int = 32,
    ref_method: str = "cg",
    out_dir: Path = Path("experiments/hp_search"),
    storage: str = None,
    study_prefix: str = "i2cgp_hp",
    n_jobs: int = 1,
):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    out_dir.mkdir(parents=True, exist_ok=True)
    best_per_type = {}

    for at in aircraft_types:
        study_name = f"{study_prefix}_AT_{at}"
        if storage:
            storage_url = f"sqlite:///{storage}"
        else:
            storage_url = None

        study = optuna.create_study(
            study_name=study_name,
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
            storage=storage_url,
            load_if_exists=True,
        )

        obj = _objective_for_type(
            aircraft_type=at,
            epochs=epochs,
            batch_size=batch_size,
            out_dir=out_dir / at,
            ref_method=ref_method,
        )

        print(f"\n[HP Search] AT_{at}: {n_trials} trials, "
              f"epochs={epochs}, ref={ref_method}")

        study.optimize(obj, n_trials=n_trials, n_jobs=n_jobs,
                       show_progress_bar=False)

        best = study.best_trial
        print(f"[HP Search] AT_{at} best trial #{best.number}: "
              f"val_loss={best.value:.6f}")
        print(f"  params: {best.params}")

        best_per_type[at] = {
            "val_loss": best.value,
            "params":   best.params,
        }

        # Save best params for this type
        params_path = out_dir / f"best_params_AT_{at}.json"
        with open(params_path, "w") as f:
            json.dump(best_per_type[at], f, indent=2)
        print(f"  saved -> {params_path}")

    # Save summary
    summary_path = out_dir / "hp_search_summary.json"
    with open(summary_path, "w") as f:
        json.dump(best_per_type, f, indent=2)
    print(f"\n[HP Search] Summary saved -> {summary_path}")

    return best_per_type


def main():
    parser = argparse.ArgumentParser(
        description="Bayesian HP search for I2CGp DNN"
    )
    parser.add_argument("--at", nargs="+", default=None,
                        help="Aircraft type(s) to search (default: all 7)")
    parser.add_argument("--n_trials",  type=int, default=50)
    parser.add_argument("--epochs",    type=int, default=50)
    parser.add_argument("--batch_size",type=int, default=32)
    parser.add_argument("--ref_method",default="cg",
                        choices=["cg", "greedy"])
    parser.add_argument("--out_dir",   default="experiments/hp_search")
    parser.add_argument("--storage",   default=None,
                        help="SQLite DB path to persist study (e.g. hp_search.db)")
    parser.add_argument("--study_prefix", default="i2cgp_hp")
    parser.add_argument("--n_jobs",    type=int, default=1)
    args = parser.parse_args()

    all_types = ["727", "09", "94", "95", "757", "319", "320"]
    target = args.at if args.at else all_types

    run_hp_search(
        aircraft_types=target,
        n_trials=args.n_trials,
        epochs=args.epochs,
        batch_size=args.batch_size,
        ref_method=args.ref_method,
        out_dir=Path(args.out_dir),
        storage=args.storage,
        study_prefix=args.study_prefix,
        n_jobs=args.n_jobs,
    )


if __name__ == "__main__":
    main()
