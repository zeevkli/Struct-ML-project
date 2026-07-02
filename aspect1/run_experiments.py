"""
run_experiments.py  —  Run all 36 experiment configurations.

36 runs = 2 tasks × 3 modes × 2 archs × 3 layers × 1 seed (seed=0)

Skips runs whose checkpoint already exists.
Called from the notebook, or can be run standalone:
  python run_experiments.py
  python run_experiments.py --dry_run   # show what would run
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT        = Path(__file__).parent
CHECKPOINTS = ROOT / "checkpoints"
PYTHON      = sys.executable

TASKS = [
    ("rel-stack", "user-engagement"),
    ("rel-avito",  "user-visits"),
]
ARCHS  = ["sage", "gat"]
MODES  = ["mpnn_u", "mpnn_d", "dir_gnn"]
LAYERS = [1, 2, 3]
SEED   = 0


def all_configs():
    for dataset, task in TASKS:
        for arch in ARCHS:
            for mode in MODES:
                for num_layers in LAYERS:
                    yield dict(
                        dataset    = dataset,
                        task       = task,
                        arch       = arch,
                        mode       = mode,
                        num_layers = num_layers,
                        seed       = SEED,
                    )


def ckpt_path(cfg) -> Path:
    return (CHECKPOINTS / cfg["dataset"] / cfg["task"]
            / f"{cfg['arch']}_{cfg['mode']}_L{cfg['num_layers']}_s{cfg['seed']}.pt")


def run_config(cfg, verbose=True):
    cmd = [
        PYTHON, "-u", "train.py",
        "--dataset",    cfg["dataset"],
        "--task",       cfg["task"],
        "--arch",       cfg["arch"],
        "--mode",       cfg["mode"],
        "--num_layers", str(cfg["num_layers"]),
        "--seed",       str(cfg["seed"]),
        "--skip_if_exists",
    ]
    label = (f"{cfg['dataset']}/{cfg['task']}  "
             f"{cfg['arch']}  {cfg['mode']}  L={cfg['num_layers']}")
    if verbose:
        print(f"\n{'─'*60}")
        print(f"Running: {label}")
        print(f"{'─'*60}")
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        print(f"[FAILED] {label}")
    return result.returncode == 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry_run", action="store_true",
                   help="Print configs without running.")
    args = p.parse_args()

    configs = list(all_configs())
    pending = [c for c in configs if not ckpt_path(c).exists()]
    done    = len(configs) - len(pending)

    print(f"Experiment grid: {len(configs)} total  |  {done} cached  |  {len(pending)} to run")

    if not pending:
        print("All checkpoints exist — nothing to do.")
        return

    if args.dry_run:
        print("\nPending runs:")
        for cfg in pending:
            label = (f"  {cfg['dataset']}/{cfg['task']}  "
                     f"{cfg['arch']}  {cfg['mode']}  L={cfg['num_layers']}")
            print(label)
        return

    failed = []
    for i, cfg in enumerate(pending, 1):
        label = (f"{cfg['dataset']}/{cfg['task']}  "
                 f"{cfg['arch']}  {cfg['mode']}  L={cfg['num_layers']}")
        print(f"\n[{i}/{len(pending)}] {label}")
        ok = run_config(cfg, verbose=False)
        if not ok:
            failed.append(label)

    print(f"\n{'='*60}")
    print(f"Done.  {len(pending) - len(failed)}/{len(pending)} succeeded.")
    if failed:
        print(f"Failed ({len(failed)}):")
        for f in failed:
            print(f"  {f}")


if __name__ == "__main__":
    main()
