"""
run_experiments.py  —  Run all Aspect-3 configurations.

Grid = 2 datasets × 3 feature modes × SEEDS   (default 12 runs at 2 seeds)

Skips runs whose checkpoint already exists. Called from the notebook, or standalone:
  python run_experiments.py
  python run_experiments.py --dry_run
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT        = Path(__file__).parent
CHECKPOINTS = ROOT / "checkpoints"
PYTHON      = sys.executable

TASKS = [
    ("rel-f1",    "driver-dnf"),
    ("rel-event", "user-repeat"),
]
FEAT_MODES = ["id", "column", "llm"]
NUM_LAYERS = 2
SEEDS      = [0, 1]


def all_configs():
    for dataset, task in TASKS:
        for feat_mode in FEAT_MODES:
            for seed in SEEDS:
                yield dict(dataset=dataset, task=task, feat_mode=feat_mode,
                           num_layers=NUM_LAYERS, seed=seed)


def ckpt_path(cfg) -> Path:
    return (CHECKPOINTS / cfg["dataset"] / cfg["task"]
            / f"hgt_{cfg['feat_mode']}_L{cfg['num_layers']}_s{cfg['seed']}.pt")


def run_config(cfg) -> bool:
    cmd = [
        PYTHON, "-u", "train.py",
        "--dataset",    cfg["dataset"],
        "--task",       cfg["task"],
        "--feat_mode",  cfg["feat_mode"],
        "--num_layers", str(cfg["num_layers"]),
        "--seed",       str(cfg["seed"]),
        "--skip_if_exists",
    ]
    return subprocess.run(cmd, cwd=ROOT).returncode == 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry_run", action="store_true")
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
        for c in pending:
            print(f"  {c['dataset']}/{c['task']}  hgt  {c['feat_mode']}  "
                  f"L={c['num_layers']}  s={c['seed']}")
        return

    failed = []
    for i, cfg in enumerate(pending, 1):
        label = (f"{cfg['dataset']}/{cfg['task']}  hgt  {cfg['feat_mode']}  "
                 f"L={cfg['num_layers']}  s={cfg['seed']}")
        print(f"\n[{i}/{len(pending)}] {label}")
        if not run_config(cfg):
            failed.append(label)

    print(f"\n{'='*60}\nDone.  {len(pending) - len(failed)}/{len(pending)} succeeded.")
    for f in failed:
        print(f"  [FAILED] {f}")


if __name__ == "__main__":
    main()
