"""
run_experiments.py  —  Run all Aspect-4 configurations.

Grid = 2 datasets × DEPTHS × MITIGATIONS × SEEDS
     = 2 × 5 × 3 × 1 = 30 runs (default)

DEPTHS sweep is what exposes oversmoothing; MITIGATIONS compares the plain deep stack
against skip connections and DropEdge. Skips runs whose checkpoint already exists.

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
    ("rel-trial", "study-outcome"),
]
DEPTHS      = [1, 2, 4, 8, 16]
MITIGATIONS = ["none", "residual", "dropedge"]
SEEDS       = [0]


def all_configs():
    for dataset, task in TASKS:
        for mit in MITIGATIONS:
            for L in DEPTHS:
                for s in SEEDS:
                    yield dict(dataset=dataset, task=task, mitigation=mit, num_layers=L, seed=s)


def ckpt_path(cfg):
    return (CHECKPOINTS / cfg["dataset"] / cfg["task"]
            / f"sage_{cfg['mitigation']}_L{cfg['num_layers']}_s{cfg['seed']}.pt")


def run_config(cfg):
    cmd = [PYTHON, "-u", "train.py",
           "--dataset", cfg["dataset"], "--task", cfg["task"],
           "--mitigation", cfg["mitigation"], "--num_layers", str(cfg["num_layers"]),
           "--seed", str(cfg["seed"]), "--skip_if_exists"]
    return subprocess.run(cmd, cwd=ROOT).returncode == 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    configs = list(all_configs())
    pending = [c for c in configs if not ckpt_path(c).exists()]
    print(f"Experiment grid: {len(configs)} total  |  {len(configs)-len(pending)} cached  |  {len(pending)} to run")
    if not pending:
        print("All checkpoints exist — nothing to do.")
        return
    if args.dry_run:
        for c in pending:
            print(f"  {c['dataset']}/{c['task']}  sage  {c['mitigation']}  L={c['num_layers']}  s={c['seed']}")
        return

    failed = []
    for i, cfg in enumerate(pending, 1):
        label = f"{cfg['dataset']}/{cfg['task']}  {cfg['mitigation']}  L={cfg['num_layers']}  s={cfg['seed']}"
        print(f"\n[{i}/{len(pending)}] {label}")
        if not run_config(cfg):
            failed.append(label)
    print(f"\n{'='*60}\nDone.  {len(pending)-len(failed)}/{len(pending)} succeeded.")
    for f in failed:
        print(f"  [FAILED] {f}")


if __name__ == "__main__":
    main()
