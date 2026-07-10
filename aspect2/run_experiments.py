"""
run_experiments.py  —  Run all experiment configurations for Aspect 2.

Main grid:    3 tasks × 3 archs × 4 settings × 1 seed = 36 runs
  settings: homo, hetero, homo_noenc (no per-type encoders), hybrid (hetero layer 1 + homo rest)
Ablation:     3 tasks × 3 archs × homo only  × 1 seed =  9 runs  (hidden_dim=128)
Total: 45 runs  (homo & hetero cached from first pass — 18 new runs)

Skips runs whose checkpoint already exists.
Called from the notebook, or run standalone:
  python run_experiments.py
  python run_experiments.py --dry_run    # show what would run
  python run_experiments.py --ablation   # run ablation only
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT        = Path(__file__).parent
CHECKPOINTS = ROOT / "checkpoints"
PYTHON      = sys.executable

TASKS = [
    ("rel-stack",  "user-engagement"),
    ("rel-avito",  "user-visits"),
    ("rel-arxiv",  "author-category"),
]
ARCHS     = ["sage", "gat", "hgt"]
SETTINGS  = ["homo", "hetero", "homo_noenc", "hybrid"]
LAYERS    = [2, 3]
SEED      = 0
HIDDEN_DIM_MAIN     = 64
HIDDEN_DIM_ABLATION = 128


def all_configs():
    # Main grid: all settings at hidden_dim=64, layers in LAYERS
    for dataset, task in TASKS:
        for arch in ARCHS:
            for setting in SETTINGS:
                for num_layers in LAYERS:
                    yield dict(
                        dataset    = dataset,
                        task       = task,
                        arch       = arch,
                        setting    = setting,
                        num_layers = num_layers,
                        hidden_dim = HIDDEN_DIM_MAIN,
                        seed       = SEED,
                        ablation   = False,
                    )
    # Ablation: homo at hidden_dim=128, layers in LAYERS
    for dataset, task in TASKS:
        for arch in ARCHS:
            for num_layers in LAYERS:
                yield dict(
                    dataset    = dataset,
                    task       = task,
                    arch       = arch,
                    setting    = "homo",
                    num_layers = num_layers,
                    hidden_dim = HIDDEN_DIM_ABLATION,
                    seed       = SEED,
                    ablation   = True,
                )


def ckpt_path(cfg) -> Path:
    return (CHECKPOINTS / cfg["dataset"] / cfg["task"]
            / f"{cfg['arch']}_{cfg['setting']}_h{cfg['hidden_dim']}_L{cfg['num_layers']}_s{cfg['seed']}.pt")


def run_config(cfg, verbose=True, force=False):
    cmd = [
        PYTHON, "-u", "train.py",
        "--dataset",    cfg["dataset"],
        "--task",       cfg["task"],
        "--arch",       cfg["arch"],
        "--setting",    cfg["setting"],
        "--num_layers", str(cfg["num_layers"]),
        "--hidden_dim", str(cfg["hidden_dim"]),
        "--seed",       str(cfg["seed"]),
    ]
    if not force:
        cmd.append("--skip_if_exists")
    label = (f"{cfg['dataset']}/{cfg['task']}  "
             f"{cfg['arch']}  {cfg['setting']}  h={cfg['hidden_dim']}")
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
    p.add_argument("--dry_run",  action="store_true", help="Print configs without running.")
    p.add_argument("--ablation", action="store_true", help="Run ablation configs only.")
    p.add_argument("--main",     action="store_true", help="Run main grid only (no ablation).")
    p.add_argument("--force",    action="store_true",
                   help="Re-run all configs, overwriting existing checkpoints and results.")
    args = p.parse_args()

    configs = list(all_configs())
    if args.ablation:
        configs = [c for c in configs if c["ablation"]]
    elif args.main:
        configs = [c for c in configs if not c["ablation"]]

    pending = configs if args.force else [c for c in configs if not ckpt_path(c).exists()]
    done    = len(configs) - len(pending)

    print(f"Experiment grid: {len(configs)} total  |  {done} cached  |  {len(pending)} to run")

    if not pending:
        print("All checkpoints exist — nothing to do.")
        return

    if args.dry_run:
        print("\nPending runs:")
        for cfg in pending:
            tag = "[ablation]" if cfg["ablation"] else ""
            print(f"  {cfg['dataset']}/{cfg['task']}  "
                  f"{cfg['arch']}  {cfg['setting']}  h={cfg['hidden_dim']}  {tag}")
        return

    failed = []
    for i, cfg in enumerate(pending, 1):
        label = (f"{cfg['dataset']}/{cfg['task']}  "
                 f"{cfg['arch']}  {cfg['setting']}  h={cfg['hidden_dim']}")
        print(f"\n[{i}/{len(pending)}] {label}")
        ok = run_config(cfg, verbose=False, force=args.force)
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
