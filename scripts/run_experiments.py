"""Run the reproduction experiments.

Examples (from the project root):
  python scripts/run_experiments.py --exp all
  python scripts/run_experiments.py --exp main --seeds 0 --iters 50      (quick check)
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from satecn import experiments as E  # noqa: E402

ORDER = ["main", "exhaustive", "jumpstart", "beta", "system", "scalability", "continual", "ablation"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="all", help="comma list of: " + ", ".join(ORDER) + " or all")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--iters", type=int, default=None, help="override training iterations of the main run")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--quick", action="store_true", help="1 seed, short runs (smoke test, ~5 min)")
    a = ap.parse_args()

    seeds = [int(s) for s in a.seeds.split(",")]
    exps = ORDER if a.exp == "all" else a.exp.split(",")
    cfg = E.load_config(a.config)
    iters = a.iters or cfg["learner"]["iterations"]
    if a.quick:
        seeds, iters = [0], min(iters, 30)
    for name in exps:
        t0 = time.time()
        print(f"--- experiment: {name}", flush=True)
        if name == "main":
            E.run_main(a.config, seeds, iters, a.workers)
        elif name == "exhaustive":
            E.run_exhaustive(a.config, seeds[:1], a.workers)
        elif name == "jumpstart":
            E.run_jumpstart(a.config, seeds, a.workers, pre_rounds=5 if a.quick else 15)
        elif name == "beta":
            E.run_beta(a.config, seeds[:1], a.workers, iters=20 if a.quick else 100)
        elif name == "system":
            E.run_system(a.config, seeds)
        elif name == "scalability":
            E.run_scalability(a.config, a.workers, slots=120 if a.quick else 600, seeds=(0,) if a.quick else (0, 1, 2))
        elif name == "continual":
            E.run_continual(a.config, seeds, a.workers, n_enc=8 if a.quick else 32)
        elif name == "ablation":
            E.run_ablation(a.config, seeds[:1], a.workers, iters=20 if a.quick else 100)
        else:
            raise SystemExit(f"unknown experiment: {name}")
        print(f"--- {name} done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
