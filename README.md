# CDLL-SATECN: reproduction of "Lifelong Learning-Based SDN Design for Dynamic Configuration and Resource Allocation in Satellite-Terrestrial Networks"

This repository re-implements the system model and the learning algorithm of

> Y. Wu, N. Wang, B. Lorenzo, S. Samarakoon, B. Liu, "Lifelong Learning-Based SDN Design for Dynamic
> Configuration and Resource Allocation in Satellite-Terrestrial Networks," *IEEE Transactions on Wireless
> Communications*, 2026, DOI 10.1109/TWC.2026.3716226.

and re-runs the paper's evaluation (Figs. 3-9, Table V) plus ablations.

Advanced Computer Networks course project (Project ID 22), Department of CSE, NIT Warangal,
under the guidance of Dr. Venkatarami Reddy Chintapalli.

| Team member | Roll number |
|---|---|
| Vedant Amit Agrawal | 23CSB0A08 |
| Abu Fahad Ansari | 23CSB0A45 |
| Shwetang Wasnik | 23CSB0A55 | The paper provides no code; everything
here was written from the paper's equations. Parameters the paper does not state are marked `[assumed]` in
`configs/default.yaml` and listed under [Deviations](#deviations-from-the-paper).

## Features

- Slotted SATECN simulator (IoT / 2 UAVs / 18 Starlink LEOs): Poisson task arrivals, no task splitting (Eq. 1),
  local and remote computing delay and energy (Eqs. 2-7), air-to-ground and LEO uplink rates (Eqs. 8-10),
  queue evolution (Eqs. 11-12), AoI (Eq. 14) and the AoI-energy penalty (Eq. 15).
- Real Starlink geometry: CelesTrak TLEs propagated with SGP4 and screened as in Sec. V-A
  (55 deg minimum elevation, 90 s continuous visibility). The screened set gives **2.73 visible satellites on
  average, the value the paper reports**.
- The six RL domains of Table II, with heterogeneous policy dimensions (3 to 15 parameters).
- Learners: episodic NAC, MSR-NAC (NAC with n-step returns), MSR-NAC-ELLA (within-domain lifelong learning),
  MSR-NAC-CDLL (cross-domain lifelong learning, Algorithm 1, Eqs. 20-25), and an exhaustive grid-search reference.
- One script per experiment and one script for all figures and tables; unit tests.

## Architecture

```
configs/default.yaml        every parameter, tagged [paper] or [assumed]
data/starlink_screened.txt  the 18 screened Starlink TLEs used by the simulator
src/satecn/
  orbits.py       TLE + SGP4 propagation, elevation/range, screening, synthetic fallback
  channel.py      Eqs. 8-10
  domains.py      Table II domains, RL task sampling (Eq. 17)
  environment.py  the slotted simulator (Eqs. 1-15)
  policy.py       softmax executor head + logit-normal resource heads, score function
  learners.py     roll-outs, NAC, MSR-NAC, Fisher / Hessian, natural step
  lifelong.py     CDLL knowledge base (L, Psi_g, s_k; Eqs. 22-25) and the ELLA baseline
  trainer.py      Algorithm 1 loop over an interleaved task stream
  exhaustive.py   grid-search baseline
  experiments.py  Figs. 3-9, Table V, ablations
scripts/
  run_experiments.py   run experiments -> experiments/results/*.json
  make_figures.py      figures -> experiments/graphs, tables -> experiments/results/*.csv
  screen_tle.py        rebuild data/starlink_screened.txt from a full CelesTrak TLE file
tests/test_core.py     unit tests
```

## Technology stack

Python 3.10+ (tested on 3.12 and 3.13) with NumPy, Matplotlib, PyYAML and sgp4. CPU only; no GPU or deep-learning framework is needed
(policies are linear with at most 15 parameters).

## Installation (Windows PowerShell)

```powershell
git clone https://github.com/AnsFahad/satecn-cdll.git
cd satecn-cdll
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# if activation is blocked:  Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m pytest -q
```

Linux/macOS: `python3 -m venv .venv && source .venv/bin/activate`, then the same `pip`/`pytest` commands.

## Dataset preparation

This is a simulation study, so there is no dataset to download. The only external data is the Starlink TLE
snapshot, already screened and included in `data/starlink_screened.txt`. To rebuild it from a fresh snapshot:

```powershell
Invoke-WebRequest "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle" -OutFile starlink_full.txt
python scripts/screen_tle.py --tle starlink_full.txt --start 2026-11-08T00:00:00
```

A fresh snapshot will select different satellites; update `leo.window_start_utc` in the config to the
start time the script prints. If `sgp4` is missing, the simulator falls back to synthetic overhead passes and
prints a notice.

## Execution

```powershell
# smoke test of every experiment at reduced size (about 5 minutes)
python scripts/run_experiments.py --exp all --quick --workers 4
# full reproduction (3 seeds x 4 algorithms x 24 tasks x 200 iterations, then all other experiments)
python scripts/run_experiments.py --exp all --workers 4
# single experiments
python scripts/run_experiments.py --exp main --seeds 0 --workers 4
python scripts/run_experiments.py --exp jumpstart,beta --workers 4
# figures and CSV tables from whatever results exist
python scripts/make_figures.py
```

`system`, `scalability` and `continual` reuse the agents trained by `main` (saved in `experiments/checkpoints`),
so run `main` first.

## Training and evaluation

One *iteration* visits all 24 tasks (4 per domain) in random order. Each visit rolls out one 120-slot episode
(the paper's learning round is 120-150 s), with all IoTs of the task sharing its policy, followed by one update.
Learning curves record the penalty of that exploring, on-policy episode. Metrics: penalty, AoI and scaled energy
per IoT per slot; jumpstart; convergence speedup relative to NAC; forgetting on old tasks.

## Results

See `RESULTS.md` (generated numbers and a comparison with the paper) and `experiments/graphs/`.

## Deviations from the paper

| Item | Paper | Here | Why |
|---|---|---|---|
| Arrival rate | 6-20 tasks/s per IoT | x0.05 (0.3-1.0 tasks/s) | 6-20 tasks/s of 1.2 MB tasks is ~100x the uplink and CPU capacity; Fig. 8b implies near-capacity load |
| LEO altitude filter | 520-580 km | 440-600 km | Starlink shells were lowered to ~480 km after the paper; 520-580 km now selects almost nothing |
| Ground site | not given | 42.37 N, 72.52 W (Amherst, MA) | authors' institution |
| Policy form | "compact, normalised" | softmax executor head + logit-normal resource heads, features [1, AoI, D_rem] | clipped Gaussians gave zero gradient outside [0,1] |
| MSR-NAC | NAC + n-step returns | critic with horizon/backlog features, n = 40, gamma = 1, fixed-KL natural step | details not given; discounting below 1 biased learning towards myopic energy savings |
| Hessian H_k | "estimate from PG" | Fisher information, normalised + 0.01 I | standard curvature choice for natural-gradient methods |
| KB step sizes | eta_L, eta_Psi | relative steps divided by a Lipschitz bound | raw steps diverged |
| s_k update | current task only | Lasso re-solved for all stored tasks after each update | otherwise changes to the shared L move other tasks' policies |
| New task init | s_k <- rand | mean code of known tasks | gives the jumpstart the paper measures; ablation included |
| Energy scale epsilon | not given | 1.0 | makes the AoI and energy terms comparable at beta = 0.8 |
| Exhaustive search | "optimum" | grid over stationary decisions (960 points in domain 6) | true optimum over state-dependent policies is intractable |
| Control-plane overhead (Table V) | measured | modelled (message sizes and link rates in `experiments.py`) | no SDN testbed |

## Limitations

- Simulation only. The channel parameters marked `[assumed]` change the absolute numbers.
- The paper's training details (learning rates, h, d_base, n, episodes per iteration) are unknown; ours were
  chosen once for stable learning and then fixed for every algorithm. They were not tuned per method.
- Learning curves are on-policy with exploration noise, so they sit above the deterministic grid-search line.

## Citation

```bibtex
@article{wu2026cdll,
  author  = {Wu, Yinxuan and Wang, Ning and Lorenzo, Beatriz and Samarakoon, Sumudu and Liu, Bing},
  title   = {Lifelong Learning-Based {SDN} Design for Dynamic Configuration and Resource Allocation in Satellite-Terrestrial Networks},
  journal = {IEEE Transactions on Wireless Communications},
  year    = {2026},
  doi     = {10.1109/TWC.2026.3716226}
}
```
