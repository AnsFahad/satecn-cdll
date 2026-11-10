# Results

All numbers below were produced by `python scripts/run_experiments.py --exp all` followed by
`python scripts/make_figures.py` (3 seeds unless stated). Figures are in `experiments/graphs/`.
Paper values are from Wu et al., IEEE TWC 2026.

## Penalty at the end of training (last 10 of 200 iterations)
| algorithm | D1 | D2 | D3 | D4 | D5 | D6 | all |
|---|---|---|---|---|---|---|---|
| MSR-NAC-CDLL | 27.81 | 25.45 | 18.74 | 20.12 | 18.16 | 19.60 | 21.65 |
| MSR-NAC-ELLA | 29.17 | 25.55 | 18.73 | 22.13 | 19.58 | 18.22 | 22.23 |
| MSR-NAC | 26.40 | 24.53 | 18.22 | 18.20 | 18.41 | 17.12 | 20.48 |
| NAC | 28.26 | 25.04 | 19.89 | 21.69 | 20.36 | 19.83 | 22.51 |

## Domain-6 learning-curve summary (Fig. 3)
| algorithm | first5_iters | last10_iters | mean_200 |
|---|---|---|---|
| MSR-NAC-CDLL | 21.40 | 19.60 | 20.02 |
| MSR-NAC-ELLA | 19.93 | 18.22 | 18.78 |
| MSR-NAC | 19.45 | 17.12 | 18.07 |
| NAC | 19.67 | 19.83 | 19.74 |
| Exhaustive search (grid, deterministic) | - | 14.88 | - |

## Convergence speedup relative to NAC (Fig. 7)
| algorithm | D1 | D2 | D3 | D4 | D5 | D6 | mean |
|---|---|---|---|---|---|---|---|
| MSR-NAC-CDLL | 0.36 | 0.35 | 1.66 | 0.22 | 0.69 | 0.41 | 0.61 |
| MSR-NAC-ELLA | 1.03 | 0.02 | 4.72 | 3.12 | 0.34 | 0.78 | 1.67 |
| MSR-NAC | 2.78 | 3.50 | 2.99 | 2.25 | 1.00 | 2.26 | 2.46 |
| NAC | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

## Jumpstart on unseen tasks, % penalty reduction vs MSR-NAC from random init (Fig. 6)
| algorithm | D1 | D2 | D3 | D4 | D5 | D6 | mean |
|---|---|---|---|---|---|---|---|
| MSR-NAC-CDLL | 3.1 | -6.0 | 5.5 | -5.1 | -7.2 | -4.9 | -2.4 |
| MSR-NAC-ELLA | -0.8 | -8.3 | 0.5 | -9.8 | -7.1 | -9.2 | -5.8 |

## AoI-energy trade-off vs beta, CDLL on domain-6 tasks, seed 0 (Fig. 5)
| beta | aoi | energy | penalty |
|---|---|---|---|
| 0.0 | 38.953 | 1.996 | 1.996 |
| 0.2 | 24.117 | 3.012 | 7.233 |
| 0.4 | 23.169 | 5.635 | 12.649 |
| 0.6 | 18.734 | 15.568 | 17.468 |
| 0.7 | 17.560 | 19.127 | 18.030 |
| 0.8 | 16.869 | 21.959 | 17.887 |
| 0.85 | 16.606 | 22.981 | 17.562 |
| 0.9 | 16.504 | 24.418 | 17.296 |
| 0.95 | 16.235 | 26.278 | 16.737 |
| 1.0 | 16.127 | 27.131 | 16.127 |

## Scalability, U = N/5, L = 18, 600 slots (Table V; control-plane columns are modelled)
| N | aoi | total_energy_per_slot | energy_per_iot_per_slot | ctrl_energy_pct_modelled | ctrl_latency_ms_modelled | sim_wall_s |
|---|---|---|---|---|---|---|
| 80 | 98.069 | 875.302 | 10.941 | 0.00403 | 0.213 | 0.833 |
| 120 | 89.086 | 1303.902 | 10.866 | 0.00422 | 0.233 | 1.225 |
| 160 | 93.512 | 1682.448 | 10.515 | 0.00413 | 0.254 | 2.091 |
| 200 | 86.377 | 2043.199 | 10.216 | 0.00432 | 0.274 | 2.536 |
| 240 | 87.154 | 2429.022 | 10.121 | 0.00429 | 0.295 | 2.967 |
| 280 | 87.909 | 2763.173 | 9.868 | 0.00426 | 0.315 | 3.870 |

## Continual learning on new tasks 25-28 and forgetting on tasks 1-24 (Fig. 9)
| algorithm | new_first_encounter | new_subsequent | old_tasks_before | old_tasks_after |
|---|---|---|---|---|
| MSR-NAC-CDLL | 18.57 | 19.10 | 21.62 | 21.61 |
| MSR-NAC-ELLA | 18.56 | 18.26 | 22.20 | 22.19 |
| MSR-NAC | 19.43 | 19.23 | 20.48 | 20.48 |
| NAC | 19.01 | 19.03 | 22.52 | 22.52 |

## Ablations (seed 0, 100 iterations)
| variant | mean_penalty | final_penalty |
|---|---|---|
| n_step=1 | 22.527 | 23.754 |
| n_step=10 | 18.745 | 18.616 |
| n_step=40 (default) | 18.993 | 19.207 |
| n_step=120 (MC) | 18.929 | 18.671 |
| CDLL h=4 | 23.705 | 23.398 |
| CDLL h=8 (default) | 22.519 | 22.188 |
| CDLL h=15 | 22.746 | 22.962 |
| CDLL s_k=rand (Alg.1) | 22.619 | 21.770 |
| CDLL no refit | 24.354 | 25.630 |

## Comparison with the paper

| Claim | Paper | Ours | Verdict |
|---|---|---|---|
| Mean visible LEOs after screening | 2.73 | 2.73 | reproduced |
| MSR-NAC beats NAC in every domain | yes | overall yes (-9.0% at end); per domain 6/6 on Linux, 5/6 on Windows | mostly |
| CDLL beats single-domain ELLA | most domains | 5 of 6 domains | reproduced |
| Up to 70% lower AoI/energy | up to 70% | largest gap ~18% | not reproduced |
| Convergence speedup vs NAC | > 8x | CDLL 0.61x (Linux) / 1.10x (Windows), MSR-NAC ~2.4x | not reproduced |
| Jumpstart vs MSR-NAC | 27-67% | -2.4% (CDLL), -5.8% (ELLA) | not reproduced |
| AoI from beta=0 to beta=1 | 38.58 -> 20.54 | 38.95 -> 16.13 | reproduced |
| Win-win region beta in [0, 0.85] | AoI and energy both fall | energy rises | not reproduced |
| LEOs help (domain 5 -> 6) | 61.9 -> 48.5 | 17.9 -> 19.7 | not reproduced |
| Scaling 80 -> 280 IoTs | AoI +3%, energy/IoT -29% | AoI -10%, energy/IoT -10% | trend reproduced |
| No catastrophic forgetting | qualitative | old-task change <= 0.01 | reproduced |

## Cross-machine check

The full suite was re-run on a Windows laptop (Python 3.12.4, NumPy 2.5.4) with the same config and seeds.
The tables above are from the reference run (Linux, Python 3.13, NumPy 2.5.3).

| End of training | All: Linux | All: Windows | D3: Linux | D3: Windows |
|---|---|---|---|---|
| MSR-NAC | 20.48 | 21.08 | 18.22 | 21.52 |
| MSR-NAC-CDLL | 21.65 | 22.13 | 18.74 | 20.65 |
| MSR-NAC-ELLA | 22.23 | 22.95 | 18.73 | 23.25 |
| NAC | 22.51 | 22.51 | 19.89 | 19.69 |

Stable across machines: overall ranking of MSR-NAC (best) and CDLL (second), domains 1, 2, 4, 5, 6, jumpstart,
no forgetting. Not stable: domain 3 (MSR-NAC vs NAC reverses) and the convergence-speedup values for CDLL and
ELLA. Tiny floating-point differences flip early sampled decisions; domain-3 task outcomes are very sensitive.
Expect the same trends, not identical decimals, when you re-run.

Why the numbers differ from the paper, and the limitations of this study, are discussed in Chapters 7-8 of the report.
