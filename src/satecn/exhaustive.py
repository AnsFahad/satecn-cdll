"""Exhaustive-search baseline: brute force over a discretised grid of stationary decisions.

Each grid point fixes the executor-selection probabilities and the normalised resource levels
(CPU share, remote CPU share, transmit power); every point is simulated and the best is kept.
"""
from __future__ import annotations

import itertools

import numpy as np

from .domains import DOMAINS, N_FEAT
from .learners import rollout
from .policy import LinearPolicy


def simplex_grid(k: int, step: float):
    m = int(round(1 / step))
    for c in itertools.product(range(m + 1), repeat=k):
        if sum(c) == m:
            yield np.array(c) / m


def grid_theta(gid: int, probs, cont_vals) -> np.ndarray:
    spec = DOMAINS[gid]
    th = []
    if len(spec.cats) > 1 and not spec.assigned:
        p = np.maximum(np.asarray(probs), 1e-6)
        for j in range(1, len(spec.cats)):
            th += [np.log(p[j] / p[0]), 0.0, 0.0]
    for v in cont_vals:
        v = min(max(v, 1e-3), 1 - 1e-3)
        th += [np.log(v / (1 - v)), 0.0, 0.0]          # logit-normal head: sigmoid(theta . phi) = v
    return np.array(th)


def exhaustive_search(cfg, geom, task, prob_step=0.25, levels=(0.25, 0.5, 0.75, 1.0), n_episodes=2, seed=0):
    spec = DOMAINS[task.gid]
    pol = LinearPolicy(spec, sigma=1e-9)
    ncat = len(spec.cats) if not spec.assigned else 1
    prob_set = list(simplex_grid(ncat, prob_step)) if ncat > 1 else [np.ones(1)]
    best = None
    results = []
    for probs in prob_set:
        for vals in itertools.product(levels, repeat=len(spec.cont)):
            th = grid_theta(task.gid, probs, vals)
            m = rollout(cfg, geom, task, pol, th, np.random.default_rng(seed), n_episodes=n_episodes)["metrics"]
            results.append((m["penalty"], probs.tolist(), list(vals)))
            if best is None or m["penalty"] < best[0]["penalty"]:
                best = (m, th, probs.tolist(), list(vals))
    return {"metrics": best[0], "theta": best[1], "probs": best[2], "levels": best[3], "n_evaluated": len(results)}
