"""Training loops for NAC, MSR-NAC, MSR-NAC-ELLA and MSR-NAC-CDLL (Alg. 1) over a task stream."""
from __future__ import annotations

import time

import numpy as np

from .domains import DOMAINS
from .learners import hessian_from_fisher, msr_nac, nac_episodic, natural_step, rollout
from .lifelong import KnowledgeBase
from .policy import LinearPolicy

ALGOS = ("msr_nac_cdll", "msr_nac_ella", "msr_nac", "nac")
LABELS = {"msr_nac_cdll": "MSR-NAC-CDLL", "msr_nac_ella": "MSR-NAC-ELLA", "msr_nac": "MSR-NAC", "nac": "NAC",
          "exhaustive": "Exhaustive search"}


class Agent:
    """SAT-SDNC learning agent holding either per-task policies or a knowledge base."""

    def __init__(self, algo: str, cfg: dict, seed: int, kb: KnowledgeBase | None = None):
        assert algo in ALGOS, algo
        self.algo, self.cfg = algo, cfg
        self.rng = np.random.default_rng(seed)
        self.lc = cfg["learner"]
        self.sigma = cfg["policy"]["sigma"]
        self.policies = {g: LinearPolicy(DOMAINS[g], self.sigma) for g in DOMAINS}
        self.theta: dict[int, np.ndarray] = {}
        self.iters: dict[int, int] = {}
        self.kb = kb
        if algo in ("msr_nac_cdll", "msr_nac_ella") and kb is None:
            dims = {g: DOMAINS[g].dim for g in DOMAINS}
            self.kb = KnowledgeBase(dims, cfg["lifelong"], self.rng, cross_domain=(algo == "msr_nac_cdll"))

    @property
    def lifelong(self) -> bool:
        return self.kb is not None

    def ensure_task(self, task):
        if task.kid in self.iters:
            return
        self.iters[task.kid] = 0
        if self.lifelong:
            self.kb.add_task(task.kid, task.gid)
        else:
            self.theta[task.kid] = self.rng.uniform(0, self.cfg["policy"]["init_scale"], DOMAINS[task.gid].dim)

    def current_theta(self, task) -> np.ndarray:
        self.ensure_task(task)
        return self.kb.theta(task.kid) if self.lifelong else self.theta[task.kid]

    def train_step(self, task, geom, learn=True):
        """One iteration on task k: collect trajectories, PG step, KB update. Returns metrics."""
        th = self.current_theta(task)
        pol = self.policies[task.gid]
        batch = rollout(self.cfg, geom, task, pol, th, self.rng, n_episodes=self.lc["episodes_per_iter"])
        if not learn:
            return batch["metrics"]
        lc = self.lc
        if self.algo == "nac":
            w, F = nac_episodic(batch, lc["gamma"], lc["ridge"])
        else:
            w, F = msr_nac(batch, lc["gamma"], lc["n_step"], lc["ridge"])
        step = lc["step_size"] * lc["step_decay"] ** self.iters[task.kid]
        theta_hat = natural_step(th, w, F, step)
        if self.lifelong:
            self.kb.update(task.kid, theta_hat, hessian_from_fisher(F, self.cfg["lifelong"]["hessian_reg"]))
        else:
            self.theta[task.kid] = theta_hat
        self.iters[task.kid] += 1
        return batch["metrics"]

    def evaluate(self, task, geom, n_episodes=3, slots=None, seed=None):
        rng = np.random.default_rng(seed if seed is not None else self.rng.integers(1 << 31))
        return rollout(self.cfg, geom, task, self.policies[task.gid], self.current_theta(task), rng,
                       n_episodes=n_episodes, slots=slots)["metrics"]


def train_stream(agent: Agent, tasks, geom, iterations: int, log_every: int = 20, tag: str = ""):
    """Interleaved lifelong training: every iteration visits all tasks in random order."""
    hist = {t.kid: {"penalty": [], "aoi": [], "energy": []} for t in tasks}
    t0 = time.time()
    for it in range(iterations):
        for idx in agent.rng.permutation(len(tasks)):
            task = tasks[idx]
            m = agent.train_step(task, geom)
            for k in ("penalty", "aoi", "energy"):
                hist[task.kid][k].append(m[k])
        if log_every and (it + 1) % log_every == 0:
            avg = np.mean([hist[t.kid]["penalty"][-1] for t in tasks])
            print(f"[{tag}{LABELS[agent.algo]}] iter {it + 1}/{iterations} mean penalty {avg:.2f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    return hist
