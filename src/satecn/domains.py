"""RL domains of Table II and the RL-task descriptor Z_{i,t}^k of Eq. (17)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

N_FEAT = 3  # policy features phi(x) = [1, AoI/norm, D_rem/norm]


@dataclass(frozen=True)
class DomainSpec:
    gid: int
    configs: tuple            # SAT configurations of Table I covered by this domain
    cats: tuple               # executor categories chosen by the softmax head
    cont: tuple               # continuous resource controls (Gaussian heads)
    n_iot: int
    n_uav: int
    use_leo: bool
    local: bool               # IoT local computing available
    assigned: bool = False    # domain 2: executor fixed per IoT, only power is learned

    @property
    def dim(self) -> int:
        n_logit = 0 if self.assigned else max(len(self.cats), 1) - 1
        return n_logit * N_FEAT + len(self.cont) * N_FEAT


# Table II. Domain 2 assigns 10 IoTs to the UAVs and 10 to the LEOs and learns power only.
DOMAINS = {
    1: DomainSpec(1, (1,), ("hold", "local"), ("eps",), 10, 0, False, True),
    2: DomainSpec(2, (2, 3), ("uav", "leo"), ("p_uav", "p_leo"), 20, 2, True, False, assigned=True),
    3: DomainSpec(3, (6,), ("uav", "leo"), (), 10, 2, True, False),
    4: DomainSpec(4, (6,), ("uav", "leo"), ("p",), 10, 2, True, False),
    5: DomainSpec(5, (4,), ("local", "uav"), ("eps", "eps_o", "p"), 10, 2, False, True),
    6: DomainSpec(6, (4, 5, 7), ("local", "uav", "leo"), ("eps", "eps_o", "p"), 10, 2, True, True),
}


@dataclass
class TaskSpec:
    """One RL task k: a domain plus sampled environment parameters (Eq. 17)."""
    kid: int
    gid: int
    lam: np.ndarray           # task arrival rate per IoT (tasks/slot)
    d_mean_mb: np.ndarray     # mean task size per IoT (MB)
    rho: np.ndarray           # processing density per IoT (cycles/bit)
    iot_xy: np.ndarray        # IoT positions (m)
    uav_xy0: np.ndarray       # initial UAV positions (m)
    leo_offset: int           # start offset (s) inside the LEO visibility window
    seed: int
    meta: dict = field(default_factory=dict)

    @property
    def domain(self) -> DomainSpec:
        return DOMAINS[self.gid]

    def descriptor(self) -> np.ndarray:
        """Compact descriptor used for task identification (isNewTask in Alg. 1)."""
        return np.array([self.gid, self.lam.mean(), self.d_mean_mb.mean(), self.rho.mean(), self.leo_offset])


def make_task(kid: int, gid: int, cfg: dict, rng: np.random.Generator, n_iot: int | None = None,
              n_uav: int | None = None) -> TaskSpec:
    """Sample one RL task. n_iot / n_uav override the domain defaults (scalability study, Table V)."""
    spec = DOMAINS[gid]
    n = n_iot or spec.n_iot
    U = spec.n_uav if n_uav is None else n_uav
    io = cfg["iot"]
    lam = rng.uniform(*io["lambda_range"], size=n) * io["lambda_scale"] * cfg["sim"]["slot_s"]
    d = np.maximum(rng.normal(io["task_mb_mean"], io["task_mb_std"], size=n), io["task_mb_min"])
    rho = rng.uniform(*io["rho_range"], size=n)
    size = cfg["area"]["size_m"]
    iot_xy = rng.uniform(0, size, size=(n, 2))
    uav_xy0 = rng.uniform(0, size, size=(max(U, 1), 2))
    off = int(rng.integers(0, cfg["leo"]["window_s"]))
    return TaskSpec(kid, gid, lam, d, rho, iot_xy, uav_xy0, off, int(rng.integers(1 << 30)), meta={"n_uav": U})


def make_task_set(cfg: dict, seed: int, per_domain: int, domains=(1, 2, 3, 4, 5, 6), start_kid: int = 1):
    """Tasks are numbered domain-major (tasks 1-4 domain 1, ..., 21-24 domain 6) as in Sec. V."""
    rng = np.random.default_rng(seed)
    tasks, kid = [], start_kid
    for g in domains:
        for _ in range(per_domain):
            tasks.append(make_task(kid, g, cfg, rng))
            kid += 1
    return tasks
