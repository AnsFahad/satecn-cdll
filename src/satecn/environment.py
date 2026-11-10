"""Slotted SATECN simulator implementing the system model of Sec. II-III.

Per slot and per IoT: Poisson arrivals, FCFS dispatch to the executor chosen by the policy
(local CPU, UAV or LEO; no task splitting, Eq. 1), local/remote computation (Eqs. 2-4),
uplink transmission (Eqs. 5-6, 9-11), queue evolution (Eq. 12), AoI (Eqs. 13-14) and the
AoI-energy penalty (Eq. 15).
"""
from __future__ import annotations

from collections import deque

import numpy as np

from .channel import leo_gain, leo_rate, uav_path_loss_lin, uav_rate
from .domains import TaskSpec
from .orbits import LEOGeometry

HOLD, LOCAL, UAV, LEO = 0, 1, 2, 3
CAT_CODE = {"hold": HOLD, "local": LOCAL, "uav": UAV, "leo": LEO}
BITS_PER_MB = 8.0e6


class SATECNEnv:
    def __init__(self, cfg: dict, geom: LEOGeometry | None, task: TaskSpec, seed: int | None = None):
        self.cfg, self.geom, self.task = cfg, geom, task
        self.spec = task.domain
        self.n = len(task.lam)
        self.U = task.meta.get("n_uav", self.spec.n_uav)
        self.use_leo = self.spec.use_leo and geom is not None
        self.L = geom.n_sats if self.use_leo else 0
        self.dt = cfg["sim"]["slot_s"]
        self.rng = np.random.default_rng(task.seed if seed is None else seed)
        io, uv, lo = cfg["iot"], cfg["uav"], cfg["leo"]
        self.eps_max_i = io["eps_max"] * self.dt
        self.eps_max_u = uv["eps_max"] * self.dt
        self.eps_max_l = lo["eps_max"] * self.dt
        self.beta = cfg["objective"]["beta"]
        self.escale = cfg["objective"]["energy_scale"]
        # domain 2: first half of the IoTs is served by UAVs, second half by LEOs
        self.assign = np.where(np.arange(self.n) < self.n // 2, UAV, LEO) if self.spec.assigned else None

    # ------------------------------------------------------------------ episode
    def reset(self):
        n = self.n
        self.t = 0
        self.unbound = [deque() for _ in range(n)]       # [gen, bits]
        self.lpipe = [deque() for _ in range(n)]         # [gen, cycles_left]
        self.lback = np.zeros(n)                          # local backlog (cycles)
        self.txq = {UAV: [deque() for _ in range(n)], LEO: [deque() for _ in range(n)]}  # [gen, bits_left, cycles]
        self.txback = {UAV: np.zeros(n), LEO: np.zeros(n)}
        self.nodeq = {}                                   # (node, i) -> deque [gen, cycles_left]
        self.unbound_bits = np.zeros(n)
        self.aoi = np.zeros(n)
        self.uav_xy = self.task.uav_xy0[: max(self.U, 1)].copy()
        self.uav_wp = self.rng.uniform(0, self.cfg["area"]["size_m"], size=self.uav_xy.shape)
        self.uav_v = self.rng.uniform(*self.cfg["uav"]["speed_mps"], size=self.uav_xy.shape[0])
        self.stats = {k: 0.0 for k in ("gen", "done_iot", "done_uav", "done_leo", "e_local", "e_tx_uav",
                                       "e_tx_leo", "e_uav", "e_leo")}
        return self.state()

    def state(self) -> np.ndarray:
        """x_t = {AoI, D_rem} per IoT; D_rem in MB (Sec. IV-B)."""
        drem = self.unbound_bits + self.txback[UAV] + self.txback[LEO]
        drem = drem + self.lback / self.task.rho
        return np.stack([self.aoi, drem / BITS_PER_MB], axis=1)

    def critic_state(self) -> np.ndarray:
        """Extra per-IoT observables for the critic only (pipeline backlogs, MB). Not used by the policy."""
        node = np.zeros(self.n)
        for (nd, i), q in self.nodeq.items():
            node[i] += sum(it[1] for it in q) / self.task.rho[i]
        return np.stack([self.unbound_bits, self.lback / self.task.rho, self.txback[UAV], self.txback[LEO], node],
                        axis=1) / BITS_PER_MB

    def queue_lengths(self) -> dict:
        loc = sum(len(q) + len(p) for q, p in zip(self.unbound, self.lpipe))
        to_u = sum(len(q) for q in self.txq[UAV]) + sum(len(q) for (nd, _), q in self.nodeq.items() if nd < 100)
        to_l = sum(len(q) for q in self.txq[LEO]) + sum(len(q) for (nd, _), q in self.nodeq.items() if nd >= 100)
        return {"local": loc, "to_uav": to_u, "to_leo": to_l}

    # ------------------------------------------------------------------ helpers
    def _move_uavs(self):
        d = self.uav_wp - self.uav_xy
        dist = np.linalg.norm(d, axis=1)
        step = self.uav_v * self.dt
        arrive = dist <= step
        self.uav_xy = np.where(arrive[:, None], self.uav_wp, self.uav_xy + d / np.maximum(dist, 1e-9)[:, None] * step[:, None])
        if arrive.any():
            self.uav_wp[arrive] = self.rng.uniform(0, self.cfg["area"]["size_m"], size=(arrive.sum(), 2))

    @staticmethod
    def _serve(q: deque, budget: float, done_gens: list) -> float:
        """FCFS service of a queue whose entries hold remaining work at index 1. Returns work used."""
        used = 0.0
        while q and budget > 1e-9:
            item = q[0]
            w = min(item[1], budget)
            item[1] -= w
            budget -= w
            used += w
            if item[1] <= 1e-9:
                done_gens.append(q.popleft())
        return used

    # ------------------------------------------------------------------ step
    def step(self, act: dict):
        """Advance one slot. act: cat (n,) codes, eps/eps_o/p (n,) fractions in [0, 1]."""
        cfg, n, t, dt = self.cfg, self.n, self.t, self.dt
        io, uv, lo = cfg["iot"], cfg["uav"], cfg["leo"]
        rho = self.task.rho
        cat = act["cat"]
        eps = np.clip(act["eps"], 0, 1)
        eps_o = np.clip(act["eps_o"], 0, 1)
        pfr = np.clip(act["p"], 0, 1)
        energy = np.zeros(n)
        fresh = np.full(n, -np.inf)       # freshest generation time completed this slot

        # 1) arrivals (Poisson, sizes sampled around D_i)
        cnt = self.rng.poisson(self.task.lam)
        for i in np.nonzero(cnt)[0]:
            sizes = np.maximum(self.rng.normal(self.task.d_mean_mb[i], io["per_task_cv"] * self.task.d_mean_mb[i], cnt[i]), 0.05)
            for s in sizes * BITS_PER_MB:
                self.unbound[i].append([t, s])
                self.unbound_bits[i] += s
        self.stats["gen"] += cnt.sum()

        # 2) channels: best UAV / best visible LEO per IoT by instantaneous rate
        r_uav = np.zeros(n)
        r_leo = np.zeros(n)
        best_u = np.zeros(n, dtype=int)
        best_l = np.zeros(n, dtype=int)
        p_uav = io["p_max_uav"] * pfr
        p_leo = io["p_max_leo"] * pfr
        if self.U > 0:
            self._move_uavs()
            pl = uav_path_loss_lin(self.task.iot_xy, self.uav_xy[: self.U], uv)
            best_u = np.argmin(pl, axis=1)
            r_uav = uav_rate(pl[np.arange(n), best_u], p_uav, uv)
        if self.use_leo:
            tt = (t + self.task.leo_offset) % self.geom.window
            vis = self.geom.elev_deg[:, tt] >= self.geom.min_elev
            if vis.any():
                g = leo_gain(self.geom.range_km[:, tt], lo, self.rng, n)
                g[:, ~vis] = 0.0
                best_l = np.argmax(g, axis=1)
                r_leo = leo_rate(g[np.arange(n), best_l], p_leo, lo)
        rate = {UAV: r_uav, LEO: r_leo}
        ptx = {UAV: p_uav, LEO: p_leo}

        # 3) dispatch: bind head-of-line tasks to the chosen executor (no splitting)
        for i in range(n):
            c = cat[i]
            ub = self.unbound[i]
            if c == HOLD or not ub:
                continue
            if c == LOCAL:
                cap = eps[i] * self.eps_max_i
                while ub and self.lback[i] < cap:
                    g_, b = ub.popleft()
                    self.unbound_bits[i] -= b
                    self.lpipe[i].append([g_, b * rho[i]])
                    self.lback[i] += b * rho[i]
            else:
                cap = rate[c][i] * dt
                q = self.txq[c][i]
                while ub and self.txback[c][i] < cap:
                    g_, b = ub.popleft()
                    self.unbound_bits[i] -= b
                    q.append([g_, b, b * rho[i]])
                    self.txback[c][i] += b

        # 4) local computation, Eqs. (3)-(4): E = xi * alpha * cycles * f^2
        f_loc = eps * self.eps_max_i / dt
        for i in range(n):
            if self.lpipe[i]:
                done = []
                used = self._serve(self.lpipe[i], eps[i] * self.eps_max_i, done)
                self.lback[i] -= used
                e = io["xi_cpt"] * io["alpha"] * used * f_loc[i] ** 2
                energy[i] += e
                self.stats["e_local"] += e
                if done:
                    self.stats["done_iot"] += len(done)
                    fresh[i] = max(fresh[i], max(d[0] for d in done))

        # 5) remote computation at UAV / LEO nodes with admission scaling (16d)
        if self.nodeq:
            by_node = {}
            for (nd, i), q in self.nodeq.items():
                if q:
                    by_node.setdefault(nd, []).append(i)
            for nd, users in by_node.items():
                is_leo = nd >= 100
                cmax = self.eps_max_l if is_leo else self.eps_max_u
                alpha = lo["alpha"] if is_leo else uv["alpha"]
                xi = lo["xi_cpt"] if is_leo else uv["xi_cpt"]
                req = np.maximum(eps_o[users], 0.05) * cmax
                if req.sum() > cmax:
                    req *= cmax / req.sum()
                for i, alloc in zip(users, req):
                    done = []
                    used = self._serve(self.nodeq[(nd, i)], alloc, done)
                    e = xi * alpha * used * (alloc / dt) ** 2
                    energy[i] += e
                    self.stats["e_leo" if is_leo else "e_uav"] += e
                    if done:
                        self.stats["done_leo" if is_leo else "done_uav"] += len(done)
                        fresh[i] = max(fresh[i], max(d[0] for d in done))

        # 6) uplink transmission, Eqs. (5)-(6), (11); single radio per IoT
        for i in range(n):
            c = cat[i]
            if c in (UAV, LEO) and self.txback[c][i] > 0:
                link = c
            elif self.txback[UAV][i] > 0:
                link = UAV
            elif self.txback[LEO][i] > 0:
                link = LEO
            else:
                continue
            r = rate[link][i]
            if r <= 0:
                continue
            q = self.txq[link][i]
            sent, budget = 0.0, r * dt
            node = (int(best_u[i]) if link == UAV else 100 + int(best_l[i]), i)
            while q and budget > 1e-9:
                item = q[0]
                w = min(item[1], budget)
                item[1] -= w
                budget -= w
                sent += w
                if item[1] <= 1e-9:
                    q.popleft()
                    self.nodeq.setdefault(node, deque()).append([item[0], item[2]])
            self.txback[link][i] -= sent
            e = io["xi_tr"] * ptx[link][i] * sent / r
            energy[i] += e
            self.stats["e_tx_uav" if link == UAV else "e_tx_leo"] += e

        # 7) AoI, Eq. (14): +1 if nothing completes, otherwise age of the freshest completed task
        upd = np.isfinite(fresh)
        self.aoi = np.where(upd, np.minimum(self.aoi + 1, t + 1 - np.where(upd, fresh, 0)), self.aoi + 1)
        self.t += 1

        # 8) penalty, Eq. (15)
        cost = self.beta * self.aoi + self.escale * (1 - self.beta) * energy
        return self.state(), cost, energy
