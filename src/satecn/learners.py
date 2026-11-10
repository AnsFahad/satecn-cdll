"""Single-task policy-gradient learners: episodic NAC [39] and the multi-step-return MSR-NAC.

Both return a natural-gradient direction w (penalty is minimised, so theta <- theta - step * w)
and the Fisher information F; a scaled F serves as the Hessian H_k in Eqs. (22)-(25).
"""
from __future__ import annotations

import numpy as np

from .environment import SATECNEnv
from .policy import LinearPolicy, features


def rollout(cfg, geom, task, policy: LinearPolicy, theta, rng, n_episodes=1, greedy=False, slots=None):
    """Run episodes with all IoTs of the task sharing policy theta.

    Returns arrays phi (E,T,n,3), grad (E,T,n,d), cost (E,T,n) and summary metrics.
    """
    T = slots or cfg["sim"]["episode_slots"]
    PHI, G, C, AOI, EN, ST, XC = [], [], [], [], [], [], []
    for _ in range(n_episodes):
        env = SATECNEnv(cfg, geom, task, seed=int(rng.integers(1 << 31)))
        s = env.reset()
        phis, grads, costs, aois, ens, xcs = [], [], [], [], [], []
        for _t in range(T):
            phi = features(s, cfg)
            xcs.append(env.critic_state())
            act, g = policy.act(theta, phi, rng, assign=env.assign, greedy=greedy)
            s, c, e = env.step(act)
            phis.append(phi); grads.append(g); costs.append(c); aois.append(s[:, 0]); ens.append(e)
        PHI.append(phis); G.append(grads); C.append(costs); AOI.append(aois); EN.append(ens); XC.append(xcs)
        st = dict(env.stats)
        st.update(env.queue_lengths())
        ST.append(st)
    PHI, G, C = np.asarray(PHI), np.asarray(G), np.asarray(C)
    escale = cfg["objective"]["energy_scale"]
    metrics = {
        "penalty": float(C.mean()),
        "aoi": float(np.mean(AOI)),
        "energy": float(escale * np.mean(EN)),
        "stats": {k: float(np.mean([s[k] for s in ST])) for k in ST[0]},
    }
    return {"phi": PHI, "grad": G, "cost": C, "xc": np.asarray(XC), "metrics": metrics}


def fisher(grad: np.ndarray) -> np.ndarray:
    """Per-step Fisher information F = E[grad log pi grad log pi^T]."""
    g = grad.reshape(-1, grad.shape[-1])
    return g.T @ g / len(g)


def hessian_from_fisher(F: np.ndarray, reg: float) -> np.ndarray:
    """H_k for Eqs. (22)-(25): Fisher scaled to unit mean eigenvalue (keeps KB steps well scaled)."""
    Fn = F / max(np.trace(F) / F.shape[0], 1e-12)
    return Fn + reg * np.eye(F.shape[0])


def _ridge_solve(X, y, ridge):
    A = X.T @ X + ridge * len(X) * np.eye(X.shape[1])
    return np.linalg.solve(A, X.T @ y)


def nac_episodic(batch, gamma: float, ridge: float):
    """Episodic NAC (Peters & Schaal 2008): regress R(tau) on [sum_t gamma^t grad log pi, 1]."""
    G, C = batch["grad"], batch["cost"]              # (E,T,n,d), (E,T,n)
    E, T, n, d = G.shape
    disc = gamma ** np.arange(T)
    psi = np.einsum("t,etnd->end", disc, G).reshape(E * n, d)
    R = np.einsum("t,etn->en", disc, C).reshape(E * n) / T
    X = np.hstack([psi, np.ones((E * n, 1))])
    sol = _ridge_solve(X, R, ridge)
    return sol[:d], fisher(G)


def _critic_feats(phi, xc=None, drem_norm=20.0):
    """Critic features: quadratic in (AoI, D_rem), remaining-horizon terms (finite window) and,
    when available, per-pipeline backlogs (critic-only information; the actor still sees x = {AoI, D_rem})."""
    a, d = phi[..., 1], phi[..., 2]
    T = phi.shape[1]
    tau = np.broadcast_to((1.0 - np.arange(T) / T)[None, :, None], a.shape)
    f = [np.ones_like(a), a, d, a * a, d * d, a * d, tau, tau * tau, tau * a, tau * d]
    if xc is not None:
        q = np.clip(xc / drem_norm, 0, 5)
        for j in range(q.shape[-1]):
            f += [q[..., j], tau * q[..., j]]
    return np.stack(f, axis=-1)


def msr_nac(batch, gamma: float, n_step: int, ridge: float):
    """MSR-NAC: NAC with compatible function approximation fitted to n-step returns.

    G_t^(n) = sum_{j<n} gamma^j c_{t+j} + gamma^n V(x_{t+n});  G_t^(n) - V(x_t) ~ grad log pi_t . w
    """
    PHI, G, C = batch["phi"], batch["grad"], batch["cost"]
    E, T, n, d = G.shape
    # critic V fitted to Monte-Carlo returns-to-go (truncated at the window end)
    mc = np.zeros_like(C)
    acc = np.zeros((E, n))
    for t in range(T - 1, -1, -1):
        acc = C[:, t] + gamma * acc
        mc[:, t] = acc
    Fc = _critic_feats(PHI, batch.get("xc"))
    v = _ridge_solve(Fc.reshape(-1, Fc.shape[-1]), mc.reshape(-1), ridge)
    V = Fc @ v                                         # (E,T,n)
    # n-step returns with bootstrap
    Gn = np.zeros_like(C)
    for t in range(T):
        end = min(t + n_step, T)
        disc = gamma ** np.arange(end - t)
        Gn[:, t] = np.einsum("k,ekn->en", disc, C[:, t:end])
        if end < T:
            Gn[:, t] += gamma ** (end - t) * V[:, end]
    adv = (Gn - V).reshape(-1)
    w = _ridge_solve(G.reshape(-1, d), adv, ridge)
    return w, fisher(G)


def natural_step(theta, w, F, step):
    """theta <- theta - step * w / sqrt(w^T F w): a fixed-KL natural-gradient step (penalty minimisation)."""
    q = float(w @ F @ w)
    return theta - step * w / np.sqrt(max(q, 1e-12))
