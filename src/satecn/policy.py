"""Compact linear policies pi_theta(a|x) for each RL domain (Table II).

theta = [softmax logits for executor categories (reference category fixed at 0) | logit-space means of
continuous resource controls], each head linear in phi(x) = [1, AoI/a_norm, D_rem/d_norm].
Continuous controls (CPU share, remote CPU share, transmit-power share) are logit-normal:
z ~ N(w.phi, sigma^2), a = sigmoid(z) in (0, 1), so no clipping flattens the gradient.
"""
from __future__ import annotations

import numpy as np

from .domains import N_FEAT, DomainSpec
from .environment import CAT_CODE, LEO, UAV


def features(state: np.ndarray, cfg: dict) -> np.ndarray:
    a = np.clip(state[:, 0] / cfg["sim"]["aoi_norm"], 0, 3)
    d = np.clip(state[:, 1] / cfg["sim"]["drem_norm_mb"], 0, 3)
    return np.stack([np.ones_like(a), a, d], axis=1)


class LinearPolicy:
    def __init__(self, spec: DomainSpec, sigma: float):
        self.spec = spec
        self.sigma = sigma
        self.n_logit = 0 if spec.assigned else max(len(spec.cats), 1) - 1
        self.codes = np.array([CAT_CODE[c] for c in spec.cats])
        self.d = spec.dim

    def act(self, theta: np.ndarray, phi: np.ndarray, rng: np.random.Generator, assign=None, greedy=False):
        """Sample actions for all IoTs. Returns (env action dict, grad log pi of shape (n, d))."""
        n = phi.shape[0]
        grad = np.zeros((n, self.d))
        act = {"cat": np.full(n, self.codes[0] if len(self.codes) else LEO), "eps": np.zeros(n),
               "eps_o": np.ones(n), "p": np.ones(n)}
        off = 0
        if self.n_logit > 0:
            W = theta[: self.n_logit * N_FEAT].reshape(self.n_logit, N_FEAT)
            z = np.concatenate([np.zeros((n, 1)), phi @ W.T], axis=1)
            z -= z.max(axis=1, keepdims=True)
            pr = np.exp(z)
            pr /= pr.sum(axis=1, keepdims=True)
            if greedy:
                k = pr.argmax(axis=1)
            else:
                k = (pr.cumsum(axis=1) > rng.random((n, 1))).argmax(axis=1)
            onehot = np.zeros_like(pr)
            onehot[np.arange(n), k] = 1.0
            g = (onehot - pr)[:, 1:]                       # d log softmax / d logits
            grad[:, : self.n_logit * N_FEAT] = (g[:, :, None] * phi[:, None, :]).reshape(n, -1)
            act["cat"] = self.codes[k]
        off = self.n_logit * N_FEAT
        if self.spec.assigned:
            act["cat"] = assign.copy()
        for name in self.spec.cont:
            w = theta[off: off + N_FEAT]
            mu = phi @ w
            z = mu if greedy else mu + self.sigma * rng.standard_normal(n)
            gw = ((z - mu) / self.sigma ** 2)[:, None] * phi
            a = 1.0 / (1.0 + np.exp(-z))
            if name in ("p_uav", "p_leo"):
                mask = assign == (UAV if name == "p_uav" else LEO)
                act["p"] = np.where(mask, a, act["p"])
                grad[:, off: off + N_FEAT] = gw * mask[:, None]
            else:
                act[name] = a
                grad[:, off: off + N_FEAT] = gw
            off += N_FEAT
        return act, grad
