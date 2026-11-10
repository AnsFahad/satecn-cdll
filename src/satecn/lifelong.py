"""Knowledge bases for lifelong policy-gradient learning.

CDLL (Sec. IV-C, Alg. 1):   theta_k = Psi_g L s_k, shared L (d_base x h), projection Psi_g (d_g x d_base),
                             updated with Eqs. (23)-(24); s_k from the Lasso of Eq. (25).
ELLA baseline (PG-ELLA):     theta_k = L_g s_k with a separate L_g per domain (no cross-domain sharing).
"""
from __future__ import annotations

import numpy as np


def lasso_quadratic(Q: np.ndarray, b: np.ndarray, mu1: float, s0: np.ndarray, iters: int = 100, tol=1e-6):
    """Coordinate descent for  min_s  s^T Q s - 2 b^T s + mu1 ||s||_1  (Q symmetric PSD)."""
    s = s0.copy()
    diag = np.maximum(np.diag(Q), 1e-12)
    for _ in range(iters):
        delta = 0.0
        for j in range(len(s)):
            r = b[j] - Q[j] @ s + Q[j, j] * s[j]
            new = np.sign(r) * max(abs(r) - mu1 / 2.0, 0.0) / diag[j]
            delta = max(delta, abs(new - s[j]))
            s[j] = new
        if delta < tol:
            break
    return s


class KnowledgeBase:
    def __init__(self, domain_dims: dict, lcfg: dict, rng: np.random.Generator, cross_domain: bool = True):
        self.cross = cross_domain
        self.h = lcfg["h"]
        self.eta_L, self.eta_psi = lcfg["eta_L"], lcfg["eta_psi"]
        self.mu1, self.mu2, self.mu3 = lcfg["mu1"], lcfg["mu2"], lcfg["mu3"]
        self.steps = lcfg["kb_steps"]
        self.warm = lcfg.get("warm_start", True)
        self.refit = lcfg.get("refit_all", True)
        self.rng = rng
        self.dims = dict(domain_dims)
        self.d_base = lcfg["d_base"]
        self.Psi: dict[int, np.ndarray] = {}
        self.Lg: dict[int, np.ndarray] = {}
        if self.cross:
            self.L = rng.normal(0, 1 / np.sqrt(self.h), (self.d_base, self.h))
        self.tasks: dict[int, dict] = {}   # kid -> {g, s, theta_hat, H}

    # ----------------------------------------------------------------- structure
    def _L(self, g):
        return self.L if self.cross else self.Lg[g]

    def _Psi(self, g):
        return self.Psi[g] if self.cross else None

    def basis(self, g) -> np.ndarray:
        """M_g = Psi_g L (CDLL) or L_g (ELLA), shape (d_g, h)."""
        return self.Psi[g] @ self.L if self.cross else self.Lg[g]

    def is_new_domain(self, g) -> bool:
        return (g not in self.Psi) if self.cross else (g not in self.Lg)

    def add_task(self, kid: int, g: int):
        """Alg. 1 lines 3-11: new task (and new domain if needed)."""
        if self.is_new_domain(g):
            d = self.dims[g]
            if self.cross:
                self.Psi[g] = self.rng.normal(0, 1 / np.sqrt(self.d_base), (d, self.d_base))
            else:
                self.Lg[g] = self.rng.normal(0, 1 / np.sqrt(self.h), (d, self.h))
        same = [t["s"] for t in self.tasks.values() if t["g"] == g]
        if self.warm and same:
            s = np.mean(same, axis=0)                # warm start from same-domain tasks
        elif self.warm and self.tasks and self.cross:
            s = np.mean([t["s"] for t in self.tasks.values()], axis=0)
        else:
            s = self.rng.random(self.h)               # Alg. 1 line 6: s_k <- rand
        self.tasks[kid] = {"g": g, "s": s, "theta_hat": None, "H": None}

    def theta(self, kid: int) -> np.ndarray:
        t = self.tasks[kid]
        return self.basis(t["g"]) @ t["s"]

    # ----------------------------------------------------------------- updates
    def _grad_L(self, only_g=None):
        """Gradient of Eq. (22) w.r.t. L (Eq. 23) and its Lipschitz bound (for a stable step)."""
        grad = self.mu3 * (self.L if self.cross else self.Lg[only_g])
        lip = self.mu3
        groups = {}
        for t in self.tasks.values():
            if t["theta_hat"] is None or (not self.cross and t["g"] != only_g):
                continue
            groups.setdefault(t["g"], []).append(t)
        for g, ts in groups.items():
            acc, lg = 0.0, 0.0
            for t in ts:
                s, H, th = t["s"], t["H"], t["theta_hat"]
                hn = np.linalg.norm(H, 2) * float(s @ s)
                if self.cross:
                    P = self.Psi[g]
                    acc = acc + (-P.T @ H @ np.outer(th, s) + P.T @ H @ P @ self.L @ np.outer(s, s))  # Eq. (23)
                    lg += hn * np.linalg.norm(P, 2) ** 2
                else:
                    acc = acc + (-H @ np.outer(th, s) + H @ self.Lg[g] @ np.outer(s, s))
                    lg += hn
            grad = grad + acc / len(ts)
            lip += lg / len(ts)
        return grad, lip

    def _grad_Psi(self, g):
        """Gradient of Eq. (22) w.r.t. Psi_g (Eq. 24) and its Lipschitz bound."""
        ts = [t for t in self.tasks.values() if t["g"] == g and t["theta_hat"] is not None]
        grad = self.mu2 * self.Psi[g]
        lip = self.mu2
        if ts:
            acc, lg = 0.0, 0.0
            for t in ts:
                Ls = self.L @ t["s"]
                acc = acc + (-t["H"] @ np.outer(t["theta_hat"], Ls) + t["H"] @ self.Psi[g] @ np.outer(Ls, Ls))  # Eq. (24)
                lg += np.linalg.norm(t["H"], 2) * float(Ls @ Ls)
            grad = grad + acc / len(ts)
            lip += lg / len(ts)
        return grad, lip

    def update(self, kid: int, theta_hat: np.ndarray, H: np.ndarray):
        """Alg. 1 lines 15-17: update L, Psi_g, then s_k (Lasso)."""
        t = self.tasks[kid]
        t["theta_hat"], t["H"] = theta_hat.copy(), H.copy()
        g = t["g"]
        # eta_L / eta_psi are relative step sizes, scaled by 1/Lipschitz so the bilinear updates stay stable
        for _ in range(self.steps):
            if self.cross:
                gL, lip = self._grad_L()
                self.L -= self.eta_L / lip * gL
                gP, lip = self._grad_Psi(g)
                self.Psi[g] -= self.eta_psi / lip * gP
            else:
                gL, lip = self._grad_L(only_g=g)
                self.Lg[g] -= self.eta_L / lip * gL
        if self.refit:
            self.refit_all()                                                  # keep stored policies consistent
        else:
            M = self.basis(g)
            t["s"] = lasso_quadratic(M.T @ H @ M, M.T @ H @ theta_hat, self.mu1, t["s"])  # Eq. (25)

    def refit_all(self):
        """Re-solve Eq. (25) for every stored task against the current basis, so a change of the
        shared L does not silently move the reconstructed policies of tasks not visited this round."""
        for t in self.tasks.values():
            if t["theta_hat"] is not None:
                M = self.basis(t["g"])
                t["s"] = lasso_quadratic(M.T @ t["H"] @ M, M.T @ t["H"] @ t["theta_hat"], self.mu1, t["s"])

    def recon_error(self) -> float:
        errs = [np.linalg.norm(self.theta(k) - t["theta_hat"]) / max(np.linalg.norm(t["theta_hat"]), 1e-9)
                for k, t in self.tasks.items() if t["theta_hat"] is not None]
        return float(np.mean(errs)) if errs else float("nan")

    def n_params(self) -> int:
        n = sum(t["s"].size for t in self.tasks.values())
        if self.cross:
            return n + self.L.size + sum(P.size for P in self.Psi.values())
        return n + sum(L.size for L in self.Lg.values())
