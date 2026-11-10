"""Unit tests: run with  python -m pytest -q  (or  python tests/test_core.py)."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from satecn.config import load_config  # noqa: E402
from satecn.domains import DOMAINS, make_task_set  # noqa: E402
from satecn.environment import LEO, LOCAL, SATECNEnv  # noqa: E402
from satecn.lifelong import KnowledgeBase, lasso_quadratic  # noqa: E402
from satecn.orbits import synthetic_geometry  # noqa: E402
from satecn.policy import LinearPolicy  # noqa: E402

CFG = load_config()
GEOM = synthetic_geometry(18, 600, 55.0, seed=1)


def test_domain_dimensions():
    # heterogeneous policy dimensions d_g of Table II domains
    assert [DOMAINS[g].dim for g in range(1, 7)] == [6, 6, 3, 6, 12, 15]


def test_env_invariants():
    task = make_task_set(CFG, 0, 1, domains=(6,))[0]
    env = SATECNEnv(CFG, GEOM, task, seed=0)
    s = env.reset()
    n = env.n
    for t in range(60):
        cat = np.where(np.arange(n) % 2 == 0, LOCAL, LEO)
        prev = s[:, 0].copy()
        s, c, e = env.step({"cat": cat, "eps": np.full(n, .7), "eps_o": np.ones(n), "p": np.ones(n)})
        assert np.all(e >= 0) and np.all(np.isfinite(c))
        assert np.all(s[:, 0] <= prev + 1 + 1e-9)          # Eq. (14): AoI grows by at most one per slot
        assert np.all(s[:, 1] >= -1e-9)                      # remaining data is non-negative (Eq. 12)
    st = env.stats
    assert st["done_iot"] + st["done_uav"] + st["done_leo"] <= st["gen"]


def test_policy_score_function():
    """grad log pi from the policy equals a finite-difference of log pi (softmax + logit-normal heads)."""
    spec = DOMAINS[6]
    pol = LinearPolicy(spec, sigma=0.5)
    rng = np.random.default_rng(3)
    theta = rng.normal(size=spec.dim)
    phi = np.array([[1.0, 0.4, 0.2]])

    def logp(th, cat_idx, z):
        W = th[:6].reshape(2, 3)
        logits = np.concatenate([[0.0], W @ phi[0]])
        lp = logits[cat_idx] - np.log(np.exp(logits).sum())
        for j in range(3):
            mu = th[6 + 3 * j: 9 + 3 * j] @ phi[0]
            lp += -0.5 * ((z[j] - mu) / 0.5) ** 2
        return lp

    act, g = pol.act(theta, phi, np.random.default_rng(11))
    cat_idx = list(pol.codes).index(act["cat"][0])
    z = [np.log(act[k][0] / (1 - act[k][0])) for k in ("eps", "eps_o", "p")]
    num = np.zeros(spec.dim)
    for i in range(spec.dim):
        d = np.zeros(spec.dim)
        d[i] = 1e-6
        num[i] = (logp(theta + d, cat_idx, z) - logp(theta - d, cat_idx, z)) / 2e-6
    assert np.allclose(num, g[0], atol=1e-4)


def test_lasso_kkt():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(10, 5))
    Q = A.T @ A
    b = rng.normal(size=5)
    mu = 0.5
    s = lasso_quadratic(Q, b, mu, np.zeros(5), iters=2000, tol=1e-12)
    grad = 2 * (Q @ s - b)
    for j in range(5):  # KKT conditions of min s'Qs - 2b's + mu|s|_1
        if abs(s[j]) > 1e-8:
            assert abs(grad[j] + mu * np.sign(s[j])) < 1e-5
        else:
            assert abs(grad[j]) <= mu + 1e-5


def test_kb_fits_shared_structure():
    """CDLL KB recovers policies generated from a shared low-rank basis across two domains."""
    rng = np.random.default_rng(1)
    lcfg = dict(CFG["lifelong"], h=3, d_base=6, kb_steps=50, mu1=1e-4)
    kb = KnowledgeBase({1: 6, 6: 15}, lcfg, rng, cross_domain=True)
    Ltrue = rng.normal(size=(6, 3))
    P = {1: rng.normal(size=(6, 6)) / 2, 6: rng.normal(size=(15, 6)) / 2}
    for k in range(8):
        g = 1 if k % 2 else 6
        kb.add_task(k, g)
    err0 = None
    for it in range(40):
        for k in range(8):
            g = kb.tasks[k]["g"]
            th = P[g] @ Ltrue @ np.array([1.0, -0.5, 0.3]) * (1 + 0.2 * k)
            kb.update(k, th, np.eye(len(th)))
        if err0 is None:
            err0 = kb.recon_error()
    assert kb.recon_error() < 0.5 * err0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
