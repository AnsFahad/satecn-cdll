"""Experiment drivers reproducing Sec. V (Figs. 3-9, Table V) plus ablations.

Every experiment writes JSON into experiments/results; figures are produced by scripts/make_figures.py.
"""
from __future__ import annotations

import json
import pickle
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .config import ROOT, load_config
from .domains import DOMAINS, make_task, make_task_set
from .exhaustive import exhaustive_search
from .orbits import build_geometry
from .trainer import ALGOS, Agent, train_stream

RES = ROOT / "experiments" / "results"
CKPT = ROOT / "experiments" / "checkpoints"


def _save(name: str, obj):
    RES.mkdir(parents=True, exist_ok=True)
    with open(RES / f"{name}.json", "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else float(o))
    print(f"[saved] {RES / (name + '.json')}", flush=True)


def _load(name: str):
    with open(RES / f"{name}.json", encoding="utf-8") as f:
        return json.load(f)


def base_tasks(cfg, seed):
    return make_task_set(cfg, seed=1000 + seed, per_domain=cfg["tasks"]["per_domain"])


def new_tasks_domain6(cfg, seed):
    rng = np.random.default_rng(5000 + seed)
    n0 = 6 * cfg["tasks"]["per_domain"] + 1
    return [make_task(n0 + j, 6, cfg, rng) for j in range(cfg["tasks"]["new_domain6"])]


# ----------------------------------------------------------------------------- main (Figs. 3, 4, 7)
def _main_job(args):
    algo, seed, cfg_path, iters = args
    cfg = load_config(cfg_path)
    geom = build_geometry(cfg, ROOT)
    tasks = base_tasks(cfg, seed)
    agent = Agent(algo, cfg, seed=10 * seed + ALGOS.index(algo))
    t0 = time.time()
    hist = train_stream(agent, tasks, geom, iters, log_every=20, tag=f"s{seed} ")
    wall = time.time() - t0
    CKPT.mkdir(parents=True, exist_ok=True)
    with open(CKPT / f"{algo}_seed{seed}.pkl", "wb") as f:
        pickle.dump(agent, f)
    extra = {"recon_error": agent.kb.recon_error(), "kb_params": agent.kb.n_params()} if agent.kb else {}
    return algo, seed, {str(k): v for k, v in hist.items()}, {t.kid: t.gid for t in tasks}, wall, extra


def run_main(cfg_path, seeds, iters, workers):
    jobs = [(a, s, cfg_path, iters) for s in seeds for a in ALGOS]
    with Pool(workers) as pool:
        out = pool.map(_main_job, jobs)
    res = {"iterations": iters, "seeds": list(seeds), "algos": list(ALGOS), "runs": {}}
    for algo, seed, hist, kid2g, wall, extra in out:
        res["runs"][f"{algo}|{seed}"] = {"hist": hist, "wall_s": wall, **extra}
        res["task_domain"] = {str(k): g for k, g in kid2g.items()}
    _save("main", res)


# ----------------------------------------------------------------------------- exhaustive (Fig. 3 line)
def _exh_job(args):
    cfg_path, seed, kid = args
    cfg = load_config(cfg_path)
    geom = build_geometry(cfg, ROOT)
    task = [t for t in base_tasks(cfg, seed) if t.kid == kid][0]
    r = exhaustive_search(cfg, geom, task, n_episodes=1, seed=seed)
    from .learners import rollout
    from .policy import LinearPolicy
    m = rollout(cfg, geom, task, LinearPolicy(task.domain, 1e-9), r["theta"], np.random.default_rng(77), 3)["metrics"]
    return seed, kid, {"grid_best": r["metrics"]["penalty"], "eval": m, "probs": r["probs"], "levels": r["levels"],
                       "n_evaluated": r["n_evaluated"]}


def run_exhaustive(cfg_path, seeds, workers, domain=6):
    cfg = load_config(cfg_path)
    kids = [t.kid for t in base_tasks(cfg, 0) if t.gid == domain]
    jobs = [(cfg_path, s, k) for s in seeds for k in kids]
    with Pool(workers) as pool:
        out = pool.map(_exh_job, jobs)
    _save("exhaustive", {"domain": domain, "runs": {f"{s}|{k}": r for s, k, r in out}})


# ----------------------------------------------------------------------------- jumpstart (Fig. 6)
def _jump_job(args):
    cfg_path, seed, pre_rounds, n_eval, n_inits = args
    cfg = load_config(cfg_path)
    geom = build_geometry(cfg, ROOT)
    tasks = base_tasks(cfg, seed)
    rng = np.random.default_rng(9000 + seed)
    held = [make_task(200 + g, g, cfg, rng) for g in DOMAINS]           # one unseen task per domain
    out = {}
    for algo in ("msr_nac_cdll", "msr_nac_ella"):
        ag = Agent(algo, cfg, seed=100 + seed)
        train_stream(ag, tasks, geom, pre_rounds, log_every=0)
        out[algo] = {}
        for t in held:
            ag.ensure_task(t)
            out[algo][t.gid] = ag.evaluate(t, geom, n_episodes=n_eval, seed=seed)["penalty"]
    out["msr_nac"] = {}
    for t in held:                                                       # learning from scratch: random init
        vals = []
        for j in range(n_inits):
            ag = Agent("msr_nac", cfg, seed=300 + 17 * seed + j)
            vals.append(ag.evaluate(t, geom, n_episodes=n_eval, seed=seed)["penalty"])
        out["msr_nac"][t.gid] = float(np.mean(vals))
    return seed, out


def run_jumpstart(cfg_path, seeds, workers, pre_rounds=15, n_eval=3, n_inits=5):
    jobs = [(cfg_path, s, pre_rounds, n_eval, n_inits) for s in seeds]
    with Pool(workers) as pool:
        out = pool.map(_jump_job, jobs)
    _save("jumpstart", {"pre_rounds": pre_rounds, "runs": {str(s): r for s, r in out}})


# ----------------------------------------------------------------------------- beta sweep (Fig. 5)
def _beta_job(args):
    cfg_path, beta, seed, iters = args
    cfg = load_config(cfg_path, overrides={"objective.beta": beta})
    geom = build_geometry(cfg, ROOT)
    tasks = [t for t in base_tasks(cfg, seed) if t.gid == 6]
    ag = Agent("msr_nac_cdll", cfg, seed=400 + seed)
    train_stream(ag, tasks, geom, iters, log_every=0)
    ms = [ag.evaluate(t, geom, n_episodes=3, seed=seed) for t in tasks]
    return beta, seed, {k: float(np.mean([m[k] for m in ms])) for k in ("penalty", "aoi", "energy")}


def run_beta(cfg_path, seeds, workers, betas=(0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0), iters=100):
    jobs = [(cfg_path, b, s, iters) for b in betas for s in seeds]
    with Pool(workers) as pool:
        out = pool.map(_beta_job, jobs)
    res = {"iterations": iters, "betas": list(betas), "runs": {}}
    for b, s, m in out:
        res["runs"][f"{b}|{s}"] = m
    _save("beta_sweep", res)


# ----------------------------------------------------------------------------- system analysis (Fig. 8)
def run_system(cfg_path, seeds, n_eval=5):
    cfg = load_config(cfg_path)
    geom = build_geometry(cfg, ROOT)
    res = {}
    for seed in seeds:
        with open(CKPT / f"msr_nac_cdll_seed{seed}.pkl", "rb") as f:
            ag = pickle.load(f)
        for t in base_tasks(cfg, seed):
            m = ag.evaluate(t, geom, n_episodes=n_eval, seed=seed)
            res[f"{seed}|{t.kid}"] = {"gid": t.gid, **m}
    _save("system", res)


# ----------------------------------------------------------------------------- scalability (Table V)
CTRL = {"status_bytes": 32, "policy_bytes_per_param": 4, "seg_report_bytes": 64, "hap_km": 20.0,
        "backhaul_bps": 1.0e8, "proc_s": 1.0e-4}


def _scal_job(args):
    cfg_path, N, seed, slots = args
    cfg = load_config(cfg_path)
    geom = build_geometry(cfg, ROOT)
    with open(CKPT / "msr_nac_cdll_seed0.pkl", "rb") as f:
        ag = pickle.load(f)
    rng = np.random.default_rng(7000 + seed)
    task = make_task(900 + N, 6, cfg, rng, n_iot=N, n_uav=N // 5)
    ref = [k for k, t in ag.kb.tasks.items() if t["g"] == 6]
    theta = np.mean([ag.kb.theta(k) for k in ref], axis=0)              # domain-6 policy, size independent of N
    from .environment import SATECNEnv
    from .policy import LinearPolicy, features
    pol = LinearPolicy(DOMAINS[6], cfg["policy"]["sigma"])
    env = SATECNEnv(cfg, geom, task, seed=seed)
    s = env.reset()
    aoi, en, t0 = [], 0.0, time.time()
    for _ in range(slots):
        act, _ = pol.act(theta, features(s, cfg), rng)
        s, c, e = env.step(act)
        aoi.append(s[:, 0].mean())
        en += e.sum()
    sim_wall = time.time() - t0
    # control-plane accounting (modelled): per-slot status upload from every IoT, per-slot segment report
    # from each local SDNC to the SAT-SDNC on the HAP, and one policy dissemination per learning round.
    st = env.stats
    tx_e = st["e_tx_uav"] + st["e_tx_leo"]
    p_up = cfg["iot"]["p_max_uav"]
    r_ctrl = 1.0e6                                                     # control uplink on the 1 MHz IoT channel
    ctrl_bits = slots * N * CTRL["status_bytes"] * 8 + (slots // cfg["sim"]["episode_slots"] + 1) * N * \
        DOMAINS[6].dim * CTRL["policy_bytes_per_param"] * 8
    ctrl_e = ctrl_bits / r_ctrl * p_up
    seg_bits = CTRL["seg_report_bytes"] * 8 * (1 + N / 10)
    lat_ms = 1e3 * (seg_bits / CTRL["backhaul_bps"] + CTRL["hap_km"] * 1e3 / 3e8 + CTRL["proc_s"])
    total_e = en * cfg["objective"]["energy_scale"]
    return N, seed, {"aoi": float(np.mean(aoi)), "total_energy": float(total_e / slots),
                     "energy_per_iot": float(total_e / slots / N), "ctrl_energy_ratio": float(ctrl_e / max(tx_e, 1e-9)),
                     "ctrl_latency_ms": float(lat_ms), "sim_wall_s": sim_wall}


def run_scalability(cfg_path, workers, Ns=(80, 120, 160, 200, 240, 280), seeds=(0, 1, 2), slots=600):
    jobs = [(cfg_path, N, s, slots) for N in Ns for s in seeds]
    with Pool(workers) as pool:
        out = pool.map(_scal_job, jobs)
    _save("scalability", {"slots": slots, "ctrl_model": CTRL, "runs": {f"{N}|{s}": r for N, s, r in out}})


# ----------------------------------------------------------------------------- continual learning (Fig. 9)
def _cont_job(args):
    cfg_path, algo, seed, n_enc = args
    cfg = load_config(cfg_path)
    geom = build_geometry(cfg, ROOT)
    olds = base_tasks(cfg, seed)
    news = new_tasks_domain6(cfg, seed)
    p = CKPT / f"{algo}_seed{seed}.pkl"
    with open(p, "rb") as f:
        ag = pickle.load(f)                                              # pre-trained on tasks 1-24
    old_before = {t.kid: ag.evaluate(t, geom, n_episodes=2, seed=seed)["penalty"] for t in olds}
    rng = np.random.default_rng(6000 + seed)
    order = np.concatenate([rng.permutation(len(news)) for _ in range(n_enc // len(news))])
    seen, seq = set(), []
    for idx in order:
        t = news[idx]
        first = t.kid not in seen
        seen.add(t.kid)
        m = ag.train_step(t, geom)
        seq.append({"kid": t.kid, "first": first, "penalty": m["penalty"]})
    if ag.kb is not None:
        ag.kb.refit_all()
    old_after = {t.kid: ag.evaluate(t, geom, n_episodes=2, seed=seed)["penalty"] for t in olds}
    return algo, seed, {"seq": seq, "old_before": old_before, "old_after": old_after}


def run_continual(cfg_path, seeds, workers, n_enc=32):
    jobs = [(cfg_path, a, s, n_enc) for s in seeds for a in ALGOS]
    with Pool(workers) as pool:
        out = pool.map(_cont_job, jobs)
    _save("continual", {"runs": {f"{a}|{s}": r for a, s, r in out}})


# ----------------------------------------------------------------------------- ablations
def _abl_job(args):
    cfg_path, name, ov, algo, seed, iters = args
    doms = (5, 6) if algo == "msr_nac" else tuple(DOMAINS)              # CDLL variants need all domains
    cfg = load_config(cfg_path, overrides=ov)
    geom = build_geometry(cfg, ROOT)
    tasks = [t for t in base_tasks(cfg, seed) if t.gid in doms]
    ag = Agent(algo, cfg, seed=500 + seed)
    h = train_stream(ag, tasks, geom, iters, log_every=0)
    curve = np.mean([h[t.kid]["penalty"] for t in tasks], axis=0)
    return name, seed, {"curve": curve.tolist(), "mean": float(curve.mean()), "final": float(curve[-10:].mean())}


ABLATIONS = {
    "n_step=1": ({"learner.n_step": 1}, "msr_nac"),
    "n_step=10": ({"learner.n_step": 10}, "msr_nac"),
    "n_step=40 (default)": ({}, "msr_nac"),
    "n_step=120 (MC)": ({"learner.n_step": 120}, "msr_nac"),
    "CDLL h=4": ({"lifelong.h": 4}, "msr_nac_cdll"),
    "CDLL h=8 (default)": ({}, "msr_nac_cdll"),
    "CDLL h=15": ({"lifelong.h": 15}, "msr_nac_cdll"),
    "CDLL s_k=rand (Alg.1)": ({"lifelong.warm_start": False}, "msr_nac_cdll"),
    "CDLL no refit": ({"lifelong.refit_all": False}, "msr_nac_cdll"),
}


def run_ablation(cfg_path, seeds, workers, iters=100):
    jobs = [(cfg_path, n, ov, a, s, iters) for n, (ov, a) in ABLATIONS.items() for s in seeds]
    with Pool(workers) as pool:
        out = pool.map(_abl_job, jobs)
    _save("ablation", {"iterations": iters, "runs": {f"{n}|{s}": r for n, s, r in out}})
