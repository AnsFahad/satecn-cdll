"""Turn experiments/results/*.json into figures (experiments/graphs) and tables (experiments/results).

Usage: python scripts/make_figures.py
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "experiments" / "results"
FIG = ROOT / "experiments" / "graphs"
FIG.mkdir(parents=True, exist_ok=True)

ALGOS = ["msr_nac_cdll", "msr_nac_ella", "msr_nac", "nac"]
LABEL = {"msr_nac_cdll": "MSR-NAC-CDLL", "msr_nac_ella": "MSR-NAC-ELLA", "msr_nac": "MSR-NAC", "nac": "NAC"}
COLOR = {"msr_nac_cdll": "#2a78d6", "msr_nac_ella": "#eb6834", "msr_nac": "#1baf7a", "nac": "#eda100"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"
DOMS = [1, 2, 3, 4, 5, 6]

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight",
})
SUMMARY = {}


def load(name):
    p = RES / f"{name}.json"
    if not p.exists():
        print(f"[skip] {p.name} missing")
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def smooth(x, w=10):
    x = np.asarray(x, float)
    if len(x) < w:
        return x
    c = np.cumsum(np.insert(x, 0, 0.0))
    head = c[1:w] / np.arange(1, w)
    return np.concatenate([head, (c[w:] - c[:-w]) / w])


def write_csv(name, header, rows):
    with open(RES / f"{name}.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(header)
        wr.writerows(rows)


def bars(ax, data, algos, ylabel, title):
    """Grouped bars per domain; data[algo] = (means[6], stds[6])."""
    n = len(algos)
    width = 0.8 / n
    x = np.arange(len(DOMS))
    for j, a in enumerate(algos):
        m, s = data[a]
        ax.bar(x + (j - (n - 1) / 2) * width, m, width * 0.92, yerr=s, color=COLOR[a], label=LABEL[a],
               error_kw={"elinewidth": 0.8, "ecolor": MUTED, "capsize": 2})
    ax.set_xticks(x, [f"D{g}" for g in DOMS])
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10, color=INK)
    ax.grid(axis="x", visible=False)


# ------------------------------------------------------------------------------------------- main
def per_domain_curves(main, algo, metric):
    """{seed: {g: mean curve over that domain's tasks}}"""
    out = {}
    for s in main["seeds"]:
        h = main["runs"][f"{algo}|{s}"]["hist"]
        out[s] = {g: np.mean([h[k][metric] for k, gg in main["task_domain"].items() if gg == g], axis=0)
                  for g in DOMS}
    return out


def fig_main(main, exh):
    iters = main["iterations"]
    # Fig. 3: domain-6 convergence
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    rows = []
    for a in ALGOS:
        cur = np.array([per_domain_curves(main, a, "penalty")[s][6] for s in main["seeds"]])
        m, sd = smooth(cur.mean(0), 5), smooth(cur.std(0), 5)
        ax.plot(np.arange(1, iters + 1), m, color=COLOR[a], lw=2, label=LABEL[a])
        ax.fill_between(np.arange(1, iters + 1), m - sd, m + sd, color=COLOR[a], alpha=0.12, lw=0)
        rows.append([LABEL[a], f"{cur.mean(0)[:5].mean():.2f}", f"{cur.mean(0)[-10:].mean():.2f}", f"{cur.mean():.2f}"])
    if exh:
        ev = np.mean([r["eval"]["penalty"] for r in exh["runs"].values()])
        ax.axhline(ev, color=MUTED, ls="--", lw=1.5, label="Exhaustive search (grid)")
        rows.append(["Exhaustive search (grid, deterministic)", "-", f"{ev:.2f}", "-"])
        SUMMARY["exhaustive_d6"] = ev
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Average penalty (domain 6)")
    ax.set_title("Average penalty vs. iterations, RL domain 6 (mean \u00b1 std over seeds)", fontsize=10)
    ax.legend(ncol=2, fontsize=8)
    fig.savefig(FIG / "fig3_domain6_convergence.png")
    plt.close(fig)
    write_csv("fig3_domain6", ["algorithm", "first5_iters", "last10_iters", "mean_200"], rows)

    # Fig. 4: per-domain averages over all training iterations
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
    rows = []
    for ax, (met, yl, tt) in zip(axs, [("penalty", "Penalty", "(a) System penalty per RL domain"),
                                       ("aoi", "AoI (slots)", "(b) AoI per RL domain"),
                                       ("energy", "Scaled energy cost", "(c) Scaled energy per RL domain")]):
        data = {}
        for a in ALGOS:
            cur = per_domain_curves(main, a, met)
            per_seed = np.array([[cur[s][g].mean() for g in DOMS] for s in main["seeds"]])
            data[a] = (per_seed.mean(0), per_seed.std(0))
            for gi, g in enumerate(DOMS):
                rows.append([met, LABEL[a], g, f"{per_seed.mean(0)[gi]:.3f}", f"{per_seed.std(0)[gi]:.3f}"])
        bars(ax, data, ALGOS, yl, tt)
        SUMMARY[f"fig4_{met}"] = {LABEL[a]: data[a][0].round(2).tolist() for a in ALGOS}
    axs[0].legend(fontsize=8, ncol=2)
    fig.savefig(FIG / "fig4_domain_comparison.png")
    plt.close(fig)
    write_csv("fig4_per_domain", ["metric", "algorithm", "domain", "mean", "std_over_seeds"], rows)

    # final-iteration (converged) per-domain penalty
    rows = []
    fin = {}
    for a in ALGOS:
        cur = per_domain_curves(main, a, "penalty")
        v = np.array([[cur[s][g][-10:].mean() for g in DOMS] for s in main["seeds"]])
        fin[a] = v.mean(0)
        rows.append([LABEL[a]] + [f"{x:.2f}" for x in v.mean(0)] + [f"{v.mean():.2f}"])
    write_csv("final_penalty_per_domain", ["algorithm"] + [f"D{g}" for g in DOMS] + ["all"], rows)
    SUMMARY["final_penalty"] = {LABEL[a]: fin[a].round(2).tolist() for a in ALGOS}

    # Fig. 7: convergence speedup relative to NAC
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    data, rows = {}, []
    for a in ALGOS:
        sp = []
        for s in main["seeds"]:
            row = []
            for g in DOMS:
                nac = smooth(per_domain_curves(main, "nac", "penalty")[s][g], 10)
                target = nac[-20:].mean()
                def conv(c):
                    idx = np.nonzero(smooth(c, 10) <= target)[0]
                    return (idx[0] + 1) if len(idx) else iters
                row.append(conv(per_domain_curves(main, "nac", "penalty")[s][g]) /
                           conv(per_domain_curves(main, a, "penalty")[s][g]))
            sp.append(row)
        sp = np.array(sp)
        data[a] = (sp.mean(0), sp.std(0))
        rows.append([LABEL[a]] + [f"{x:.2f}" for x in sp.mean(0)] + [f"{sp.mean():.2f}"])
    bars(ax, data, ALGOS, "Speedup vs. NAC (\u00d7)", "Convergence speedup relative to NAC")
    ax.axhline(1, color=MUTED, lw=0.8)
    ax.legend(fontsize=8, ncol=2)
    fig.savefig(FIG / "fig7_convergence_speedup.png")
    plt.close(fig)
    write_csv("fig7_speedup", ["algorithm"] + [f"D{g}" for g in DOMS] + ["mean"], rows)
    SUMMARY["speedup"] = {r[0]: r[1:] for r in rows}
    SUMMARY["wall_s"] = {k: v["wall_s"] for k, v in main["runs"].items()}
    SUMMARY["recon_error"] = {k: v.get("recon_error") for k, v in main["runs"].items() if "recon_error" in v}


# ------------------------------------------------------------------------------------------- others
def fig_jumpstart(js):
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    data, rows = {}, []
    for a in ("msr_nac_cdll", "msr_nac_ella"):
        red = np.array([[100 * (r["msr_nac"][str(g)] - r[a][str(g)]) / r["msr_nac"][str(g)] for g in DOMS]
                        for r in js["runs"].values()])
        data[a] = (red.mean(0), red.std(0))
        rows.append([LABEL[a]] + [f"{x:.1f}" for x in red.mean(0)] + [f"{red.mean():.1f}"])
    bars(ax, data, ["msr_nac_cdll", "msr_nac_ella"], "Penalty reduction vs. MSR-NAC (%)",
         f"Jumpstart on unseen tasks after {js['pre_rounds']} pre-training rounds")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.legend(fontsize=8)
    fig.savefig(FIG / "fig6_jumpstart.png")
    plt.close(fig)
    write_csv("fig6_jumpstart", ["algorithm"] + [f"D{g}" for g in DOMS] + ["mean"], rows)
    SUMMARY["jumpstart_pct"] = {r[0]: r[1:] for r in rows}


def fig_beta(bs):
    betas = bs["betas"]
    seeds = sorted({k.split("|")[1] for k in bs["runs"]})
    agg = {k: np.array([np.mean([bs["runs"][f"{b}|{s}"][k] for s in seeds]) for b in betas]) for k in ("aoi", "energy", "penalty")}
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
    axs[0].plot(agg["aoi"], agg["energy"], "-o", color=COLOR["msr_nac_cdll"], ms=6, lw=1.5)
    for b, x, y in zip(betas, agg["aoi"], agg["energy"]):
        axs[0].annotate(f"{b:g}", (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8, color=MUTED)
    axs[0].set_xlabel("Average AoI")
    axs[0].set_ylabel("Average scaled energy")
    axs[0].set_title("(a) AoI-energy trade-off across \u03b2", fontsize=10)
    axs[1].plot(betas, agg["aoi"], "-o", color=COLOR["msr_nac_cdll"], ms=6, lw=2)
    axs[1].set_xlabel("\u03b2")
    axs[1].set_ylabel("Average AoI")
    axs[1].set_title("(b) AoI vs. \u03b2", fontsize=10)
    axs[2].plot(betas, agg["energy"], "-o", color=COLOR["msr_nac_ella"], ms=6, lw=2)
    axs[2].set_xlabel("\u03b2")
    axs[2].set_ylabel("Average scaled energy")
    axs[2].set_title("(c) Energy vs. \u03b2", fontsize=10)
    for ax in axs[1:]:
        ax.axvspan(0.8, 0.9, color=GRID, alpha=0.6, lw=0)
    fig.savefig(FIG / "fig5_beta_sweep.png")
    plt.close(fig)
    write_csv("fig5_beta", ["beta", "aoi", "energy", "penalty"],
              [[b, f"{a:.3f}", f"{e:.3f}", f"{p:.3f}"] for b, a, e, p in zip(betas, agg["aoi"], agg["energy"], agg["penalty"])])
    SUMMARY["beta"] = {str(b): [round(a, 2), round(e, 2)] for b, a, e in zip(betas, agg["aoi"], agg["energy"])}


def fig_system(sysr):
    keys = ["penalty", "aoi", "energy"]
    st_keys = ["gen", "done_iot", "done_uav", "done_leo", "local", "to_uav", "to_leo",
               "e_local", "e_tx_uav", "e_tx_leo", "e_uav", "e_leo"]
    per = {g: {k: [] for k in keys + st_keys} for g in DOMS}
    for r in sysr.values():
        g = r["gid"]
        for k in keys:
            per[g][k].append(r[k])
        for k in st_keys:
            per[g][k].append(r["stats"][k])
    avg = {g: {k: float(np.mean(v)) for k, v in per[g].items()} for g in DOMS}
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    x = np.arange(6)
    axs[0].bar(x - 0.2, [avg[g]["penalty"] for g in DOMS], 0.37, color=COLOR["msr_nac_ella"], label="Penalty")
    axs[0].bar(x + 0.2, [avg[g]["aoi"] for g in DOMS], 0.37, color=COLOR["msr_nac_cdll"], label="AoI")
    for i, g in enumerate(DOMS):
        axs[0].text(i - 0.2, avg[g]["penalty"], f"{avg[g]['penalty']:.1f}", ha="center", va="bottom", fontsize=7, color=INK)
        axs[0].text(i + 0.2, avg[g]["aoi"], f"{avg[g]['aoi']:.1f}", ha="center", va="bottom", fontsize=7, color=INK)
    axs[0].set_title("(a) Penalty and AoI, converged CDLL policy", fontsize=10)
    axs[0].legend(fontsize=8)
    parts = [("done_iot", "Completed by IoT", "#2a78d6"), ("done_uav", "Completed by UAV", "#1baf7a"),
             ("done_leo", "Completed by LEO", "#eb6834"), ("local", "Waiting at IoT", "#86b6ef"),
             ("to_uav", "Waiting: to UAV", "#9fdcc4"), ("to_leo", "Waiting: to LEO", "#f5b394")]
    bottom = np.zeros(6)
    for k, lab, c in parts:
        v = np.array([avg[g][k] for g in DOMS])
        axs[1].bar(x, v, 0.6, bottom=bottom, color=c, label=lab, edgecolor="white", linewidth=1)
        bottom += v
    axs[1].plot(x, [avg[g]["gen"] for g in DOMS], "--o", color=INK, ms=4, lw=1, label="Generated tasks")
    axs[1].set_ylabel("Tasks per 120-slot episode")
    axs[1].set_title("(b) Where tasks are completed or waiting", fontsize=10)
    axs[1].legend(fontsize=7, ncol=2)
    eparts = [("e_local", "IoT compute", "#184f95"), ("e_tx_uav", "Tx to UAV", "#0e7a54"),
              ("e_tx_leo", "Tx to LEO", "#a8461f"), ("e_uav", "UAV compute", "#5ec79f"), ("e_leo", "LEO compute", "#f08d63")]
    bottom = np.zeros(6)
    for k, lab, c in eparts:
        v = np.array([avg[g][k] for g in DOMS]) / 1e3
        axs[2].bar(x, v, 0.6, bottom=bottom, color=c, label=lab, edgecolor="white", linewidth=1)
        bottom += v
    axs[2].set_ylabel("Energy per episode (kJ)")
    axs[2].set_title("(c) Energy breakdown", fontsize=10)
    axs[2].legend(fontsize=7)
    for ax in axs:
        ax.set_xticks(x, [f"D{g}" for g in DOMS])
        ax.grid(axis="x", visible=False)
    fig.savefig(FIG / "fig8_system_analysis.png")
    plt.close(fig)
    write_csv("fig8_system", ["domain"] + keys + st_keys, [[g] + [f"{avg[g][k]:.3f}" for k in keys + st_keys] for g in DOMS])
    SUMMARY["system"] = {g: {k: round(avg[g][k], 2) for k in keys + ["gen", "done_iot", "done_uav", "done_leo"]} for g in DOMS}


def table_scal(sc):
    Ns = sorted({int(k.split("|")[0]) for k in sc["runs"]})
    rows = []
    for N in Ns:
        rs = [v for k, v in sc["runs"].items() if int(k.split("|")[0]) == N]
        m = {k: np.mean([r[k] for r in rs]) for k in ("aoi", "total_energy", "energy_per_iot", "ctrl_energy_ratio",
                                                        "ctrl_latency_ms", "sim_wall_s")}
        rows.append([N, f"{m['aoi']:.3f}", f"{m['total_energy']:.3f}", f"{m['energy_per_iot']:.3f}",
                     f"{100 * m['ctrl_energy_ratio']:.5f}", f"{m['ctrl_latency_ms']:.3f}", f"{m['sim_wall_s']:.3f}"])
    write_csv("table5_scalability", ["N", "aoi", "total_energy_per_slot", "energy_per_iot_per_slot",
                                     "ctrl_energy_pct_modelled", "ctrl_latency_ms_modelled", "sim_wall_s"], rows)
    SUMMARY["scalability"] = rows


def fig_continual(ct):
    seeds = sorted({k.split("|")[1] for k in ct["runs"]})
    s0 = seeds[0]
    fig = plt.figure(figsize=(14, 4))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.2, 1.1, 1.2])
    ax = fig.add_subplot(gs[0])
    for a in ALGOS:
        seq = ct["runs"][f"{a}|{s0}"]["seq"]
        y = [e["penalty"] for e in seq]
        ax.plot(np.arange(1, len(y) + 1), y, color=COLOR[a], lw=1.5, label=LABEL[a])
        f = [i for i, e in enumerate(seq) if e["first"]]
        ax.plot(np.array(f) + 1, [y[i] for i in f], "o", color=COLOR[a], ms=8, mec="white", mew=1.5)
    ax.set_xlabel("Encounter index (randomised sequence of new tasks 25-28; dots = first encounter)")
    ax.set_ylabel("Penalty")
    ax.set_title(f"(a) Penalty on new domain-6 tasks (seed {s0})", fontsize=10)
    ax.legend(fontsize=8, ncol=2)
    ax = fig.add_subplot(gs[1])
    data, pos, cols = [], [], []
    for j, a in enumerate(ALGOS):
        first = [e["penalty"] for s in seeds for e in ct["runs"][f"{a}|{s}"]["seq"] if e["first"]]
        sub = [e["penalty"] for s in seeds for e in ct["runs"][f"{a}|{s}"]["seq"] if not e["first"]]
        data += [first, sub]
        pos += [3 * j, 3 * j + 1]
        cols += [COLOR[a], COLOR[a]]
    bp = ax.boxplot(data, positions=pos, widths=0.8, patch_artist=True, showfliers=False)
    for i, (b, c) in enumerate(zip(bp["boxes"], cols)):
        b.set_facecolor(c)
        b.set_alpha(0.9 if i % 2 == 0 else 0.45)
        if i % 2:
            b.set_hatch("//")
    ax.set_xticks([3 * j + 0.5 for j in range(4)], [LABEL[a].replace("MSR-NAC-", "") for a in ALGOS], fontsize=8)
    ax.set_title("(b) First (solid) vs. later (hatched) encounters", fontsize=10)
    ax.grid(axis="x", visible=False)
    ax = fig.add_subplot(gs[2])
    kids = sorted({e["kid"] for e in ct["runs"][f"{ALGOS[0]}|{s0}"]["seq"]})
    M = np.array([[np.mean([e["penalty"] for s in seeds for e in ct["runs"][f"{a}|{s}"]["seq"] if e["kid"] == k])
                   for a in ALGOS] for k in kids])
    im = ax.imshow(M, cmap="Blues", aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=8,
                    color="white" if M[i, j] > M.mean() else INK)
    ax.set_xticks(range(4), [LABEL[a].replace("MSR-NAC-", "") for a in ALGOS], fontsize=8)
    ax.set_yticks(range(len(kids)), [f"Task {k}" for k in kids])
    ax.grid(False)
    ax.set_title("(c) Mean penalty per new task", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.savefig(FIG / "fig9_continual_learning.png")
    plt.close(fig)
    rows = []
    for a in ALGOS:
        fb = [np.mean(list(ct["runs"][f"{a}|{s}"]["old_before"].values())) for s in seeds]
        fa = [np.mean(list(ct["runs"][f"{a}|{s}"]["old_after"].values())) for s in seeds]
        first = [e["penalty"] for s in seeds for e in ct["runs"][f"{a}|{s}"]["seq"] if e["first"]]
        sub = [e["penalty"] for s in seeds for e in ct["runs"][f"{a}|{s}"]["seq"] if not e["first"]]
        rows.append([LABEL[a], f"{np.mean(first):.2f}", f"{np.mean(sub):.2f}", f"{np.mean(fb):.2f}", f"{np.mean(fa):.2f}"])
    write_csv("fig9_continual", ["algorithm", "new_first_encounter", "new_subsequent", "old_tasks_before",
                                 "old_tasks_after"], rows)
    SUMMARY["continual"] = rows


def fig_ablation(ab):
    names = list(dict.fromkeys(k.split("|")[0] for k in ab["runs"]))
    mean = [np.mean([v["mean"] for k, v in ab["runs"].items() if k.split("|")[0] == n]) for n in names]
    final = [np.mean([v["final"] for k, v in ab["runs"].items() if k.split("|")[0] == n]) for n in names]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    y = np.arange(len(names))
    ax.barh(y + 0.2, mean, 0.38, color=COLOR["msr_nac_cdll"], label="mean over training")
    ax.barh(y - 0.2, final, 0.38, color=COLOR["msr_nac_ella"], label="last 10 iterations")
    ax.set_yticks(y, names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Average penalty")
    ax.set_title("Ablations (n-step: domains 5-6, MSR-NAC; CDLL: all domains)", fontsize=10)
    ax.legend(fontsize=8)
    fig.savefig(FIG / "fig10_ablation.png")
    plt.close(fig)
    write_csv("ablation", ["variant", "mean_penalty", "final_penalty"],
              [[n, f"{m:.3f}", f"{f:.3f}"] for n, m, f in zip(names, mean, final)])
    SUMMARY["ablation"] = {n: [round(m, 2), round(f, 2)] for n, m, f in zip(names, mean, final)}


def main():
    main_r, exh = load("main"), load("exhaustive")
    if main_r:
        fig_main(main_r, exh)
    for name, fn in [("jumpstart", fig_jumpstart), ("beta_sweep", fig_beta), ("system", fig_system),
                     ("scalability", table_scal), ("continual", fig_continual), ("ablation", fig_ablation)]:
        r = load(name)
        if r:
            fn(r)
    (RES / "summary.json").write_text(json.dumps(SUMMARY, indent=1, default=float), encoding="utf-8")
    print(f"figures -> {FIG}\ntables  -> {RES}")


if __name__ == "__main__":
    sys.exit(main())
