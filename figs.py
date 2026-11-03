import os as _os
_ROOT = _os.path.dirname(_os.path.abspath(__file__))
FIGDIR = _os.environ.get("FIGDIR", _os.path.join(_ROOT, "figures"))
PNGDIR = _os.path.join(FIGDIR, "png")
_os.makedirs(PNGDIR, exist_ok=True)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, json, sys, os
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5})
C = {"loop": "#1f4e79", "ucb_screen": "#2e8b57", "twin_only": "#b8860b", "twin_validate": "#c0504d",
     "measure_once": "#7f7f7f", "loop_blind": "#8064a2"}
OUT = FIGDIR
R = lambda n: json.load(open(f"results/{n}.json"))


def twinq():
    d = R("twinq"); qs = sorted(d, key=float)
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    lab = {"twin_only": "twin only", "twin_validate": "simulate-then-validate", "loop": "loop",
           "ucb_screen": "UCB, pre-screened", "measure_once": "measure-once", "loop_blind": "loop, no pre-screening"}
    for p in ["twin_only", "twin_validate", "loop", "ucb_screen", "measure_once", "loop_blind"]:
        m = [100 * d[q][p][0] for q in qs]; h = [100 * d[q][p][1] for q in qs]
        ax.errorbar([float(q) for q in qs], m, yerr=h, marker="o", ms=3.5, lw=1.2, capsize=2, color=C[p], label=lab[p])
    ax.set_xlabel("twin error $q$ (sd as a multiple of the spread of utilities)"); ax.set_ylabel("shipped utility (% of optimum)")
    ax.legend(fontsize=6.5, ncol=2, frameon=False, loc="lower left")
    fig.savefig(f"{OUT}/fig_twinq.pdf", bbox_inches="tight")


def landscape():
    import worlds as W
    smp = W.testbed_samples(); V = W.testbed_twin()
    U = np.array([s.mean() for s in smp]); sd = np.array([s.std(ddof=1) for s in smp])
    wn = [json.loads(l) for l in open("data/window_noise.jsonl")]
    fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.5))
    ax[0].hist(U, bins=40, color="#1f4e79", alpha=0.85, label="measured (bank mean)")
    ax[0].hist(V, bins=40, color="#b8860b", alpha=0.5, label="twin")
    ax[0].set_xlabel("mean utility $U$ (dimensionless, 0–1)"); ax[0].set_ylabel("number of configurations"); ax[0].legend(fontsize=6.5, frameon=False)
    ax[0].set_title("(a) landscape", fontsize=8, loc="left")
    ax[1].scatter(V, U, s=3, alpha=0.5, color="#1f4e79"); lim = [min(U.min(), V.min()), max(U.max(), V.max())]
    ax[1].plot(lim, lim, color="#999", lw=0.8, ls="--"); ax[1].set_xlabel("twin-predicted utility (dimensionless, 0–1)"); ax[1].set_ylabel("measured utility $U$ (0–1)")
    ax[1].set_title("(b) twin vs testbed", fontsize=8, loc="left")
    by = {}
    for d in wn:
        by.setdefault((d["x"], d["dur"]), []).append(d["u"])
    xs = sorted({k[0] for k in by}); ws = sorted({k[1] for k in by})
    for x in xs:
        ax[2].plot(ws, [np.std(by[(x, w)], ddof=1) for w in ws], marker="o", ms=3, lw=1, label=x)
    ax[2].plot(ws, [np.sqrt(np.mean([np.var(by[(x, w)], ddof=1) for x in xs])) for w in ws], color="k", lw=1.6, label="pooled")
    ax[2].set_xlabel("measurement window (s)"); ax[2].set_ylabel("sd of one test (utility, 0–1)")
    ax[2].set_xticks(ws); ax[2].set_ylim(0, None); ax[2].legend(fontsize=5.5, frameon=False)
    ax[2].set_title("(c) noise vs window", fontsize=8, loc="left")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_landscape.pdf", bbox_inches="tight")


def dynamics():
    d = R("dynamics"); T = len(d["shipped"]); x = 5 * (np.arange(T) + 1)
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    for k, c, l in [("belief", "#c0504d", "belief about incumbent"), ("found", "#2e8b57", "best tested (found)"),
                    ("shipped", "#1f4e79", "incumbent (shipped)"), ("cautious", "#8064a2", "lower-confidence choice")]:
        m = np.array([100 * a for a, b in d[k]]); h = np.array([100 * b for a, b in d[k]])
        ax.plot(x, m, color=c, lw=1.3, label=l); ax.fill_between(x, m - h, m + h, color=c, alpha=0.2, lw=0)
    ax.axhline(100, color="#999", lw=0.8, ls="--"); ax.set_xscale("log")
    ax.set_xlabel("number of tests"); ax.set_ylabel("shipped utility (% of optimum)"); ax.legend(fontsize=6.5, frameon=False)
    fig.savefig(f"{OUT}/fig_dynamics.pdf", bbox_inches="tight")


def allocation():
    d = R("allocation")
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=False)
    for i, (tier, lab) in enumerate([("testbed", "noise scale"), ("zoo", "noise")]):
        order = {"0.5": 0, "1.0": 1, "2.0": 2, "tm": 0, "mid": 1, "high": 2}
        for key, col in zip(sorted(d[tier], key=lambda k: order[k]), ["#2e8b57", "#1f4e79", "#c0504d"]):
            nn = sorted(d[tier][key], key=float)
            m = [100 * d[tier][key][n][0] for n in nn]; h = [100 * d[tier][key][n][1] for n in nn]
            nm = {"0.5": "$\\sigma/\\sqrt{2}$", "1.0": "$\\sigma$ (measured)", "2.0": "$2\\sigma$",
                  "tm": "low", "mid": "mid", "high": "high"}[key]
            ax[i].errorbar([float(n) for n in nn], m, yerr=h, marker="o", ms=3.5, capsize=2, lw=1.2, color=col,
                           label=f"{lab}: {nm}")
            j = int(np.argmax(m)); ax[i].plot(float(nn[j]), m[j], "o", ms=8, mfc="none", mec=col)
        ax[i].set_xlabel("new configurations per round of 5 tests"); ax[i].legend(fontsize=6.5, frameon=False)
    ax[0].set_ylabel("shipped utility (% of optimum)"); ax[1].set_ylabel("shipped utility (% of optimum)")
    ax[0].set_title("(a) testbed (replay)", fontsize=8, loc="left"); ax[1].set_title("(b) topology landscapes", fontsize=8, loc="left")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_allocation.pdf", bbox_inches="tight")


def bias():
    d = R("bias")
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    for tier, col, lab in [("testbed", "#1f4e79", "testbed (replay)"), ("zoo", "#c0504d", "topology landscapes, mid noise")]:
        ks = sorted(d[tier], key=float); base = d[tier][ks[0]][0]
        m = [100 * (d[tier][k][0] - base) for k in ks]; h = [100 * d[tier][k][1] for k in ks]
        ax.errorbar([float(k) for k in ks], m, yerr=h, marker="o", ms=3.5, capsize=2, lw=1.2, color=col, label=lab)
    ax.axvline(1, color="#999", ls="--", lw=0.8); ax.axhline(0, color="#999", lw=0.6)
    ax.set_xlabel("bias of the incumbent's re-test ($b/\\sigma$)"); ax.set_ylabel("change in shipped utility\n(percentage points of optimum)")
    ax.legend(fontsize=6.5, frameon=False)
    fig.savefig(f"{OUT}/fig_bias.pdf", bbox_inches="tight")




def big():
    import worlds as W
    smp = W.big_samples(); V = np.load("data/twin_big.npy")
    U = np.array([s.mean() for s in smp]); us = U.max()
    P = json.load(open("results/big_policies.json"))
    A = json.load(open("results/big_allocation.json"))
    fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.6), gridspec_kw={"width_ratios": [1, 1.5, 1]})
    ax[0].scatter(V, U, s=3, alpha=0.5, color="#1f4e79"); lim = [min(U.min(), V.min()), max(U.max(), V.max())]
    ax[0].plot(lim, lim, color="#999", ls="--", lw=0.8); ax[0].set_xlabel("twin-predicted utility (0–1)"); ax[0].set_ylabel("measured utility $U$ (0–1)")
    ax[0].set_title("(a) twin vs GÉANT testbed", fontsize=8, loc="left")
    names = [("twin_only", "twin only"), ("twin_validate", "sim.-then-valid."), ("random", "random"), ("hill", "hill-climb."),
             ("measure_once", "measure-once"), ("eps", "ε-greedy"), ("thompson", "Thompson"), ("ucb", "UCB"),
             ("ucb_screen", "UCB pre-scr."), ("loop_blind", "loop, no pre-scr."), ("loop_twinstart", "twin-seeded loop"), ("loop", "loop")]
    vals, errs = [], []
    for k, _ in names:
        best = max(("mid", "high"), key=lambda l: np.mean(P[l]["300"][k]["shipped"]))
        v = np.array(P[best]["300"][k]["shipped"]) * 100
        vals.append(v.mean()); errs.append(1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0)
    cols = ["#1f4e79" if k == "loop" else ("#b8860b" if k.startswith("twin") else "#9aaabb") for k, _ in names]
    ax[1].barh(range(len(names)), vals, xerr=errs, color=cols, height=0.7)
    ax[1].set_yticks(range(len(names))); ax[1].set_yticklabels([n for _, n in names], fontsize=6.5)
    ax[1].set_xlim(min(vals) - 3, 100); ax[1].set_xlabel("shipped after 300 tests (% of optimum)"); ax[1].set_ylabel("policy")
    ax[1].set_title("(b) policies (replay)", fontsize=8, loc="left")
    nn = sorted(A, key=float); m = [100 * A[n][0] for n in nn]; h = [100 * A[n][1] for n in nn]
    ax[2].errorbar([float(n) for n in nn], m, yerr=h, marker="o", ms=3.5, capsize=2, color="#1f4e79")
    ax[2].set_xlabel("new configurations per 5 tests"); ax[2].set_ylabel("shipped utility (% of optimum)")
    ax[2].set_title("(c) allocation", fontsize=8, loc="left")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_geant_results.pdf", bbox_inches="tight")




def extra():
    import worlds as W
    # (1) multi-criteria structure and (2) heteroscedastic noise, both testbeds
    recs = {"small": [json.loads(l) for l in open("data/bank.jsonl")], "geant": [json.loads(l) for l in open("data/big_bank.jsonl")]}
    fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.5))
    for k, (name, col) in enumerate([("small", "#1f4e79"), ("geant", "#c0504d")]):
        by = {}
        for d in recs[name]:
            by.setdefault(d["x"], []).append(d)
        T = np.array([np.mean([d["T"] for d in v]) for v in by.values()]); L = np.array([np.mean([d["L"] for d in v]) for v in by.values()])
        E = np.array([np.mean([d["E"] for d in v]) for v in by.values()]); U = np.array([np.mean([d["u"] for d in v]) for v in by.values()])
        sd = np.array([np.std([d["u"] for d in v], ddof=1) for v in by.values()])
        if name == "small":
            sc = ax[0].scatter(T, L, c=E, s=6, cmap="viridis", alpha=0.8)
            ax[0].set_xlabel("throughput satisfaction $T$ (0–1)"); ax[0].set_ylabel("delay score $L$ (0–1)")
            ax[0].set_title("(a) criteria, small testbed", fontsize=8, loc="left")
            cb = fig.colorbar(sc, ax=ax[0], fraction=0.05, pad=0.02); cb.set_label("energy score $E$ (0–1)", fontsize=7)
        ax[1].scatter(100 * U / U.max(), sd, s=4, alpha=0.45, color=col, label="small testbed" if name == "small" else "GÉANT testbed")
    ax[1].set_xlabel("mean utility (% of optimum)"); ax[1].set_ylabel("sd of one test (utility, 0–1)"); ax[1].legend(fontsize=6.5, frameon=False, markerscale=2)
    ax[1].set_title("(b) noise per configuration", fontsize=8, loc="left")
    # (3) main effect of each knob
    names_s = ["F1alt", "F2alt", "F3alt", "F4alt", "F5alt", "PRalt", "meterF1", "meterF2", "sleepS4", "sleepS23"]
    for name, col, off in [("small", "#1f4e79", -0.2), ("geant", "#c0504d", 0.2)]:
        smp = W.testbed_samples() if name == "small" else W.big_samples()
        U = np.array([s.mean() for s in smp]); us = U.max(); c = np.arange(U.size)
        eff = [100 * (U[(c >> i) & 1 == 1].mean() - U[(c >> i) & 1 == 0].mean()) / us for i in range(10)]
        ax[2].barh(np.arange(10) + off, eff, height=0.4, color=col, label="small" if name == "small" else "GÉANT")
    ax[2].set_yticks(range(10)); ax[2].set_yticklabels(["reroute 1", "reroute 2", "reroute 3", "reroute 4", "reroute 5", "probe reroute", "meter 1", "meter 2", "sleep group 1", "sleep group 2"], fontsize=6)
    ax[2].invert_yaxis(); ax[2].axvline(0, color="#999", lw=0.8); ax[2].set_xlabel("main effect (percentage points of $U^*$)"); ax[2].set_ylabel("knob")
    ax[2].legend(fontsize=6.5, frameon=False); ax[2].set_title("(c) effect of each knob", fontsize=8, loc="left")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_structure.pdf", bbox_inches="tight")


def curse():
    A = json.load(open("results/policies_testbed.json")); B = json.load(open("results/big_policies.json"))["high"]
    pols = [("measure_once", "measure-once"), ("random", "random"), ("hill", "hill-climbing"), ("ucb", "UCB"),
            ("ucb_screen", "UCB pre-scr."), ("loop", "loop")]
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.5))
    for k, (P, t) in enumerate([(A, "(a) small testbed"), (B, "(b) GÉANT testbed")]):
        for j, (p, n) in enumerate(pols):
            ov = [100 * (np.nanmean(P[b][p]["belief"]) - np.mean(P[b][p]["shipped"])) for b in ("100", "200", "300")]
            ax[k].plot([100, 200, 300], ov, marker="o", ms=3.5, lw=1.3, label=n,
                       color=["#7f7f7f", "#bbbbbb", "#8064a2", "#2e8b57", "#9acd32", "#1f4e79"][j])
        ax[k].axhline(0, color="#999", lw=0.8); ax[k].set_xlabel("number of tests"); ax[k].set_title(t, fontsize=8, loc="left")
        ax[k].set_xticks([100, 200, 300])
    ax[0].set_ylabel("belief minus true utility\n(percentage points of $U^*$)"); ax[1].set_ylabel("belief minus true utility\n(percentage points of $U^*$)"); ax[1].legend(fontsize=6.3, frameon=False, ncol=2)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_curse.pdf", bbox_inches="tight")


def converge():
    d = json.load(open("results/dynamics.json"))["horizon"]; g = json.load(open("results/big_dynamics.json"))
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    hs = sorted(d, key=float); ax.plot([5 * float(h) for h in hs], [100 * d[h]["inc_lo_r2"] for h in hs], marker="o", color="#1f4e79", label="incumbent, small testbed")
    ax.plot([5 * float(h) for h in hs], [100 * d[h]["found_lo_r2"] for h in hs], marker="s", ls="--", color="#1f4e79", label="best tested, small testbed")
    hg = sorted(g, key=float); ax.plot([5 * float(h) for h in hg], [100 * g[h]["inc_lo_r2"] for h in hg], marker="o", color="#c0504d", label="incumbent, GÉANT testbed")
    ax.set_xscale("log"); ax.set_xlabel("number of tests (log scale)"); ax.set_ylabel("runs at a two-knob local optimum (%)"); ax.set_ylim(0, 102)
    ax.legend(fontsize=6.5, frameon=False, loc="lower right")
    fig.savefig(f"{OUT}/fig_converge.pdf", bbox_inches="tight")




def ctrlcost():
    d = json.load(open("results/big_ctrl_cost.json"))
    fig, ax = plt.subplots(figsize=(4.8, 2.6))
    bins = np.arange(-0.5, 16.5, 1)
    for k, c, n in [("random_order", "#bbbbbb", "random order (bank)"), ("measure_once", "#7f7f7f", "measure-once"),
                    ("loop", "#1f4e79", "loop"), ("ucb_screen", "#2e8b57", "UCB, pre-screened")]:
        v = np.array(d[k]["switches"]); h, _ = np.histogram(v, bins=bins)
        ax.step(np.arange(0, 16), 100 * h / h.sum(), where="mid", color=c, lw=1.5, label=f"{n} (mean {v.mean():.1f})")
    ax.set_xlabel("number of switches whose tables change between consecutive tests (of 40)"); ax.set_ylabel("share of tests (%)")
    ax.legend(fontsize=6.5, frameon=False)
    fig.savefig(f"{OUT}/fig_ctrlcost.pdf", bbox_inches="tight")


if __name__ == "__main__":
    for f in sys.argv[1:]:
        globals()[f](); print(f, "ok")
