"""Schematics of the experimental process: test timeline, pipeline, and a live run trace."""
import os as _os
_ROOT = _os.path.dirname(_os.path.abspath(__file__))
FIGDIR = _os.environ.get("FIGDIR", _os.path.join(_ROOT, "figures"))
PNGDIR = _os.path.join(FIGDIR, "png")
_os.makedirs(PNGDIR, exist_ok=True)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, json, sys
from matplotlib.patches import FancyBboxPatch
import icons as I
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5})
SCR = PNGDIR
OUT = FIGDIR


def timeline():
    fig, ax = plt.subplots(figsize=(7.2, 2.6)); ax.set_xlim(-0.22, 2.9); ax.set_ylim(-0.9, 5.4); ax.axis("off")
    rows = [("controller: push changed tables (bundles)", 0.0, 0.08, "#2e8b57"),
            ("settle", 0.08, 0.38, "#bbbbbb"),
            ("TCP demands (iperf3, 1 s)", 0.38, 1.38, "#1f4e79"),
            ("UDP cross-traffic (1 s)", 0.38, 1.38, "#c98a2b"),
            ("probe: ICMP every 100 ms (1.5 s)", 0.38, 1.88, "#2e8b57"),
            ("collect rates and RTT, compute U", 1.88, 2.0, "#8064a2")]
    for i, (t, a, b, c) in enumerate(rows):
        y = 4.6 - i * 0.85
        ax.add_patch(FancyBboxPatch((a, y - 0.27), b - a, 0.54, boxstyle="round,pad=0,rounding_size=0.03", fc=c, ec="none", alpha=0.85))
        tx = b + 0.02 if (b - a) < 0.5 else a + 0.02
        ax.text(tx, y, t, va="center", fontsize=7.6, color="black" if (b - a) < 0.5 else "white")
    for t in np.arange(0, 2.01, 0.5):
        ax.plot([t, t], [-0.05, 0.05], color="#555", lw=0.8); ax.text(t, -0.18, f"{t:.1f} s", ha="center", va="top", fontsize=6.5)
    ax.plot([0, 2.0], [0, 0], color="#555", lw=0.8)
    ax.text(1.0, -0.62, "time from the start of the test (s)", ha="center", va="top", fontsize=7)
    ax.text(-0.13, 2.5, "activity (one row per process)", rotation=90, ha="center", va="center", fontsize=7)
    fig.savefig(f"{OUT}/fig_test_timeline.pdf", bbox_inches="tight"); fig.savefig(f"{SCR}/timeline.png", dpi=110, bbox_inches="tight")


def pipeline():
    fig, ax = plt.subplots(figsize=(7.4, 3.6)); ax.set_xlim(0, 100); ax.set_ylim(0, 50); ax.axis("off"); ax.set_aspect("equal")
    def box(x, y, w, h, t, fc="#eef3f8", ec="#2b4c6f", ty=None):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", fc=fc, ec=ec, lw=0.9))
        ax.text(x + w / 2, ty if ty else y + h / 2, t, ha="center", va="center", fontsize=6.7)
    def arr(p, q, c="#2b4c6f"):
        ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-|>", color=c, lw=1.0, mutation_scale=9, shrinkA=0, shrinkB=0))
    # sources of worlds
    box(1, 34, 24, 14, "Small testbed\n6 switches, 10 knobs\n1024 configs x 5 tests", ty=44)
    for k in range(3): I.switch(ax, 7 + k * 6, 37.0, s=4.2, ms=3)
    box(1, 17, 24, 14, "GÉANT testbed\n40 switches, 10 knobs\n1024 configs x 4 tests", ty=27)
    for k in range(3): I.switch(ax, 7 + k * 6, 20.0, s=4.2, ms=3)
    box(1, 0, 24, 14, "Topology Zoo landscapes\n8 backbones x 3 matrices\n16,384 configs x 32 draws", ty=10)
    I.server(ax, 13, 3.6, s=6, fc="#b8860b")
    # banks
    box(32, 22, 22, 22, "Measurement banks\n(ground truth: mean of\nthe measurements)\n\nreplay: a test draws\none measured value", fc="#fff8e8", ec="#9a6b16")
    box(32, 0, 22, 14, "Fluid model\n(twin, pre-screening)", fc="#fff8e8", ec="#9a6b16")
    # policies and outputs
    box(61, 22, 18, 22, "12 policies\nloop, bandits,\nlocal search,\ntwin-based\n(300 tests)")
    box(61, 0, 18, 14, "Live closed loop\n(every test deployed\nand measured)")
    box(85, 22, 14, 22, "Scores vs\noptimum:\nshipped,\nfound,\nbelief", fc="#e9f4ec", ec="#2f6b3c")
    box(85, 0, 14, 14, "Live vs\nreplay", fc="#e9f4ec", ec="#2f6b3c")
    arr((25, 41), (32, 38)); arr((25, 24), (32, 28)); arr((25, 7), (32, 24)); arr((25, 7), (32, 7))
    arr((54, 11), (61, 26)); arr((54, 33), (61, 33)); arr((54, 7), (61, 7)); arr((79, 33), (85, 33)); arr((79, 7), (85, 7))
    arr((70, 14), (70, 22), c="#999")
    ax.text(71, 17.5, "same policy code", fontsize=6, color="#666")
    fig.savefig(f"{OUT}/fig_pipeline.pdf", bbox_inches="tight"); fig.savefig(f"{SCR}/pipeline.png", dpi=110, bbox_inches="tight")


def livetrace(tag="loop#0", bank="small"):
    import worlds as W, sdnloop as S
    if bank == "small":
        w = W.testbed_world(); T = [json.loads(l) for l in open("data/live_tests.jsonl")]
        names = ["F1alt", "F2alt", "F3alt", "F4alt", "F5alt", "PRalt", "meterF1", "meterF2", "sleepS4", "sleepS23"]
    T = [t for t in T if t["tag"] == tag]
    U = w.U; us = w.ustar
    bel = {}; seen = set(); inc_u, inc_c, newflag, ys = [], [], [], []
    for t in T:
        c = W.code_of(t["x"]); newflag.append(c not in seen); seen.add(c)
        if c not in bel: bel[c] = S.Belief(w.sigma)
        bel[c].update(t["u"]); ys.append(t["u"])
        L = max(bel, key=lambda q: bel[q].mu); inc_c.append(L); inc_u.append(U[L])
    n = len(T); x = np.arange(1, n + 1)
    fig, ax = plt.subplots(2, 1, figsize=(7.2, 4.6), gridspec_kw={"height_ratios": [2.2, 1.3]}, sharex=True)
    nf = np.array(newflag)
    ax[0].scatter(x[nf], 100 * np.array(ys)[nf] / us, s=12, marker="o", color="#c0504d", label="test of a new configuration", zorder=3)
    ax[0].scatter(x[~nf], 100 * np.array(ys)[~nf] / us, s=9, marker="s", color="#7a8ea3", label="re-test", zorder=3)
    ax[0].step(x, 100 * np.array(inc_u) / us, where="post", color="#1f4e79", lw=1.6, label="true utility of the incumbent")
    ax[0].axhline(100, color="#999", ls="--", lw=0.8)
    ax[0].set_ylabel("measured / true utility\n(% of optimum)"); ax[0].tick_params(labelbottom=True); ax[0].set_xlabel("test number (live testbed)", fontsize=7); ax[0].legend(fontsize=6.3, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    M = np.array([[(c >> i) & 1 for i in range(10)] for c in inc_c]).T
    ax[1].imshow(M, aspect="auto", cmap="Blues", vmin=-0.3, vmax=1.2, extent=(0.5, n + 0.5, 9.5, -0.5), interpolation="nearest")
    ax[1].set_yticks(range(10)); ax[1].set_yticklabels(names, fontsize=6); ax[1].set_xlabel("test number (live testbed)"); ax[1].set_ylabel("knob")
    ax[1].set_title("knobs of the incumbent configuration (dark = on)", fontsize=7, loc="left")
    ax[1].grid(False)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_live_trace.pdf", bbox_inches="tight"); fig.savefig(f"{SCR}/trace.png", dpi=110, bbox_inches="tight")
    print("final incumbent", inc_c[-1], round(100 * inc_u[-1] / us, 1), "new", int(nf.sum()), "of", n)


if __name__ == "__main__":
    for a in sys.argv[1:]:
        globals()[a]()
