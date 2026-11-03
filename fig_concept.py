import os as _os
_ROOT = _os.path.dirname(_os.path.abspath(__file__))
FIGDIR = _os.environ.get("FIGDIR", _os.path.join(_ROOT, "figures"))
PNGDIR = _os.path.join(FIGDIR, "png")
_os.makedirs(PNGDIR, exist_ok=True)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import icons as I
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.6})
fig, ax = plt.subplots(figsize=(7.4, 4.3)); ax.set_aspect("equal"); ax.set_xlim(0, 100); ax.set_ylim(0, 58); ax.axis("off")
C, F = "#2b4c6f", "#9a6b16"


def box(x, y, w, h, t, fc="#eef3f8", ec=C, ty=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.5", fc=fc, ec=ec, lw=1.0))
    ax.text(x + w / 2, ty if ty is not None else y + h / 2, t, ha="center", va="center", fontsize=7.4)


def arrow(pts, c=C, ls="-"):
    xs, ys = zip(*pts)
    ax.plot(xs[:-1] + (xs[-1],), ys[:-1] + (ys[-1],), color=c, lw=1.1, ls=ls, solid_capstyle="butt")
    ax.annotate("", xy=pts[-1], xytext=pts[-2], arrowprops=dict(arrowstyle="-|>", color=c, lw=1.1,
                                                              mutation_scale=10, shrinkA=0, shrinkB=0))


# frame
box(2, 49, 96, 7, "Frame = intent: configuration knobs $F$ and utility $U$ (weights of throughput, delay, energy)", fc="#fbf3e4", ec=F)
# row 1
W, H, Y = 20, 15, 26
xs = [2, 27.3, 52.6, 78]
box(xs[0], Y, W, H, "Ideation\nvariants within $r$\nknobs of a leader", ty=Y + 9.5)
box(xs[1], Y, W, H, "Pre-screening\nnetwork model /\ndigital twin", ty=Y + 9.5)
box(xs[2], Y, W, H, "Prototype\ncompile and install\n(OpenFlow bundles)", ty=Y + 9.5)
box(xs[3], Y, W, H, "Test\nmeasurement epoch\n$y=U(p)+b+\\varepsilon$", ty=Y + 9.5)
I.server(ax, xs[1] + W / 2, Y + 3.4, s=8, fc="#b8860b")
I.switch(ax, xs[2] + W / 2 - 3.5, Y + 3.0, s=6.0, ms=5)
I.switch(ax, xs[2] + W / 2 + 3.5, Y + 3.0, s=6.0, ms=5)
I.host(ax, xs[3] + W / 2 - 3, Y + 3.2, s=6.5)
I.server(ax, xs[3] + W / 2 + 3.5, Y + 3.4, s=8)
for a, b in zip(xs[:-1], xs[1:]):
    arrow([(a + W, Y + H / 2), (b, Y + H / 2)])
# row 2
box(44, 6, 44, 12, "Beliefs $M_k^p$ and archive $P_k$ (retention)\nincumbent $p^*_k=\\arg\\max_p M_k^p$;\nre-tests: three leaders + one in rotation")
box(2, 6, 30, 12, "Commit: incumbent or\nlower-confidence choice;\nits belief $X_k$ is optimistic", fc="#e9f4ec", ec="#2f6b3c")
arrow([(xs[3] + W / 2, Y), (xs[3] + W / 2, 18)])                 # test -> beliefs
arrow([(44, 12), (32, 12)])                                     # beliefs -> commit
arrow([(56, 18), (56, 22), (12, 22), (12, Y)])                   # beliefs -> ideation (next round)
ax.text(34, 23.2, "next round", fontsize=6.8, color=C, ha="center")
arrow([(50, 49), (50, 41)], c=F, ls="--")
ax.text(51.5, 45, "reframe = new intent or new knobs (outer loop)", fontsize=6.8, color=F, style="italic")
fig.savefig(_os.path.join(FIGDIR, "fig1_loop.pdf"), bbox_inches="tight")
fig.savefig(_os.path.join(PNGDIR, "fig1.png"), dpi=110, bbox_inches="tight")
print("ok")
