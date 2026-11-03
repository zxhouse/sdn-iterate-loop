import os as _os
_ROOT = _os.path.dirname(_os.path.abspath(__file__))
FIGDIR = _os.environ.get("FIGDIR", _os.path.join(_ROOT, "figures"))
PNGDIR = _os.path.join(FIGDIR, "png")
_os.makedirs(PNGDIR, exist_ok=True)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import icons as I
import testbed as tb
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5})
pos = {"s1": (1.6, 2.0), "s2": (4.4, 3.4), "s3": (4.4, 2.0), "s4": (2.9, 0.4), "s6": (5.9, 0.4), "s5": (7.2, 2.0)}
fig, ax = plt.subplots(figsize=(7.2, 4.4)); ax.axis("off"); ax.set_aspect("equal")
sleep_links = set(tb.SLEEP["sleepS4"]) | set(tb.SLEEP["sleepS23"])
for (a, b), c in tb.LINKS.items():
    sl = (a, b) in sleep_links
    I.link(ax, pos[a], pos[b], lw=0.8 + c / 3.5, color="#c98a2b" if sl else "#7a8ea3", ls=(0, (4, 2)) if sl else "-")
    mx, my = (pos[a][0] + pos[b][0]) / 2, (pos[a][1] + pos[b][1]) / 2
    off = (0.42, 0.0) if a == "s2" and b == "s3" else (0.0, 0.13)
    ax.text(mx + off[0], my + off[1], f"{c} Mb/s", ha="center", fontsize=6.3, color="#333", zorder=3,
            bbox=dict(fc="white", ec="none", pad=0.4))
# controller and OpenFlow control channels
cpos = (4.4, 5.0)
I.controller(ax, *cpos, s=0.9)
for s, (x, y) in pos.items():
    ax.plot([cpos[0], x], [cpos[1] - 0.25, y + 0.18], color="#2e8b57", lw=0.5, ls=":", zorder=0.5)
ax.text(5.35, 4.75, "OpenFlow 1.3\n(flow + meter tables, bundles)", fontsize=6.2, color="#2e8b57")
for s, (x, y) in pos.items():
    I.switch(ax, x, y, s=0.62, label=s, fs=7)
# sources on s1
src = [("F1", 4.5), ("F2", 4.0), ("F3", 3.0), ("F5", 2.5), ("probe", None), ("B1", None), ("B2", None)]
for k, (n, d) in enumerate(src):
    x, y = 0.1 + (k % 2) * 0.62, 3.3 - (k // 2) * 0.72
    I.host(ax, x, y, s=0.62, fc="#ffe9c7" if n.startswith("B") else ("#dff0e0" if n == "probe" else "#e8eef5"))
    ax.text(x, y - 0.24, n + (f" {d}" if d else ""), ha="center", va="top", fontsize=5.6)
    I.link(ax, (x + 0.15, y), pos["s1"], lw=0.4, color="#aaa", z=0.5)
# F4 source on s2
I.host(ax, 3.0, 3.85, s=0.62); ax.text(3.0, 3.61, "F4 3.5", ha="center", va="top", fontsize=5.6)
I.link(ax, (3.15, 3.85), pos["s2"], lw=0.4, color="#aaa", z=0.5)
# sinks on s5 (servers)
for k, n in enumerate(["F1", "F2", "F3", "F4", "F5", "probe", "B1", "B2"]):
    x, y = 8.35 + (k % 2) * 0.5, 3.15 - (k // 2) * 0.66
    I.server(ax, x, y, s=0.62); ax.text(x, y - 0.2, n, ha="center", va="top", fontsize=5.6)
    I.link(ax, (x - 0.1, y), pos["s5"], lw=0.4, color="#aaa", z=0.5)
ax.text(0.41, 3.75, "sources", fontsize=6.5, ha="center", color="#444", weight="bold")
ax.text(8.6, 3.55, "sinks", fontsize=6.5, ha="center", color="#444", weight="bold")
# legend
ax.plot([0.0, 0.5], [-0.35, -0.35], color="#7a8ea3", lw=2); ax.text(0.6, -0.35, "link (TBF-shaped veth, 50 ms queue)", va="center", fontsize=6.3)
ax.plot([3.9, 4.4], [-0.35, -0.35], color="#c98a2b", lw=1.5, ls=(0, (4, 2))); ax.text(4.5, -0.35, "link that a sleep knob can switch off", va="center", fontsize=6.3)
ax.text(0.0, -0.72, "host colours: blue = elastic TCP demand (rate in Mb/s), green = latency probe, orange = uncontrolled UDP cross-traffic",
        fontsize=6.0, color="#444")
ax.set_xlim(-0.35, 9.2); ax.set_ylim(-0.95, 5.6)
fig.savefig(_os.path.join(FIGDIR, "fig2_testbed.pdf"), bbox_inches="tight")
fig.savefig(_os.path.join(PNGDIR, "fig2.png"), dpi=110, bbox_inches="tight")
print("ok")
