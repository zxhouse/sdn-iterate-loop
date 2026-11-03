"""GEANT testbed schematic and Topology Zoo panel with switch icons."""
import os as _os
_ROOT = _os.path.dirname(_os.path.abspath(__file__))
FIGDIR = _os.environ.get("FIGDIR", _os.path.join(_ROOT, "figures"))
PNGDIR = _os.path.join(FIGDIR, "png")
_os.makedirs(PNGDIR, exist_ok=True)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, networkx as nx, json, math, sys
import icons as I
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7})
SCR = PNGDIR
CAPITAL = {"IS": (64.1, -21.9), "IE": (53.3, -6.3), "UK": (51.5, -0.1), "NO": (59.9, 10.8), "SE": (59.3, 18.1),
           "FI": (60.2, 24.9), "EE": (59.4, 24.8), "LV": (56.9, 24.1), "LT": (54.7, 25.3), "RU": (55.8, 37.6),
           "DK": (55.7, 12.6), "PL": (52.2, 21.0), "NL": (52.4, 4.9), "BE": (50.6, 3.6), "LU": (49.4, 6.6),
           "FR": (48.9, 2.4), "DE": (50.1, 8.7), "CZ": (50.1, 14.4), "CH": (46.2, 6.1), "IT": (45.5, 9.2),
           "ES": (40.4, -3.7), "PT": (38.7, -9.1), "SK": (49.2, 18.9), "HU": (47.0, 20.3), "AT": (47.8, 14.8),
           "SL": (45.9, 13.4), "HR": (45.0, 16.4), "RS": (44.8, 20.5), "ME": (42.0, 18.6), "MK": (41.4, 21.9),
           "BG": (42.7, 23.3), "RO": (44.4, 26.1), "TR": (39.9, 32.9), "GR": (38.0, 23.7), "CY": (35.2, 33.4),
           "IL": (32.1, 34.8), "MT": (35.9, 14.5), "BY": (53.9, 27.6), "UA": (50.5, 30.5), "MD": (47.0, 28.9)}


def geo_pos(g, labels=None):
    pos, miss = {}, []
    for n in g.nodes:
        a = g.nodes[n]
        if "Latitude" in a and "Longitude" in a:
            pos[n] = (float(a["Longitude"]), float(a["Latitude"]))
        elif labels and labels.get(n) in CAPITAL:
            la, lo = CAPITAL[labels[n]]; pos[n] = (lo, la)
        else:
            miss.append(n)
    if miss:
        fixed = list(pos) or None
        init = {n: pos.get(n, (0, 0)) for n in g.nodes}
        pos = nx.spring_layout(g, pos=init, fixed=fixed, seed=1, k=1.0 / math.sqrt(len(g))) if fixed else nx.spring_layout(g, seed=1)
    lat0 = np.mean([p[1] for p in pos.values()])
    return {n: (p[0] * math.cos(math.radians(lat0)), p[1]) for n, p in pos.items()}


def geant():
    import bigbed as B
    d = json.load(open("data/big_describe.json"))
    G = B.G; lab = {n: d["labels"][str(n)] for n in G.nodes}
    pos = geo_pos(G, lab)
    fig, ax = plt.subplots(figsize=(7.4, 7.0)); ax.axis("off"); ax.set_aspect("equal")
    hot = {tuple(sorted(v)) for v in d["bg"].values()}
    sleep = {tuple(sorted(e)) for v in d["sleep"].values() for e in v}
    alt = set()
    for j, p in d["rr"]:
        for k in range(len(p) - 1): alt.add(tuple(sorted((p[k], p[k + 1]))))
    for u, v in G.edges:
        e = tuple(sorted((u, v)))
        if e in sleep:
            I.link(ax, pos[u], pos[v], lw=1.3, color="#c98a2b", ls=(0, (4, 2)))
        elif e in hot:
            I.link(ax, pos[u], pos[v], lw=3.0, color="#c0504d")
        else:
            I.link(ax, pos[u], pos[v], lw=1.0, color="#9aaabb")
        if e in alt and e not in sleep:
            I.link(ax, pos[u], pos[v], lw=0.6, color="#2e8b57", ls=(0, (1, 1.5)), z=2)
    ends = {}
    for j, (s, t, r) in enumerate(d["demands"]):
        ends.setdefault(s, []).append(f"D{j}s"); ends.setdefault(t, []).append(f"D{j}d")
    for n in G.nodes:
        x, y = pos[n]
        I.switch(ax, x, y, s=1.15, ms=3.2)
        ax.text(x, y + 0.42, lab[n], ha="center", va="bottom", fontsize=5.6, color="#222", zorder=8)
    # hosts at PoPs with demand endpoints, probe and cross-traffic
    for n, l in ends.items():
        x, y = pos[n]
        I.host(ax, x + 0.95, y - 0.5, s=1.4)
    for n in d["probe"][:2]:
        x, y = pos[n]; I.host(ax, x - 0.95, y - 0.5, s=1.4, fc="#dff0e0")
    for k, (u, v) in d["bg"].items():
        x, y = pos[u]; I.host(ax, x - 0.95, y + 0.6, s=1.4, fc="#ffe9c7")
    xs = [p[0] for p in pos.values()]; ys = [p[1] for p in pos.values()]
    cx, cy = min(xs) + 1.5, min(ys) + 2.0
    I.controller(ax, cx, cy, s=2.2, label="SDN controller\n(OpenFlow 1.3 to all 40 switches)", fs=6.3)
    # legend
    lx, ly = min(xs) + 6.0, min(ys) - 1.6
    items = [("#9aaabb", "-", 1.2, "backbone link (TBF, 4.9 Mb/s)"), ("#c0504d", "-", 3.0, "link with uncontrolled cross-traffic"),
             ("#c98a2b", (0, (4, 2)), 1.3, "link in a sleep group"), ("#2e8b57", (0, (1, 1.5)), 1.0, "alternative path of a reroute knob")]
    for i, (c, ls, lw, t) in enumerate(items):
        xx, yy = lx + (i % 2) * 14, ly - (i // 2) * 1.1
        ax.plot([xx, xx + 1.6], [yy] * 2, color=c, ls=ls, lw=lw); ax.text(xx + 2.0, yy, t, va="center", fontsize=6.2)
    for i, (fc, t) in enumerate([("#e8eef5", "demand source/sink (iperf3)"), ("#dff0e0", "latency probe end"), ("#ffe9c7", "cross-traffic source")]):
        xx, yy = lx + i * 12.5, ly - 2.4
        I.host(ax, xx + 0.8, yy + 0.1, s=1.4, fc=fc); ax.text(xx + 2.0, yy, t, va="center", fontsize=6.2)
    fig.savefig(_os.path.join(FIGDIR, "fig_geant_testbed.pdf"), bbox_inches="tight")
    fig.savefig(f"{SCR}/geant.png", dpi=110, bbox_inches="tight"); print("geant ok")


def zoo():
    names = [("Abilene", "Abilene"), ("Nsfnet", "NSFNET"), ("Janetbackbone", "Janet backbone"), ("PionierL3", "PIONIER L3"),
             ("Geant2012", "GÉANT 2012"), ("Renater2010", "RENATER 2010"), ("Cesnet201006", "CESNET 2010"), ("Garr201201", "GARR 2012")]
    fig, axs = plt.subplots(2, 4, figsize=(7.4, 4.4))
    for ax, (f, t) in zip(axs.flat, names):
        g = nx.Graph(nx.read_graphml(_os.path.join(_ROOT, "zoo", f"{f}.graphml"))); g.remove_edges_from(nx.selfloop_edges(g))
        g = g.subgraph(max(nx.connected_components(g), key=len)).copy()
        lab = {n: g.nodes[n].get("label") for n in g.nodes}
        pos = nx.kamada_kawai_layout(g)
        xs = np.array([p[0] for p in pos.values()]); ys = np.array([p[1] for p in pos.values()])
        span = max(xs.max() - xs.min(), ys.max() - ys.min()) or 1
        for u, v in g.edges:
            I.link(ax, pos[u], pos[v], lw=0.6, color="#9aaabb")
        s = span / (11.0 if g.number_of_nodes() < 30 else 15.0)
        for n in g.nodes:
            I.switch(ax, *pos[n], s=s, ms=1.6)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(f"{t}\n{g.number_of_nodes()} switches, {g.number_of_edges()} links", fontsize=7)
        ax.set_xlim(xs.min() - s, xs.max() + s); ax.set_ylim(ys.min() - s, ys.max() + s)
    fig.tight_layout()
    fig.savefig(_os.path.join(FIGDIR, "fig_zoo_topologies.pdf"), bbox_inches="tight")
    fig.savefig(f"{SCR}/zoo.png", dpi=110, bbox_inches="tight"); print("zoo ok")


if __name__ == "__main__":
    for a in sys.argv[1:]:
        globals()[a]()
