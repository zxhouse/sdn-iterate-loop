"""Graphical schematics of the course of each experiment (E1-E11)."""
import os as _os
_ROOT = _os.path.dirname(_os.path.abspath(__file__))
FIGDIR = _os.environ.get("FIGDIR", _os.path.join(_ROOT, "figures"))
PNGDIR = _os.path.join(FIGDIR, "png")
_os.makedirs(PNGDIR, exist_ok=True)
import sys, numpy as np
from schem import S, C, G, O, R, P, GR
import icons as I

rng = np.random.default_rng(3)
def rb(): return list((rng.random(10) < 0.45).astype(int))


def head(s, x, y, w, h, title, c=C):
    s.panel(x, y, w, h)
    s.title(x + w / 2, y + h - 1.6, title, c=c)


def loop_parts(s, x0, y0, marks=None):
    """The loop drawn as its four components; marks: {component: red label}."""
    comps = [("ideation", "#eef3f8", C), ("pre-screen", "#fff8e8", O), ("test", "#eef3f8", C), ("archive", "#e8f0e8", G)]
    pts = [(x0, y0 + 10), (x0 + 14, y0 + 10), (x0 + 14, y0), (x0, y0)]
    for (n, fc, ec), (x, y) in zip(comps, pts):
        s.ax.add_patch(__import__("matplotlib").patches.FancyBboxPatch((x - 5, y - 2), 10, 4, boxstyle="round,pad=0,rounding_size=0.6",
                                                                      fc=fc, ec=ec, lw=0.8, zorder=3))
        s.t(x, y, n, fs=5.8, inside=True)
    s.arrow((x0 + 5, y0 + 10), (x0 + 9, y0 + 10)); s.arrow((x0 + 14, y0 + 8), (x0 + 14, y0 + 2))
    s.arrow((x0 + 9, y0), (x0 + 5, y0)); s.arrow((x0, y0 + 2), (x0, y0 + 8))
    return pts


def e5():
    s = S(100, 36)
    variants = [("A2", "reach of ideation", (0, "1 / 3 / any knobs")), ("A3", "generativity", (1, "1 or 15 ideas, no screen")),
                ("A4", "repeated tests", (2, "all new / no rotation")), ("A5", "retention", (3, "keep 1 or 3"))]
    for k, (a, n, (ci, lab)) in enumerate(variants):
        x = 1 + k * 24.8
        head(s, x, 1, 23.5, 34, f"{a}: {n}", c=R)
        pts = loop_parts(s, x + 4.8, 12)
        px, py = pts[ci]
        s.ax.add_patch(__import__("matplotlib").patches.Circle((px, py), 3.6, fc="none", ec=R, lw=1.4, zorder=6))
        s.t(x + 11.75, 6.5, lab, fs=5.6, c=R)
        s.t(x + 11.75, 3.6, "all else as baseline", fs=5.4, c="#666")
    return s.save("exp_e5")


def e9():
    s = S(100, 40)
    replay(s, 1)
    head(s, 52, 1, 47, 38, "Exact check on the landscape", c=P)
    # a piece of the configuration graph (hypercube neighbourhood)
    nodes = {0: (63, 18), 1: (72, 27), 2: (72, 9), 3: (81, 18), 4: (90, 27), 5: (90, 9)}
    edges = [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (3, 5), (1, 4), (2, 5)]
    hw, hh = 3.6, 1.0
    def clip(a, b, m=0.4):
        v = np.array(nodes[b], float) - np.array(nodes[a], float); u = v / np.linalg.norm(v)
        t = min(hw / abs(u[0]) if abs(u[0]) > 1e-9 else 1e9, hh / abs(u[1]) if abs(u[1]) > 1e-9 else 1e9) + m
        return tuple(np.array(nodes[a]) + u * t), tuple(np.array(nodes[b]) - u * t)
    for a, b in edges:
        p, q = clip(a, b, 0.1); I.link(s.ax, p, q, lw=0.8, color=GR)
    import matplotlib.patches as mp
    for n, (x, y) in nodes.items():
        s.ax.add_patch(mp.FancyBboxPatch((x - hw, y - hh), 2 * hw, 2 * hh, boxstyle="round,pad=0,rounding_size=0.4",
                                         fc="white", ec=R if n == 3 else "#c9d3df", lw=0.7, zorder=2.6))
        s.config(x - 3.05, y - 0.43, rb()[:6], cell=0.85, gap=0.17, c=R if n == 3 else C)
    for a, b in ((0, 1), (1, 3)):
        p, q = clip(a, b, 0.6)
        s.arrow(p, q, c=G, lw=1.8, z=2.7)
    s.t(66.0, 24.2, "incumbent", fs=5.6, c=G, ha="right")
    s.t(85.4, 18.0, "local optimum", fs=5.8, c=R, ha="left")
    s.t(75.5, 5.4, "every neighbour of the final incumbent within", fs=5.6)
    s.t(75.5, 3.4, "1 or 2 knobs is checked (up to 5000 tests)", fs=5.6)
    s.arrow((50, 20), (52, 20))
    return s.save("exp_e9")


def e11():
    import json
    from fig_maps import geo_pos
    import bigbed as B
    d = json.load(open("data/big_describe.json")); lab = {n: d["labels"][str(n)] for n in B.G.nodes}
    pos = geo_pos(B.G, lab)
    xs = np.array([p[0] for p in pos.values()]); ys = np.array([p[1] for p in pos.values()])
    s = S(100, 44)
    head(s, 1, 1, 44, 42, "GÉANT testbed (40 switches)")
    P_ = {n: (4 + 38 * (pos[n][0] - xs.min()) / (xs.max() - xs.min()), 7 + 29 * (pos[n][1] - ys.min()) / (ys.max() - ys.min())) for n in B.G.nodes}
    changed = {16, 34, 22, 17, 23}
    Q = {n: np.array(p, float) for n, p in P_.items()}
    for it in range(400):
        moved = False
        for a in Q:
            for b in Q:
                if a < b:
                    d = Q[b] - Q[a]; dist = np.hypot(d[0] / 3.0, d[1] / 2.0)
                    if dist < 1.0:
                        u = d / (np.linalg.norm(d) + 1e-9) if np.linalg.norm(d) > 1e-6 else np.array([1.0, 0])
                        Q[a] -= u * 0.15; Q[b] += u * 0.15; moved = True
        for n in Q: Q[n] = np.clip(Q[n], [3.5, 6], [42.5, 37])
        if not moved: break
    P_ = {n: tuple(v) for n, v in Q.items()}
    for u, v in B.G.edges: I.link(s.ax, P_[u], P_[v], lw=0.5, color=GR)
    for n in B.G.nodes: I.switch(s.ax, *P_[n], s=1.6, ms=1.3, fc=R if n in changed else I.SW_FC)
    I.switch(s.ax, 6, 2.9, s=1.6, ms=1.3, fc=R)
    s.t(7.6, 2.9, "rewritten by one knob change", fs=5.4, ha="left")
    head(s, 47, 23, 24, 20, "Bank: 4 passes", c=O)
    s.grid(50, 26, 10, 4, cw=2.9, ch=1.1)
    s.t(56, 25.0, "", fs=5)
    head(s, 47, 1, 24, 20, "Live runs", c=G)
    s.loop(53.5, 10.5, r=2.8); s.t(53.5, 5.4, "3 policies", fs=5.6)
    s.config(58.8, 10, rb(), cell=0.9, gap=0.12)
    s.t(64.5, 7.8, "config sent", fs=5.6)
    head(s, 73, 1, 26, 42, "Analyses", c=P)
    D = data()
    X, Y = s.chart(81.5, 29, 15.5, 7, (0.55, 0.95), (0.55, 0.95), [0.6, 0.75, 0.9], [0.6, 0.9], "twin U (0–1)", "meas. U (0–1)", title="twin vs testbed")
    idx = np.random.default_rng(2).choice(len(D["Ub"]), 150, replace=False)
    s.ax.scatter(X(D["Vb"][idx]), Y(D["Ub"][idx]), s=0.8, color=C, zorder=4)
    vals = sorted(D["big"].values())
    X, Y = s.chart(81.5, 16.5, 15.5, 6.5, (-0.5, 11.5), (90, 100), [(0, "1"), (11, "12")], [90, 100], "policy (sorted)", "% of opt.", title="12 policies, 300 tests")
    for i, v in enumerate(vals):
        s.ax.add_patch(Rectangle_(float(X(i)) - 0.5, 16.5, 1.0, float(Y(v)) - 16.5, "#1f4e79"))
    cnt = np.bincount(D["cc"], minlength=14)[:14] / len(D["cc"]) * 100
    X, Y = s.chart(81.5, 4.2, 15.5, 6.5, (-0.5, 13.5), (0, 25), [0, 5, 10, 13], [0, 20], "switches changed per test", "% tests", title="control-plane cost (loop)")
    for i, v in enumerate(cnt):
        s.ax.add_patch(Rectangle_(float(X(i)) - 0.45, 4.2, 0.9, float(Y(v)) - 4.2, R))
    s.arrow((45, 33), (47, 33)); s.arrow((45, 11), (47, 11)); s.arrow((71, 33), (73, 33)); s.arrow((71, 11), (73, 11))
    return s.save("exp_e11")




def implementation():
    import matplotlib.patches as mp
    s = S(100, 64)
    s.ax.add_patch(mp.FancyBboxPatch((0.5, 0.5), 99, 63, boxstyle="round,pad=0,rounding_size=1.2", fc="white", ec="#444", lw=0.9, zorder=0))
    s.t(50, 61.6, "One virtual machine: 2 vCPU, 8 GB RAM, Linux 6.18", fs=7, w="bold")
    # user space
    s.ax.add_patch(mp.Rectangle((2, 31), 96, 28, fc="#eef3f8", ec="#9fb3c8", lw=0.6, ls="--", zorder=0))
    s.t(3, 57.6, "user space", fs=6, c=C, ha="left", style="italic")
    # controller
    s.panel(4, 33, 22, 23, fc="#e8f0e8", ec=G)
    s.t(15, 54.2, "controller (Python)", fs=6.2, w="bold", c=G)
    I.controller(s.ax, 9.5, 47.5, s=5.0, label=None)
    s.loop(19, 48.5, r=2.4)
    s.config(6.5, 39.8, rb()[:5], cell=0.9, gap=0.15, c=G)
    s.config(19.2, 39.8, rb()[:5], cell=0.9, gap=0.15)
    s.arrow((19.6, 45.4), (20.8, 41.6), c=G)
    s.t(15, 35.8, "beliefs, next p, tables", fs=5.2)
    # ovs-vswitchd
    s.panel(32, 33, 40, 23, fc="white", ec=C)
    s.t(52, 54.2, "ovs-vswitchd 3.3, userspace datapath", fs=6.2, w="bold", c=C)
    for r in range(4):
        for k in range(10):
            I.switch(s.ax, 35 + k * 3.55, 50 - r * 3.0, s=2.3, ms=1.6, fc=R if (r, k) in ((1, 3), (2, 7)) else I.SW_FC)
    # one flow table zoomed
    s.grid(62, 34.4, 3, 4, cw=2.4, ch=0.9, fc="#eef3f8", ec=C)
    s.path([(45.3, 46.2), (45.3, 38.9), (61.6, 38.9)], c="#999", lw=0.6)
    s.t(47, 36.8, "tables of g13", fs=5.2, ha="left")
    # ovsdb + traffic tools
    s.panel(75, 46, 21, 10, fc="white", ec="#9fb3c8")
    s.db(80, 50.2, w=4.4, h=3.4)
    s.t(88.5, 52.2, "ovsdb-server", fs=5.8, w="bold")
    s.t(88.5, 49.6, "bridges, ports", fs=5.2)
    s.panel(75, 33, 21, 11, fc="white", ec="#9fb3c8")
    I.host(s.ax, 80, 39.6, s=4.0); I.server(s.ax, 86, 39.6, s=3.6)
    s.arrow((82.2, 39.6), (84.8, 39.6), c=C, ms=6)
    s.t(85.5, 35.0, "iperf3 client / server, ping", fs=5.2)
    s.t(90.8, 39.6, "x32", fs=5.8, w="bold", ha="left")
    # control arrows
    s.arrow((26, 50), (32, 50), c=G, lw=1.2)
    s.arrow((26, 41), (32, 41), c="#666", lw=1.0)
    s.t(29, 52.0, "bundle", fs=5.0, c=G)
    s.t(29, 43.0, "vsctl", fs=5.0, c="#666")
    # kernel
    s.ax.add_patch(mp.Rectangle((2, 2), 96, 27, fc="#fbf6ec", ec="#c9a96a", lw=0.6, ls="--", zorder=0))
    s.t(3, 27.4, "kernel", fs=6, c=O, ha="left", style="italic")
    s.panel(4, 4, 40, 21, fc="white", ec=O)
    s.t(24, 22.8, "61 inter-switch links", fs=6.2, w="bold", c=O)
    for k in range(3):
        y = 18.2 - k * 5.0
        I.switch(s.ax, 8, y, s=2.4, ms=1.6); I.switch(s.ax, 40, y, s=2.4, ms=1.6)
        for xx in (12.0, 34.0):
            s.ax.add_patch(mp.Rectangle((xx - 1.6, y - 0.8), 3.2, 1.6, fc="#eef3f8", ec=C, lw=0.6, zorder=3))
        I.link(s.ax, (9.3, y), (10.4, y), lw=0.8, color=C); I.link(s.ax, (35.6, y), (38.7, y), lw=0.8, color=C)
        I.link(s.ax, (13.6, y), (32.4, y), lw=1.4, color=O)
        # TBF buckets, one per direction
        for bx in (19.0, 27.0):
            s.ax.add_patch(mp.Polygon([(bx - 1.1, y + 1.3), (bx + 1.1, y + 1.3), (bx + 0.6, y + 0.15), (bx - 0.6, y + 0.15)], fc="#ffe4b5", ec=O, lw=0.6, zorder=4))
    s.t(23, 6.0, "veth pair, TBF 4.9 Mbit/s per direction", fs=5.2)
    s.panel(48, 4, 48, 21, fc="white", ec=O)
    s.t(72, 22.8, "32 network namespaces", fs=6.2, w="bold", c=O)
    for k, (lab, fc) in enumerate([("hs0..hs11", "#e8eef5"), ("hd0..hd11", "#e8eef5"), ("probe", "#dff0e0"), ("cross", "#ffe9c7")]):
        x = 51 + k * 11.3
        s.ax.add_patch(mp.FancyBboxPatch((x, 7.0), 9.6, 13.0, boxstyle="round,pad=0,rounding_size=0.6", fc="#fbfbfb", ec="#999", lw=0.6, ls="--", zorder=2))
        if k == 1:
            I.server(s.ax, x + 4.8, 16.0, s=4.0)
        else:
            I.host(s.ax, x + 4.8, 16.0, s=4.4, fc=fc)
        for j in range(3):   # IP stack layers
            s.ax.add_patch(mp.Rectangle((x + 1.8, 8.4 + j * 1.15), 6.0, 0.95, fc=["#d9e4f0", "#c5d6e8", "#adc4dc"][j], ec="none", zorder=3))
        s.t(x + 4.8, 21.2, lab, fs=5.2)
    # veth ends attached as OVS ports
    s.path([(34.8, 19.0), (36.8, 29.5), (35.2, 40.0)], c=R, ls="--", lw=0.8)
    s.arrow((72.0, 25.0), (53.4, 40.2), c=R, ls="--", lw=0.8)
    s.t(38.6, 27.4, "veth ends = OVS ports", fs=5.2, c=R, ha="left")
    return s.save("fig_implementation")


def packet_path():
    import matplotlib.patches as mp
    s = S(100, 14)
    seq = [("host", "hs8"), ("veth", ""), ("sw", "g23 HU"), ("veth", ""), ("sw", "g22 SK"), ("dots", ""), ("sw", "g34 CY"), ("veth", ""), ("server", "hd8")]
    xs = np.linspace(5, 95, len(seq))
    for (k, lab), x in zip(seq, xs):
        if k == "host": I.host(s.ax, x, 7, s=7)
        elif k == "server": I.server(s.ax, x, 7, s=6.5)
        elif k == "sw": I.switch(s.ax, x, 7, s=5.0, ms=3)
        elif k == "veth":
            s.ax.add_patch(mp.Rectangle((x - 2.6, 6.2), 5.2, 1.6, fc="#eef3f8", ec=C, lw=0.6, zorder=3))
            s.ax.add_patch(mp.Polygon([(x - 1.0, 10.2), (x + 1.0, 10.2), (x + 0.55, 8.9), (x - 0.55, 8.9)], fc="#ffe4b5", ec=O, lw=0.6, zorder=4))
            lab = "TBF"
        else:
            for d in (-1.2, 0, 1.2): s.ax.add_patch(mp.Circle((x + d, 7), 0.35, fc="#555", ec="none", zorder=3))
        if lab: s.t(x, 2.6 if k != "veth" else 11.6, lab, fs=5.8)
    for a, b in zip(xs[:-1], xs[1:]):
        s.arrow((a + 3.0, 7), (b - 3.0, 7), c=R, lw=1.0)
    return s.save("fig_packet_path")


# ---------------------------------------------------------------- real data used by the mini charts
import json
_D = {}
def data():
    if _D: return _D
    import worlds as W
    smp = W.testbed_samples(); U = np.array([a.mean() for a in smp]); V = W.testbed_twin()
    wn = [json.loads(l) for l in open("data/window_noise.jsonl")]; by = {}
    for d in wn: by.setdefault((d["x"], d["dur"]), []).append(d["u"])
    xs = sorted({k[0] for k in by}); ws = sorted({k[1] for k in by})
    pooled = [float(np.sqrt(np.mean([np.var(by[(x, w)], ddof=1) for x in xs]))) for w in ws]
    pt = json.load(open("results/policies_testbed.json"))["300"]
    bp = json.load(open("results/big_policies.json"))
    big = {k: 100 * max(np.mean(bp[l]["300"][k]["shipped"]) for l in ("mid", "high")) for k in bp["mid"]["300"]}
    bs = W.big_samples(); Ub = np.array([a.mean() for a in bs]); Vb = np.load("data/twin_big.npy")
    cc = json.load(open("results/big_ctrl_cost.json"))["loop"]["switches"]
    _D.update(U=U, V=V, ws=ws, pooled=pooled, sigma=0.0763, ustar=U.max(),
              pol={k: 100 * np.mean(v["shipped"]) for k, v in pt.items()}, big=big, Ub=Ub, Vb=Vb, cc=np.array(cc),
              twinq=json.load(open("results/twinq.json")), drift=json.load(open("results/drift.json")))
    return _D


def e1():
    D = data()
    s = S(100, 42)
    head(s, 1, 1, 26, 40, "1  Testbed")
    s.mini_testbed(6.4, 14.0, s=1.15, labels=True, ctrl_h=14.5)
    s.t(14, 5.6, "OVS bridges s1..s6;", fs=5.8)
    s.t(14, 3.6, "random B1, B2 load per test", fs=5.8)
    head(s, 29, 1, 25, 40, "2  Five passes")
    for k in range(5):
        y = 32.5 - k * 5.6
        s.t(32.6, y + 0.4, f"pass {k+1}", fs=5.6)
        s.config(36.2, y, rb(), cell=0.62, gap=0.1)
        s.t(45.3, y + 0.35, "...", fs=7)
        s.config(46.6, y, rb(), cell=0.62, gap=0.1)
    s.t(41.5, 3.6, "1024 configurations,")
    s.t(41.5, 1.9, "new random order each pass", fs=5.6)
    s.arrow((27, 21), (29, 21))
    head(s, 56, 1, 18, 40, "3  Bank")
    s.grid(58.6, 11, 18, 5, cw=2.4, ch=1.1)
    for k in range(5):
        s.t(59.8 + k * 2.4, 32.4, str(k + 1), fs=5.6)
    s.t(65, 34.4, "pass", fs=5.6)
    s.t(65, 8.3, "rows: configurations")
    s.t(65, 6.2, "5120 measured U")
    s.arrow((54, 21), (56, 21))
    head(s, 76, 1, 23, 40, "4  Analyses", c=P)
    # (a) histogram of the measured landscape
    h, e = np.histogram(D["U"], bins=12, range=(0.25, 0.85))
    X, Y = s.chart(83, 29.0, 14, 5.0, (0.25, 0.85), (0, 200), [0.3, 0.5, 0.7], [0, 200], "mean U (0–1)", "configs", title="landscape")
    for k in range(12):
        s.ax.add_patch(Rectangle_(X(e[k]) + 0.05, 29.0, X(e[k + 1]) - X(e[k]) - 0.1, Y(h[k]) - 29.0, "#1f4e79"))
    # (b) twin vs measured
    X, Y = s.chart(83, 16.3, 14, 5.6, (0.25, 0.85), (0.25, 0.85), [0.3, 0.5, 0.7], [0.3, 0.8], "twin U (0–1)", "meas. U (0–1)", title="twin vs testbed")
    idx = np.random.default_rng(1).choice(len(D["U"]), 150, replace=False)
    s.ax.scatter(X(D["V"][idx]), Y(D["U"][idx]), s=0.8, color=C, zorder=4)
    # (c) noise vs window
    X, Y = s.chart(83, 4.0, 14, 5.6, (0.5, 5.5), (0, 0.12), [1, 2, 3, 5], [0, 0.1], "window (s)", "sd of U (0–1)", title="noise vs window")
    s.ax.plot(X(D["ws"]), Y(D["pooled"]), marker="o", ms=2, color=O, lw=1, zorder=4)
    s.arrow((74, 21), (76, 21))
    return s.save("exp_e1")


def Rectangle_(x, y, w, h, fc, z=4):
    import matplotlib.patches as mp
    return mp.Rectangle((x, y), w, h, fc=fc, ec="none", zorder=z)


def gauss_xy(mu, sd, lo, hi, n=80):
    xs = np.linspace(lo, hi, n); return xs, np.exp(-0.5 * ((xs - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))


def replay(s, x0, extra_bias=False):
    """bank -> draw -> belief update -> next choice"""
    D = data()
    head(s, x0, 1, 22, 40, "Bank (replay)")
    s.grid(x0 + 3, 12, 16, 5, cw=2.6, ch=1.1, hi=(6, 2))
    s.t(x0 + 11, 33.0, "config p: draw one", fs=5.6)
    s.t(x0 + 11, 31.4, "of its measurements", fs=5.6)
    s.t(x0 + 11, 9.4, "U(p) = row mean", fs=5.6)
    head(s, x0 + 25, 1, 24, 40, "Policy and beliefs", c=G)
    X, Y = s.chart(x0 + 31, 20.0, 15.5, 12.5, (0.6, 0.9), (0, 20), [0.6, 0.7, 0.8, 0.9], [0, 10, 20],
                   "utility U (0–1)", "prob. density", title="beliefs (incumbent bold)")
    sg = D["sigma"]
    for mu, n, c in [(0.70, 1, GR), (0.76, 3, GR), (0.80, 12, G)]:
        xs, ys = gauss_xy(mu, sg / np.sqrt(n), 0.6, 0.9)
        s.ax.plot(X(xs), Y(ys), color=c, lw=1.4 if c == G else 0.9, zorder=4)
        if c == G: s.ax.fill_between(X(xs), Y(0), Y(ys), color=c, alpha=0.25, zorder=3)
    
    s.loop(x0 + 37, 8.6, r=2.3)
    s.t(x0 + 37, 4.6, "choose next p", fs=5.6)
    s.arrow((x0 + 22, 26.0), (x0 + 25, 26.0), c=R)
    s.arrow((x0 + 25, 8.6), (x0 + 22, 8.6))
    if extra_bias:
        s.t(x0 + 23.5, 28.0, "+b", fs=6.2, c=R, w="bold")


def e2():
    D = data()
    s = S(100, 42)
    replay(s, 1)
    head(s, 52, 1, 22, 40, "Policies", c=P)
    names = ["loop", "UCB", "UCB pre-scr.", "Thompson", "eps-greedy", "measure-once", "twin only", "sim.-validate"]
    for k, n in enumerate(names):
        y = 33.0 - k * 3.5
        s.loop(55.5, y, r=1.2); s.t(57.8, y, n, fs=5.6, ha="left")
    s.t(63, 3.6, "+ 4 more; budgets 100-300", fs=5.4)
    s.arrow((50, 21), (52, 21))
    head(s, 76, 1, 23, 40, "Score (300 tests)", c=P)
    keys = [("loop", "loop"), ("ucb", "UCB"), ("ucb_screen", "UCB p.-s."), ("eps", "eps-gr."), ("measure_once", "m.-once"),
            ("twin_validate", "sim.-val."), ("twin_only", "twin only")]
    X, Y = s.chart(87.0, 9.0, 10.5, 25, (90, 100), (-0.5, len(keys) - 0.5), [90, 95, 100],
                   [(len(keys) - 1 - i, n) for i, (_, n) in enumerate(keys)], "shipped U (% of opt.)", "policy")
    for i, (k, _) in enumerate(keys):
        v = D["pol"][k]; yy = float(Y(len(keys) - 1 - i))
        s.ax.add_patch(Rectangle_(87.0, yy - 1.2, float(X(v)) - 87.0, 2.4, "#1f4e79" if k == "loop" else "#9aaabb"))
    s.t(87.5, 3.4, "replay, small testbed", fs=5.0)
    s.arrow((74, 21), (76, 21))
    return s.save("exp_e2")


def zoo_example():
    z = np.load("data/zoo/Abilene_0.npz"); U = z["M"].mean(1)
    idx = np.random.default_rng(5).choice(len(U), 8, replace=False)
    return U, idx


def e3():
    import networkx as nx
    s = S(100, 42)
    head(s, 1, 1, 22, 40, "Zoo backbone")
    g = nx.Graph(nx.read_graphml(_os.path.join(_ROOT, "zoo", "Abilene.graphml"))); pos = nx.kamada_kawai_layout(g)
    P_ = {n: (12 + 8.5 * p[0], 22 + 8.5 * p[1]) for n, p in pos.items()}
    for u, v in g.edges: I.link(s.ax, P_[u], P_[v], lw=0.6, color=GR)
    for n in g.nodes: I.switch(s.ax, *P_[n], s=1.9, ms=1.6)
    s.t(12, 6.0, "8 topologies x 3", fs=5.8)
    head(s, 26, 1, 22, 40, "Traffic matrices")
    import matplotlib as mpl
    for k in range(4):
        x, y = 30 + k * 1.2, 15 + k * 1.2
        M = rng.lognormal(0, 1, (6, 6)); M = M / M.max()
        for i in range(6):
            for j in range(6):
                s.ax.add_patch(mpl.patches.Rectangle((x + j * 1.6, y + (5 - i) * 1.6), 1.6, 1.6,
                               fc=mpl.cm.Blues(0.2 + 0.8 * M[i, j]), ec="white", lw=0.2, zorder=3 + k))
    s.t(37, 10.0, "gravity matrix", fs=5.8)
    s.t(37, 7.8, "x 32 random draws", fs=5.8)
    s.arrow((23, 21), (26, 21))
    head(s, 51, 1, 22, 40, "Fluid model", c=O)
    I.switch(s.ax, 55, 23, s=2.2, ms=2); I.switch(s.ax, 69, 23, s=2.2, ms=2)
    I.link(s.ax, (56.2, 23), (67.8, 23), lw=4, color="#d9dee6")
    s.ax.plot([56.2, 67.8], [23.5, 23.5], color=C, lw=1.2, zorder=4); s.ax.plot([56.2, 67.8], [22.5, 22.5], color=R, lw=1.2, zorder=4)
    s.t(62, 26.5, "max-min fair share", fs=5.8)
    s.config(55.6, 15, rb(), cell=1.1)
    s.t(62, 13.0, "$2^{14}$ configurations", fs=5.8)
    s.t(62, 8.0, "U = mean over draws", fs=5.8)
    s.arrow((48, 21), (51, 21))
    head(s, 76, 1, 23, 40, "Replay, 3 noise levels", c=G)
    X, Y = s.chart(83, 11, 14.5, 20, (-3, 3), (0, 2.6), [-3, 0, 3], [0, 1, 2], "test minus mean U (x sd of U)", "prob. density")
    for sd, c, lab, ly in [(0.163, G, "low", 2.3), (0.6, O, "mid", 0.8), (1.2, R, "high", 0.42)]:
        xs, ys = gauss_xy(0, sd, -3, 3, 200); ys = np.minimum(ys, 2.6)
        s.ax.plot(X(xs), Y(ys), color=c, lw=1.0, zorder=4); s.ax.fill_between(X(xs), Y(0), Y(ys), color=c, alpha=0.2, zorder=3)
        s.t(float(X(1.35 if lab != "low" else 0.45)), float(Y(ly)), lab, fs=5.0, c=c, ha="left")
    s.t(87.5, 5.0, "tune on 8, test on 16", fs=5.6)
    s.arrow((73, 21), (76, 21))
    return s.save("exp_e3")


def e4():
    D = data()
    s = S(100, 36)
    head(s, 1, 1, 30, 34, "Leaders and new ideas")
    for k in range(3):
        y = 26 - k * 4.0
        s.t(4.5, y + 0.55, f"L{k+1}", fs=6, c=C, w="bold")
        s.config(7, y, rb(), cell=1.0, gap=0.15)
    s.t(16, 12.6, "new: variant of a leader", fs=5.8)
    s.config(7, 9.2, rb(), cell=1.0, gap=0.15, hi=3, c=R)
    head(s, 33, 1, 42, 34, "One round = exactly five tests")
    import matplotlib.patches as mp
    for row, n in enumerate([1, 2, 5]):
        y = 25 - row * 7
        s.t(37.5, y + 1.5, f"n = {n}", fs=6)
        for k in range(5):
            new = k < n
            s.ax.add_patch(mp.FancyBboxPatch((41 + k * 6.6, y), 6.0, 3.0, boxstyle="round,pad=0,rounding_size=0.4",
                           fc="#f9d6d5" if new else "#dbe4ef", ec=R if new else C, lw=0.7, zorder=3))
            s.t(44 + k * 6.6, y + 1.5, "new" if new else f"L{(k - n) % 3 + 1}", fs=5.8, inside=True)
    s.t(54, 5.0, "n swept over 0.5, 1, 2, 3, 4, 5", fs=5.8)
    s.arrow((31, 18), (33, 18))
    head(s, 77, 1, 22, 34, "Noise levels", c=O)
    sg = D["sigma"]
    X, Y = s.chart(84, 10, 13, 16, (-0.3, 0.3), (0, 8), [-0.3, 0, 0.3], [0, 4, 8], "test minus mean U (0–1 scale)", "prob. density")
    for sd, c, lab, lx, ly in [(sg / np.sqrt(2), G, "$\\sigma/\\sqrt{2}$", 0.06, 7.0), (sg, C, "$\\sigma$", 0.09, 4.4), (2 * sg, R, "$2\\sigma$", 0.16, 2.0)]:
        xs, ys = gauss_xy(0, sd, -0.3, 0.3, 200)
        s.ax.plot(X(xs), Y(ys), color=c, lw=1.0, zorder=4); s.ax.fill_between(X(xs), Y(0), Y(ys), color=c, alpha=0.2, zorder=3)
        s.t(float(X(lx)), float(Y(ly)), lab, fs=5.0, c=c, ha="left")
    s.t(88, 4.0, "$\\sigma$ = 0.076 (testbed)", fs=5.2)
    s.arrow((75, 18), (77, 18))
    return s.save("exp_e4")


def e6():
    D = data(); sg = D["sigma"]
    s = S(100, 42)
    replay(s, 1, extra_bias=True)
    head(s, 52, 1, 47, 40, "Bias on the incumbent's re-tests", c=R)
    X, Y = s.chart(61, 12, 33, 17, (0.5, 1.1), (0, 6), [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1], [0, 2, 4, 6],
                   "result of one test, U (0–1)", "prob. density")
    for mu, c in [(0.76, C), (0.76 + sg, R)]:
        xs, ys = gauss_xy(mu, sg, 0.5, 1.1, 200)
        s.ax.plot(X(xs), Y(ys), color=c, lw=1.0, zorder=4); s.ax.fill_between(X(xs), Y(0), Y(ys), color=c, alpha=0.25, zorder=3)
    s.arrow((float(X(0.76)), float(Y(5.6))), (float(X(0.76 + sg)), float(Y(5.6))), c=R)
    s.t(float(X(0.76 + sg / 2)), float(Y(6.3)), "b = 1 $\\sigma$ (example)", fs=5.2, c=R)
    s.t(float(X(0.6)), float(Y(4.6)), "unbiased test", fs=5.4, c=C)
    s.t(float(X(1.0)), float(Y(4.6)), "biased re-test", fs=5.4, c=R)
    s.t(75.5, 5.2, "b = 0, 0.25, ... , 3 $\\sigma$;  $\\sigma$ = 0.076", fs=5.6)
    s.arrow((50, 21), (52, 21))
    return s.save("exp_e6")


def e7():
    D = data()
    U, idx = zoo_example(); sdU = U.std(); us = U.max()
    err = np.random.default_rng(7).normal(0, 0.5 * sdU, len(idx))
    s = S(100, 40)
    head(s, 1, 1, 30, 38, "True landscape vs twin")
    X, Y = s.chart(8, 12, 21, 17, (-0.5, 7.5), (0.6, 1.0), [(i, str(i + 1)) for i in range(8)], [0.6, 0.8, 1.0],
                   "configuration p (8 of 16384)", "utility U (0–1)")
    for i, p in enumerate(idx):
        s.ax.add_patch(Rectangle_(float(X(i)) - 1.1, 12, 1.05, float(Y(U[p])) - 12, "#1f4e79"))
        s.ax.add_patch(Rectangle_(float(X(i)) + 0.05, 12, 1.05, float(Y(min(1.0, U[p] + err[i]))) - 12, "#e0b85a"))
    s.t(12, 33.2, "true U(p)", fs=5.6, c="#1f4e79"); s.t(23, 33.2, "twin, q = 0.5", fs=5.6, c=O)
    s.t(16, 6.2, "Abilene; twin error sd = q x sd of U", fs=5.2)
    head(s, 34, 1, 36, 38, "How the twin is used", c=G)
    s.t(52, 32.2, "decides:", fs=6, w="bold")
    s.t(52, 29.6, "twin only, simulate-then-validate", fs=5.6)
    I.server(s.ax, 52, 24.5, s=5.5, fc="#b8860b")
    s.t(55.5, 24.5, "twin", fs=5.8, c=O, ha="left")
    s.arrow((52, 21.8), (52, 19.0))
    s.t(52, 16.6, "only reorders ideas:", fs=6, w="bold")
    s.t(52, 14.0, "loop, pre-screened UCB", fs=5.6)
    s.loop(52, 8.6, r=2.4)
    s.arrow((31, 20), (34, 20))
    head(s, 73, 1, 26, 38, "Result vs q", c=P)
    tq = D["twinq"]; qs = sorted(tq, key=float)
    X, Y = s.chart(81, 11, 16, 18, (0, 2), (95, 100), [0, 0.5, 1, 2], [95, 100], "twin error q (x sd of U)", "shipped U (% of opt.)")
    for k, c in [("twin_only", "#b8860b"), ("loop", C)]:
        s.ax.plot(X([float(q) for q in qs]), Y([100 * tq[q][k][0] for q in qs]), color=c, lw=1.3, marker="o", ms=2, zorder=4)
    s.t(89, 33.0, "twin decides (twin only)", fs=5.0, c="#b8860b"); s.t(89, 31.4, "twin reorders (loop)", fs=5.0, c=C)
    s.t(86, 5.0, "16 Zoo landscapes, mid noise", fs=5.0)
    s.arrow((70, 20), (73, 20))
    return s.save("exp_e7")


def e8():
    D = data()
    U, idx = zoo_example(); sdU = U.std()
    s = S(100, 38)
    head(s, 1, 1, 34, 36, "Drift of the objective")
    X, Y = s.chart(8, 18, 24, 11, (0, 300), (-1, 1), [0, 100, 200, 300], [-1, 0, 1], "test number k", "drift factor $s_k$")
    ks = np.linspace(0, 300, 200)
    s.ax.plot(X(ks), Y(np.sin(2 * np.pi * 2 * ks / 300)), color=R, lw=1.2, zorder=4)
    s.ax.plot(X([0, 300]), Y([0, 0]), color="#999", lw=0.6, ls="--", zorder=3)
    s.t(18, 10.6, "test = U(p) + noise + D $s_k$ Z(p)", fs=5.8)
    s.t(18, 7.0, "$|Z(p)| \\leq 1$ from the traffic", fs=5.6)
    head(s, 38, 1, 30, 36, "Utility of 6 configurations", c=O)
    Z = np.random.default_rng(8).uniform(-1, 1, 6); Dd = 2 * 0.6 * sdU
    for k, (lab, sgn) in enumerate([("round with $s_k$ = +1", 1), ("round with $s_k$ = -1", -1)]):
        x = 43.5 + k * 12.5
        X, Y = s.chart(x, 12, 9.5, 15, (-0.5, 5.5), (0.6, 1.0), [(i, str(i + 1)) for i in range(6)], [0.6, 0.8, 1.0], "configuration p", "U (0–1)")
        for i, p in enumerate(idx[:6]):
            s.ax.add_patch(Rectangle_(float(X(i)) - 0.6, 12, 1.2, float(Y(np.clip(U[p] + sgn * Dd * Z[i], 0.6, 1.0))) - 12, "#1f4e79"))
        s.t(x + 4.75, 29.2, lab, fs=5.0)
    s.t(53, 5.6, "Abilene, D = 2 $\\sigma$ (example)", fs=5.4)
    s.arrow((35, 18), (38, 18))
    head(s, 71, 1, 28, 36, "Evaluation", c=G)
    dr = D["drift"]; Ds = sorted(dr, key=float)
    X, Y = s.chart(79, 11, 17, 16, (0, 4), (98, 100), [0, 1, 2, 4], [98, 99, 100], "drift D (multiples of $\\sigma$)", "shipped U (% of ref.)")
    s.ax.plot(X([float(d) for d in Ds]), Y([100 * dr[d]["shipped"][0] for d in Ds]), color=C, lw=1.4, marker="o", ms=2, zorder=4)
    s.t(85, 4.6, "16 Zoo landscapes", fs=5.2)
    s.arrow((68, 18), (71, 18))
    return s.save("exp_e8")


def e10():
    s = S(100, 42)
    head(s, 1, 1, 44, 40, "Live small testbed")
    s.mini_testbed(13, 10, s=1.25, labels=True)
    s.t(23, 4.4, "bundle to s1..s6, then iperf3 + ping", fs=5.8)
    head(s, 47, 1, 29, 40, "One live test", c=O)
    rows = [(0.0, 0.08, "#2e8b57", "push"), (0.08, 0.38, "#bbbbbb", "settle"), (0.38, 1.38, "#1f4e79", "TCP"),
            (0.38, 1.38, "#c98a2b", "UDP"), (0.38, 1.88, "#2e8b57", "ping"), (1.88, 2.0, "#8064a2", "compute U")]
    X, Y = s.chart(59, 9, 15, 25, (0, 2), (-0.5, 5.5), [0, 0.5, 1, 1.5, 2], [(5 - k, r[3]) for k, r in enumerate(rows)],
                   "time from start of test (s)", "activity")
    for k, (a, b, c, n) in enumerate(rows):
        yy = float(Y(5 - k))
        s.ax.add_patch(Rectangle_(float(X(a)), yy - 1.2, float(X(b)) - float(X(a)), 2.4, c))
    head(s, 78, 1, 21, 40, "Policy", c=G)
    s.loop(88.5, 28, r=3.4)
    D = data()
    X, Y = s.chart(85, 10, 12, 9, (0.6, 0.9), (0, 15), [0.6, 0.75, 0.9], [0, 15], "utility U (0–1)", "prob. density", title="belief update")
    xs, ys = gauss_xy(0.78, D["sigma"] / np.sqrt(5), 0.6, 0.9)
    s.ax.plot(X(xs), Y(ys), color=G, lw=1, zorder=4); s.ax.fill_between(X(xs), Y(0), Y(ys), color=G, alpha=0.25, zorder=3)
    s.arrow((45, 21), (47, 21), c="#2e8b57"); s.arrow((76, 21), (78, 21))
    s.path([(88.5, 24.4), (88.5, 21.6)], c=G)
    return s.save("exp_e10")


if __name__ == "__main__":
    for f in sys.argv[1:]:
        pr = globals()[f]()
        print(f, "problems:", len(pr)); [print("   ", p) for p in pr]
