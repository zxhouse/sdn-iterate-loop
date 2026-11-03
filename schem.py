"""Drawing primitives for experiment schematics + an overlap checker."""
import os as _os
_ROOT = _os.path.dirname(_os.path.abspath(__file__))
FIGDIR = _os.environ.get("FIGDIR", _os.path.join(_ROOT, "figures"))
PNGDIR = _os.path.join(FIGDIR, "png")
_os.makedirs(PNGDIR, exist_ok=True)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Ellipse, FancyArrowPatch
from matplotlib.lines import Line2D
from matplotlib.text import Text
import numpy as np
import icons as I

C, G, O, R, P, GR = "#2b4c6f", "#2f6b3c", "#9a6b16", "#c0504d", "#8064a2", "#7a8ea3"
OUT = FIGDIR
SCR = PNGDIR
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 6.5})


class S:
    def __init__(self, w=100, h=40, figw=7.4):
        self.fig, self.ax = plt.subplots(figsize=(figw, figw * h / w))
        self.ax.set_xlim(0, w); self.ax.set_ylim(0, h); self.ax.set_aspect("equal"); self.ax.axis("off")
        self.free = []          # texts allowed to overlap patches (labels inside boxes)

    # ---------- text
    def t(self, x, y, s, fs=6.2, ha="center", va="center", c="black", w="normal", inside=False, **k):
        tx = self.ax.text(x, y, s, fontsize=fs, ha=ha, va=va, color=c, weight=w, zorder=10, **k)
        if inside: self.free.append(tx)
        return tx

    def title(self, x, y, s, c=C):
        return self.t(x, y, s, fs=6.8, w="bold", c=c)

    # ---------- shapes
    def panel(self, x, y, w, h, fc="#f6f8fb", ec="#c9d3df"):
        self.ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", fc=fc, ec=ec, lw=0.7, zorder=0))

    def arrow(self, p, q, c=C, lw=1.0, ls="-", ms=8, z=2):
        self.ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", color=c, lw=lw, ls=ls, mutation_scale=ms,
                                          shrinkA=0, shrinkB=0, zorder=z))

    def path(self, pts, c=C, lw=1.0, ls="-", z=2):
        xs, ys = zip(*pts)
        self.ax.add_line(Line2D(xs[:-1] + (xs[-1],), ys[:-1] + (ys[-1],), color=c, lw=lw, ls=ls, zorder=z))
        self.arrow(pts[-2], pts[-1], c=c, lw=lw, z=z)

    def config(self, x, y, bits, cell=1.1, gap=0.15, hi=None, c=C):
        """A configuration as a row of knob cells (filled = on)."""
        for i, b in enumerate(bits):
            fc = (R if hi == i else c) if b else "white"
            self.ax.add_patch(Rectangle((x + i * (cell + gap), y), cell, cell, fc=fc, ec=c, lw=0.5, zorder=3))
        return x + len(bits) * (cell + gap)

    def grid(self, x, y, rows, cols, cw=2.0, ch=1.0, hi=None, fc="#fff3d6", ec=O, dots_after=None):
        for r in range(rows):
            for k in range(cols):
                f = R if hi == (r, k) else ("#f2c5a5" if hi and hi[0] == r else fc)
                self.ax.add_patch(Rectangle((x + k * cw, y + (rows - 1 - r) * ch), cw, ch, fc=f, ec=ec, lw=0.4, zorder=3))

    def gauss(self, x, y, w, h, mu=0.5, sd=0.15, c=C, lw=1.0, fill=False):
        xs = np.linspace(0, 1, 60); ys = np.exp(-0.5 * ((xs - mu) / sd) ** 2)
        self.ax.plot(x + xs * w, y + ys * h, color=c, lw=lw, zorder=4)
        if fill: self.ax.fill_between(x + xs * w, y, y + ys * h, color=c, alpha=0.25, zorder=3)

    def bars(self, x, y, vals, bw=1.2, gap=0.3, h=6, c="#1f4e79", hi=None):
        for i, v in enumerate(vals):
            self.ax.add_patch(Rectangle((x + i * (bw + gap), y), bw, v * h, fc=R if hi == i else c, ec="none", zorder=3))

    def axes(self, x, y, w, h, c="#555"):
        self.ax.add_line(Line2D([x, x, x + w], [y + h, y, y], color=c, lw=0.6, zorder=3))

    def plot(self, x, y, w, h, xl, yl, title=None, fs=4.7, c="#555", xl_dy=1.15):
        """Small chart frame with labelled axes (label + unit) and optional end ticks."""
        self.ax.add_line(Line2D([x, x, x + w], [y + h, y, y], color=c, lw=0.6, zorder=3))
        self.t(x + w / 2, y - xl_dy, xl, fs=fs, va="center")
        self.t(x - 1.05, y + h / 2, yl, fs=fs, rotation=90)
        if title: self.t(x + w / 2, y + h + 1.2, title, fs=5.4, w="bold", c="#333")
        return (x, y, w, h)

    def chart(self, x, y, w, h, xr, yr, xt, yt, xl, yl, title=None, fs=4.6, tfs=4.2, c="#555"):
        """Chart frame in data coordinates: ticks with values, axis labels with units.
        xt/yt: lists of values or (value, label). Returns mappers X(v), Y(v)."""
        X = lambda v: x + (np.asarray(v, float) - xr[0]) / (xr[1] - xr[0]) * w
        Y = lambda v: y + (np.asarray(v, float) - yr[0]) / (yr[1] - yr[0]) * h
        self.ax.add_line(Line2D([x, x, x + w], [y + h, y, y], color=c, lw=0.6, zorder=3))
        norm = lambda t: [(v, (f"{v:g}")) if not isinstance(v, tuple) else v for v in t]
        for v, lab in norm(xt):
            px = float(X(v)); self.ax.add_line(Line2D([px, px], [y, y - 0.35], color=c, lw=0.5, zorder=3))
            self.t(px, y - 0.55, lab, fs=tfs, va="top")
        wmax = 0
        for v, lab in norm(yt):
            py = float(Y(v)); self.ax.add_line(Line2D([x, x - 0.35], [py, py], color=c, lw=0.5, zorder=3))
            va = "bottom" if abs(py - y) < 0.3 else ("top" if abs(py - y - h) < 0.3 else "center")
            self.t(x - 0.55, py + (0.15 if va == "bottom" else 0), lab, fs=tfs, ha="right", va=va); wmax = max(wmax, len(lab))
        self.t(x + w / 2, y - 2.35, xl, fs=fs)
        self.t(x - 1.55 - 0.62 * wmax, y + h / 2, yl, fs=fs, rotation=90)
        if title: self.t(x + w / 2, y + h + 1.2, title, fs=5.2, w="bold", c="#333")
        return X, Y

    def loop(self, x, y, r=3.0, c=G):
        self.ax.add_patch(Circle((x, y), r, fc="#e8f0e8", ec=c, lw=0.9, zorder=3))
        th = np.linspace(0.4, 2 * np.pi - 0.3, 40)
        self.ax.plot(x + 0.55 * r * np.cos(th), y + 0.55 * r * np.sin(th), color=c, lw=1.0, zorder=4)
        self.arrow((x + 0.55 * r * np.cos(th[-3]), y + 0.55 * r * np.sin(th[-3])),
                   (x + 0.55 * r * np.cos(th[-1]), y + 0.55 * r * np.sin(th[-1])), c=c, ms=6, z=5)

    def db(self, x, y, w=6, h=5, fc="#fff3d6", ec=O):
        a = self.ax
        a.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, fc=fc, ec=ec, lw=0.8, zorder=3))
        a.add_patch(Ellipse((x, y + h / 2), w, 1.6, fc=fc, ec=ec, lw=0.8, zorder=4))
        a.add_patch(Ellipse((x, y - h / 2), w, 1.6, fc=fc, ec=ec, lw=0.8, zorder=2))

    def mini_testbed(self, x, y, s=1.0, ctrl=True, hosts=True, hl=None, labels=False, ctrl_h=12.5):
        """The small testbed: s1..s6 in their real layout."""
        pos = {"s1": (0, 4), "s2": (6, 8), "s3": (6, 4), "s4": (3, 0), "s6": (9, 0), "s5": (12, 4)}
        L = [("s1", "s2"), ("s2", "s5"), ("s1", "s3"), ("s3", "s5"), ("s1", "s4"), ("s4", "s6"), ("s6", "s5"), ("s2", "s3")]
        P_ = {k: (x + v[0] * s, y + v[1] * s) for k, v in pos.items()}
        self._lab = []
        for a, b in L:
            sl = (a, b) in (("s1", "s4"), ("s4", "s6"), ("s6", "s5"), ("s2", "s3"))
            I.link(self.ax, P_[a], P_[b], lw=0.9, color="#c98a2b" if sl else GR, ls=(0, (3, 1.5)) if sl else "-")
        for k, (px, py) in P_.items():
            I.switch(self.ax, px, py, s=2.3 * s, ms=2.2, fc=R if hl and k in hl else I.SW_FC)
            if labels:
                self._lab.append((k, px, py, s))
        if ctrl:
            cx, cy = x + 6 * s, y + ctrl_h * s
            I.controller(self.ax, cx, cy, s=2.6 * s, label=None)
            if labels: self.t(cx + 2.6 * s, cy, "controller", fs=5.4, ha="left")
            for k, (px, py) in P_.items():
                self.ax.plot([cx, px], [cy - 1.2 * s, py + 0.6 * s], color="#2e8b57", lw=0.4, ls=":", zorder=0.5)
        if hosts:
            for k in range(3):
                I.host(self.ax, x - 3.5 * s, y + (2.2 + k * 2.2) * s, s=2.0 * s, fc=["#e8eef5", "#dff0e0", "#ffe9c7"][k])
                self.ax.plot([x - 2.8 * s, x - 1.1 * s], [y + (2.2 + k * 2.2) * s, y + 4 * s], color="#bbb", lw=0.4, zorder=0.5)
                I.server(self.ax, x + 15.5 * s, y + (2.2 + k * 2.2) * s, s=1.9 * s)
                self.ax.plot([x + 13.1 * s, x + 15.0 * s], [y + 4 * s, y + (2.2 + k * 2.2) * s], color="#bbb", lw=0.4, zorder=0.5)
            if labels:
                self.t(x - 3.5 * s, y + 9.0 * s, "sources", fs=5.2)
                self.t(x + 15.5 * s, y + 9.0 * s, "sinks", fs=5.2)
        for k, px, py, sc in self._lab:
            self.place(px, py, k, sc)
        return P_

    def place(self, px, py, txt, sc=1.0, fs=5.4):
        """Put a label next to an icon at the first free spot (no line, text or icon hit)."""
        cands = [(0, -1.3), (0, 1.3), (1.9, 0.9), (-1.9, 0.9), (1.9, -0.9), (-1.9, -0.9), (0, -2.0), (0, 2.0), (2.6, 0), (-2.6, 0)]
        cands += [(r * np.cos(a), r * np.sin(a)) for r in (1.4, 1.8, 2.3) for a in np.linspace(0, 2 * np.pi, 24, endpoint=False)]
        for dx, dy in cands:
            tx = self.t(px + dx * sc, py + dy * sc, txt, fs=fs)
            if not self._hits(tx):
                return tx
            tx.remove()
        return self.t(px, py - 1.3 * sc, txt, fs=fs)

    def _hits(self, tx):
        self.fig.canvas.draw(); rend = self.fig.canvas.get_renderer()
        bb = tx.get_window_extent(rend).expanded(1.05, 1.1)
        for ln in self.ax.lines:
            xy = ln.get_transform().transform(ln.get_xydata())
            for k in range(len(xy) - 1):
                for u in np.linspace(0, 1, 16):
                    px, py = xy[k] + (xy[k + 1] - xy[k]) * u
                    if bb.x0 < px < bb.x1 and bb.y0 < py < bb.y1: return True
        for o in self.ax.texts:
            if o is not tx and o.get_text().strip() and bb.overlaps(o.get_window_extent(rend)): return True
        for p in self.ax.patches:
            if p.get_zorder() >= 3 and not isinstance(p, FancyArrowPatch) and bb.overlaps(p.get_window_extent(rend)): return True
        return False

    # ---------- overlap check
    def check(self, name):
        self.fig.canvas.draw()
        rend = self.fig.canvas.get_renderer()
        texts = [t for t in self.ax.texts if t.get_text().strip()]
        bbs = [t.get_window_extent(rend).expanded(1.02, 1.05) for t in texts]
        problems = []
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                if bbs[i].overlaps(bbs[j]):
                    problems.append(f"text/text: '{texts[i].get_text()[:25]}' vs '{texts[j].get_text()[:25]}'")
        # text vs lines and arrows
        segs = []
        for ln in self.ax.lines:
            xy = ln.get_transform().transform(ln.get_xydata())
            for k in range(len(xy) - 1):
                segs.append((xy[k], xy[k + 1], "line"))
        for p in self.ax.patches:
            if isinstance(p, FancyArrowPatch):
                for poly in p.get_path().to_polygons(closed_only=False):
                    xy = p.get_transform().transform(poly)
                    for k in range(len(xy) - 1):
                        segs.append((xy[k], xy[k + 1], "arrow"))
        for t, bb in zip(texts, bbs):
            for a, b, kind in segs:
                n = 12
                for u in np.linspace(0, 1, n):
                    px, py = a + (b - a) * u
                    if bb.x0 < px < bb.x1 and bb.y0 < py < bb.y1:
                        problems.append(f"text/{kind}: '{t.get_text()[:30]}'"); break
                else:
                    continue
                break
        # text sticking out of a box edge (partial overlap with a filled patch outline)
        for t, bb in zip(texts, bbs):
            for p in self.ax.patches:
                if isinstance(p, FancyArrowPatch) or p.get_zorder() <= 0:
                    continue
                pb = p.get_window_extent(rend)
                if pb.width < 3 or pb.height < 3:
                    continue
                inter = bb.overlaps(pb)
                contained = pb.x0 <= bb.x0 and bb.x1 <= pb.x1 and pb.y0 <= bb.y0 and bb.y1 <= pb.y1
                if inter and not contained and t not in self.free:
                    problems.append(f"text/shape: '{t.get_text()[:30]}'"); break
                if inter and not contained and t in self.free:
                    # inside-text must be contained in at least its own box
                    pass
        # icon vs icon (switches, hosts, servers are drawn at zorder >= 5)
        ic = [p.get_window_extent(rend) for p in self.ax.patches
              if not isinstance(p, FancyArrowPatch) and p.get_zorder() >= 5 and isinstance(p, FancyBboxPatch)]
        for i in range(len(ic)):
            for j in range(i + 1, len(ic)):
                a, b = ic[i], ic[j]
                if min(a.x1, b.x1) - max(a.x0, b.x0) > 0.5 and min(a.y1, b.y1) - max(a.y0, b.y0) > 0.5:
                    problems.append(f"icon/icon at ({a.x0:.0f},{a.y0:.0f})")
        return sorted(set(problems))

    def save(self, name, check=True):
        probs = self.check(name) if check else []
        self.fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight")
        self.fig.savefig(f"{SCR}/{name}.png", dpi=120, bbox_inches="tight")
        plt.close(self.fig)
        return probs
