"""Small vector icons for network schematics (matplotlib): switch, host PC, server, controller."""
import matplotlib.patches as P
from matplotlib.path import Path

SW_FC, SW_EC = "#1f4e79", "#0f2a44"


def switch(ax, x, y, s=1.0, label=None, fs=7, z=5, fc=SW_FC, ms=4.5):
    """OpenFlow switch: flat box with two pairs of opposite arrows."""
    w, h = 1.0 * s, 0.42 * s
    ax.add_patch(P.FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle=f"round,pad=0,rounding_size={0.08*s}",
                                  fc=fc, ec=SW_EC, lw=0.6, zorder=z))
    a = 0.30 * s; dy = 0.09 * s
    for sgn, yy in ((1, y + dy), (-1, y - dy)):
        x0 = x - sgn * a / 2
        ax.annotate("", xy=(x0 + sgn * a, yy), xytext=(x0, yy), zorder=z + 1,
                    arrowprops=dict(arrowstyle="-|>", color="white", lw=0.7, mutation_scale=ms,
                                    shrinkA=0, shrinkB=0))
    if label:
        ax.text(x, y - h / 2 - 0.08 * s, label, ha="center", va="top", fontsize=fs, zorder=z + 2)


def host(ax, x, y, s=1.0, label=None, fs=6.5, z=5, fc="#e8eef5", lab_dy=None):
    """Desktop PC: monitor, stand and base."""
    w, h = 0.46 * s, 0.32 * s
    ax.add_patch(P.FancyBboxPatch((x - w / 2, y - h / 2 + 0.06 * s), w, h, boxstyle=f"round,pad=0,rounding_size={0.03*s}",
                                  fc="#3a3a3a", ec="#222", lw=0.5, zorder=z))
    ax.add_patch(P.Rectangle((x - w / 2 + 0.03 * s, y - h / 2 + 0.09 * s), w - 0.06 * s, h - 0.06 * s,
                             fc=fc, ec="none", zorder=z + 1))
    ax.add_patch(P.Rectangle((x - 0.03 * s, y - h / 2 - 0.02 * s), 0.06 * s, 0.08 * s, fc="#555", ec="none", zorder=z))
    ax.add_patch(P.Rectangle((x - 0.13 * s, y - h / 2 - 0.05 * s), 0.26 * s, 0.035 * s, fc="#555", ec="none", zorder=z))
    if label:
        dy = lab_dy if lab_dy is not None else -h / 2 - 0.08 * s
        ax.text(x, y + dy, label, ha="center", va="top", fontsize=fs, zorder=z + 2)


def server(ax, x, y, s=1.0, label=None, fs=6.5, z=5, fc="#6d6d6d"):
    """Rack/tower server with drive slots."""
    w, h = 0.30 * s, 0.50 * s
    ax.add_patch(P.FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle=f"round,pad=0,rounding_size={0.03*s}",
                                  fc=fc, ec="#333", lw=0.5, zorder=z))
    for k in range(4):
        yy = y + h / 2 - 0.09 * s - k * 0.09 * s
        ax.add_patch(P.Rectangle((x - w / 2 + 0.05 * s, yy - 0.015 * s), w - 0.10 * s, 0.03 * s, fc="#cfd8e3", ec="none", zorder=z + 1))
    ax.add_patch(P.Circle((x + w / 2 - 0.07 * s, y - h / 2 + 0.06 * s), 0.02 * s, fc="#7CFC00", ec="none", zorder=z + 1))
    if label:
        ax.text(x, y - h / 2 - 0.06 * s, label, ha="center", va="top", fontsize=fs, zorder=z + 2)


def controller(ax, x, y, s=1.0, label="SDN controller", fs=7, z=5):
    server(ax, x - 0.2 * s, y, s, fc="#2e8b57", z=z)
    server(ax, x + 0.2 * s, y, s, fc="#2e8b57", z=z)
    if label:
        ax.text(x, y - 0.33 * s, label, ha="center", va="top", fontsize=fs, zorder=z + 2, weight="bold")


def link(ax, p, q, lw=1.2, color="#7a8ea3", ls="-", z=1, alpha=1.0):
    ax.plot([p[0], q[0]], [p[1], q[1]], color=color, lw=lw, ls=ls, zorder=z, alpha=alpha, solid_capstyle="round")
