"""Fluid digital twin of the emulated testbed: predicts U(x) for every configuration
from topology, capacities, demands, meters and the cross-traffic distribution."""
import itertools, numpy as np
import testbed as tb
from fluid import maxmin

DLINKS = []
for (a, b) in tb.LINKS:
    DLINKS += [(a, b), (b, a)]
LIDX = {l: i for i, l in enumerate(DLINKS)}
CAP = np.array([tb.LINKS.get(l, tb.LINKS.get((l[1], l[0]))) for l in DLINKS], float)
BG_COMBOS = list(itertools.product(tb.BG_RATES, repeat=len(tb.BG)))
QDELAY = 50.0  # ms per saturated hop (TBF latency bound)
BASE_RTT = 1.0


def predict(x, bg_combos=BG_COMBOS):
    paths, asleep = tb.resolve_paths(x)
    flows = list(tb.FLOWS)
    A = np.zeros((len(DLINKS), len(flows)))
    for j, f in enumerate(flows):
        p = paths[f]
        for k in range(len(p) - 1):
            A[LIDX[(p[k], p[k+1])], j] = 1
    fcap = np.array([tb.FLOWS[f][2] for f in flows], float)
    for feat, (fl, kbps) in tb.METERS.items():
        if x[tb.FEATURES.index(feat)]:
            j = flows.index(fl); fcap[j] = min(fcap[j], kbps / 1000.0)
    Bn = len(bg_combos)
    cap = np.tile(CAP, (Bn, 1))
    for bi, combo in enumerate(bg_combos):
        for (b, r) in zip(tb.BG, combo):
            p = tb.BG[b][3]
            for k in range(len(p) - 1):
                cap[bi, LIDX[(p[k], p[k+1])]] -= r
    cap = np.maximum(cap, 0.0)
    rate = maxmin(A, cap, np.tile(fcap, (Bn, 1)))
    load = rate @ A.T
    sat = load >= cap - 1e-6
    T = np.mean(np.minimum(rate / np.array([tb.FLOWS[f][2] for f in flows]), 1.0), axis=1)
    pp = paths["PR"]
    hops = [LIDX[(pp[k], pp[k+1])] for k in range(len(pp) - 1)]
    rtt = BASE_RTT + QDELAY * sat[:, hops].sum(axis=1)
    L = np.maximum(0.0, 1.0 - rtt / tb.RTT_MAX)
    E = tb.energy_score(asleep) / 0.5
    U = tb.WEIGHTS["T"] * T + tb.WEIGHTS["L"] * L + tb.WEIGHTS["E"] * E
    return float(U.mean())


def all_predictions():
    out = np.zeros(1 << tb.NF)
    for xs in itertools.product([0, 1], repeat=tb.NF):
        c = int("".join(map(str, xs)), 2)
        out[c] = predict(list(xs))
    return out


if __name__ == "__main__":
    P = all_predictions()
    np.save("data/twin_testbed.npy", P)
    print("twin: max %.3f mean %.3f sd %.3f" % (P.max(), P.mean(), P.std()))
    for xs in ["0000000000", "0000010000", "1111100000", "0100110111", "0010011000"]:
        print(xs, round(P[int(xs, 2)], 3))
