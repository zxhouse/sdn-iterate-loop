"""Fluid twin of the GEANT testbed (same model as twin_testbed.py)."""
import itertools, numpy as np
import bigbed as B
from fluid import maxmin

DL = []
for (a, b) in B.G.edges:
    DL += [(a, b), (b, a)]
LI = {l: i for i, l in enumerate(DL)}
CAP = np.array([B.CAP[tuple(sorted(l))] for l in DL])
COMBOS = list(itertools.product(B.BG_RATES, repeat=len(B.BG)))


def predict(x):
    paths, asleep = B.resolve(x)
    A = np.zeros((len(DL), B.N_DEM))
    for j in range(B.N_DEM):
        p = paths[f"D{j}"]
        for k in range(len(p) - 1):
            A[LI[(p[k], p[k + 1])], j] = 1
    fcap = np.array(B.DRATE, float)
    for m in B.MT:
        if x[B.FEATURES.index(f"meterD{m}")]:
            fcap[m] = 0.5 * B.DRATE[m]
    cap = np.tile(CAP, (len(COMBOS), 1))
    for bi, combo in enumerate(COMBOS):
        for b, r in zip(B.BG, combo):
            p = paths[b]
            for k in range(len(p) - 1):
                cap[bi, LI[(p[k], p[k + 1])]] -= r
    for (a, b) in asleep:
        cap[:, LI[(a, b)]] = 0.0
    cap = np.maximum(cap, 0.0)
    rate = maxmin(A, cap, np.tile(fcap, (len(COMBOS), 1)))
    load = rate @ A.T
    sat = (load >= cap - 1e-6) & (cap > 0)
    T = np.mean(np.minimum(rate / np.array(B.DRATE), 1.0), axis=1)
    pp = paths["PR"]; hops = [LI[(pp[k], pp[k + 1])] for k in range(len(pp) - 1)]
    # the probe also queues behind saturated cross-traffic links it shares
    rtt = 1.0 + 50.0 * sat[:, hops].sum(axis=1)
    L = np.maximum(0.0, 1.0 - rtt / B.RTT_MAX)
    E = (len(asleep) // 2) / 4.0
    return float((B.WEIGHTS["T"] * T + B.WEIGHTS["L"] * L + B.WEIGHTS["E"] * E).mean())


if __name__ == "__main__":
    V = np.zeros(1 << B.NF)
    for c in range(1 << B.NF):
        V[c] = predict([(c >> i) & 1 for i in range(B.NF)])   # bit i = feature i (same as worlds.code_of)
    np.save("data/twin_big.npy", V); print("twin max %.3f mean %.3f sd %.3f" % (V.max(), V.mean(), V.std()))
