"""Loaders for the two tiers of worlds: the measured testbed bank and the topology worlds."""
import json, glob, numpy as np, os
import sdnloop as S

NF_TB = 10


def code_of(xs):  # feature i = character i  ->  bit i
    return sum(int(ch) << i for i, ch in enumerate(xs))


def testbed_samples(path="data/bank.jsonl", min_n=1):
    S_ = [[] for _ in range(1 << NF_TB)]
    for l in open(path):
        d = json.loads(l)
        S_[code_of(d["x"])].append(d["u"])
    return [np.array(s) for s in S_]


def testbed_twin():
    P = np.load("data/twin_testbed.npy")       # indexed by int(xs, 2) (MSB = feature 0)
    V = np.zeros(1 << NF_TB)
    for c in range(1 << NF_TB):
        xs = "".join(str((c >> i) & 1) for i in range(NF_TB))
        V[c] = P[int(xs, 2)]
    return V


def testbed_world(extra_sd=0.0, screen="twin"):
    smp = testbed_samples()
    V = testbed_twin() if screen == "twin" else None
    return S.bank_world(smp, screen=V, name="testbed", extra_sd=extra_sd)


def zoo_files():
    return sorted(glob.glob("data/zoo/*.npz"))


def zoo_world(fn, ratio=None, screen="twin"):
    """ratio: target total-noise / sd(U); None keeps traffic-matrix noise only."""
    z = np.load(fn); M = z["M"]; V = z["V"]
    U = M.mean(1); sd_tm = float(np.sqrt(np.mean(M.var(1, ddof=1))))
    sm = 0.0
    if ratio is not None:
        tot = ratio * U.std()
        sm = float(np.sqrt(max(tot ** 2 - sd_tm ** 2, 0.0)))
    name = os.path.basename(fn)[:-4]
    return S.matrix_world(M, screen=V if screen == "twin" else None, sigma_m=sm, name=name)


def big_samples(path="data/big_bank.jsonl"):
    S_ = [[] for _ in range(1 << NF_TB)]
    for l in open(path):
        d = json.loads(l)
        S_[code_of(d["x"])].append(d["u"])
    return [np.array(s) for s in S_]


def big_world(extra_sd=0.0, screen="twin"):
    smp = big_samples()
    V = np.load("data/twin_big.npy") if screen == "twin" else None
    return S.bank_world(smp, screen=V, name="geant", extra_sd=extra_sd)
