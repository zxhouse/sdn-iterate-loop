"""Cross-validated replay of all twelve policies (300 tests): tests drawn from some passes,
shipped configuration scored against the means of the held-out passes; averaged over splits."""
import json, math, itertools, numpy as np
import worlds as W, sdnloop as S, experiments as X
from split_replay import by_pass, world


def run(path, screen, splits, tuned, budget=300, runs=200):
    P = by_pass(path); acc = {}
    for a, b in splits:
        w = world(P, a, b, screen)
        for pol, (fn, g) in X.POLICIES.items():
            par = X.parse(tuned.get(pol)) if g else None
            if isinstance(par, list): par = tuple(par)
            v = [fn(w, budget, 100000 + s, par)["shipped"] for s in range(runs if pol != "twin_only" else 1)]
            acc.setdefault(pol, []).extend(v)
    return {p: {"mean": 100 * float(np.mean(v)), "hw": 100 * 1.96 * float(np.std(v, ddof=1)) / math.sqrt(len(v)) if len(v) > 1 else 0.0} for p, v in acc.items()}


if __name__ == "__main__":
    import sys
    T = json.load(open("results/tuned.json"))
    out = {}
    g = [([0, 1], [2, 3]), ([2, 3], [0, 1]), ([0, 2], [1, 3]), ([1, 3], [0, 2])]
    out["geant"] = {lvl: run("data/big_bank.jsonl", np.load("data/twin_big.npy"), g, T[lvl]) for lvl in ("mid", "high")}
    print("geant", {l: {p: round(v["mean"], 1) for p, v in d.items()} for l, d in out["geant"].items()}, flush=True)
    s = [([0, 1, 2], [3, 4]), ([2, 3, 4], [0, 1]), ([0, 2, 4], [1, 3]), ([1, 3, 4], [0, 2])]
    out["small"] = {lvl: run("data/bank.jsonl", W.testbed_twin(), s, T[lvl]) for lvl in ("mid", "high")}
    print("small", {l: {p: round(v["mean"], 1) for p, v in d.items()} for l, d in out["small"].items()}, flush=True)
    json.dump(out, open("results/split_full.json", "w"), indent=1)
