"""Cross-validated replay: tests are drawn from some passes of the bank and what is shipped is
scored against the means of the other passes. In the ordinary replay the same measurements
both produce the test outcomes and define the reference, so a configuration whose bank mean is
high by chance is also easy to pick; scoring on held-out passes removes that shared noise."""
import json, math, numpy as np
import worlds as W, sdnloop as S, experiments as X

POLS = ["loop", "ucb_screen", "measure_once"]


def by_pass(path):
    P = {}
    for l in open(path):
        d = json.loads(l); P.setdefault(d["pass"], {}).setdefault(W.code_of(d["x"]), []).append(d["u"])
    return P


def world(P, draw_passes, score_passes, screen):
    n = 1 << W.NF_TB
    dr = [np.array([u for q in draw_passes for u in P[q][c]]) for c in range(n)]
    sc = np.array([np.mean([u for q in score_passes for u in P[q][c]]) for c in range(n)])
    sd = float(np.sqrt(np.mean([s.var(ddof=1) for s in dr])))
    def draw(c, rng):
        s = dr[c]; return float(s[rng.integers(len(s))])
    return S.World(sc, draw, sd, screen, "split")


def run(name, path, screen, splits, tuned, runs=400):
    P = by_pass(path); out = {}
    for lab, (a, b) in splits.items():
        w = world(P, a, b, screen); out[lab] = {}
        for p in POLS:
            fn, g = X.POLICIES[p]; par = X.parse(tuned.get(p)) if g else None
            if isinstance(par, list): par = tuple(par)
            v = [fn(w, 100, 100000 + s, par)["shipped"] for s in range(runs)]
            out[lab][p] = {"mean": 100 * float(np.mean(v)), "hw": 100 * 1.96 * float(np.std(v, ddof=1)) / math.sqrt(runs)}
        print(name, lab, {p: round(x["mean"], 2) for p, x in out[lab].items()}, flush=True)
    return out


if __name__ == "__main__":
    T = json.load(open("results/tuned.json"))["high"]
    res = {}
    res["geant"] = run("geant", "data/big_bank.jsonl", np.load("data/twin_big.npy"),
                       {"same_01": ([0, 1], [0, 1]), "split_01_23": ([0, 1], [2, 3]), "split_23_01": ([2, 3], [0, 1])}, T)
    res["small"] = run("small", "data/bank.jsonl", W.testbed_twin(),
                       {"same_01": ([0, 1], [0, 1]), "split_01_23": ([0, 1], [2, 3]), "split_23_01": ([2, 3], [0, 1])}, T)
    json.dump(res, open("results/split_replay.json", "w"), indent=1)
