"""Cross-validated replay of all twelve policies on the small testbed over every split of the
five passes: tests drawn from k passes, the shipped configuration scored against the means of
the other 5-k passes (k = 3 and k = 2), 300 tests, tuned at the testbed's noise level."""
import json, math, itertools, numpy as np
import worlds as W, experiments as X
from split_replay import by_pass, world

T = json.load(open("results/tuned.json"))["high"]
P = by_pass("data/bank.jsonl"); V = W.testbed_twin()
out = {}
for k in (3, 2):
    per = {}
    for a in itertools.combinations(range(5), k):
        b = [q for q in range(5) if q not in a]
        w = world(P, list(a), b, V)
        for pol, (fn, g) in X.POLICIES.items():
            par = X.parse(T.get(pol)) if g else None
            if isinstance(par, list): par = tuple(par)
            v = [fn(w, 300, 200000 + s, par)["shipped"] for s in range(100)]
            per.setdefault(pol, []).append(100 * float(np.mean(v)))
    res = {}
    for pol, m in per.items():
        m = np.array(m); res[pol] = {"mean": float(m.mean()), "split_means": m.tolist()}
    # paired differences to the loop across splits (same reference means)
    for pol in res:
        d = np.array(per[pol]) - np.array(per["loop"])
        res[pol]["diff_loop"] = float(d.mean()); res[pol]["diff_sd_splits"] = float(d.std(ddof=1))
    out[f"draw{k}_ref{5-k}"] = res
    print(k, {p: (round(r["mean"], 1), round(r["diff_loop"], 1), round(r["diff_sd_splits"], 1)) for p, r in res.items()}, flush=True)
json.dump(out, open("results/split_small_all.json", "w"), indent=1)
