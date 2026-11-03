"""Recomputes the three policies that use the twin's ranking directly (twin only,
simulate-then-validate, twin-seeded loop) after the twin's ties are broken at random
(twin only: expectation over the tied configurations). Retunes k of simulate-then-validate
on the tuning landscapes and patches every results file in which these policies appear."""
import json, numpy as np, os
import worlds as W, experiments as X

TWIN = ["twin_only", "twin_validate", "loop_twinstart"]


def runp(w, pol, tuned, budget, runs, seed0):
    fn, g = X.POLICIES[pol]; par = X.parse(tuned.get(pol)) if g else None
    if isinstance(par, list): par = tuple(par)
    res = [fn(w, budget, seed0 + s, par) for s in range(runs if pol != "twin_only" else 1)]
    return {k: [float(r[k]) for r in res] for k in ("shipped", "found", "belief", "cautious")}


def main():
    T = json.load(open("results/tuned.json")); tr, ev = X.zoo_split()
    for lvl, r in X.NOISE.items():                     # retune k on the tuning landscapes
        ws = [W.zoo_world(fn, r) for fn in tr]; sc = {}
        for k in X.GRID["k"]:
            sc[str(k)] = float(np.mean([X.POLICIES["twin_validate"][0](w, 300, 7000 + s, k)["shipped"] for w in ws for s in range(8)]))
        T[lvl]["twin_validate"] = max(sc, key=sc.get); print("tuned k", lvl, T[lvl]["twin_validate"], sc, flush=True)
    json.dump(T, open("results/tuned.json", "w"), indent=1)

    for fname, lvl in (("policies_testbed", "high"), ("policies_testbed_mid", "mid")):
        P = json.load(open(f"results/{fname}.json")); w = W.testbed_world()
        for b in P:
            for pol in TWIN: P[b][pol] = runp(w, pol, T[lvl], int(b), 400, 100000)
            print(fname, b, {p: round(np.mean(P[b][p]["shipped"]), 4) for p in TWIN}, flush=True)
        json.dump(P, open(f"results/{fname}.json", "w"))

    P = json.load(open("results/big_policies.json")); w = W.big_world()
    for lvl in P:
        for b in P[lvl]:
            for pol in TWIN: P[lvl][b][pol] = runp(w, pol, T[lvl], int(b), 400, 100000)
            print("big", lvl, b, {p: round(np.mean(P[lvl][b][p]["shipped"]), 4) for p in TWIN}, flush=True)
    json.dump(P, open("results/big_policies.json", "w"))

    P = json.load(open("results/policies_zoo.json"))
    for lvl, r in X.NOISE.items():
        for fn in ev:
            w = W.zoo_world(fn, r)
            for pol in TWIN: P[lvl][w.name][pol] = runp(w, pol, T[lvl], 300, 30, 0)
        print("zoo", lvl, {p: round(float(np.mean([np.mean(P[lvl][n][p]["shipped"]) for n in P[lvl]])), 4) for p in TWIN}, flush=True)
    json.dump(P, open("results/policies_zoo.json", "w"))

    # twin's best configuration: expectation and range over its ties
    S = json.load(open("results/zoo_stats.json"))
    for row, fn in zip(S, W.zoo_files()):
        z = np.load(fn); U = z["M"].mean(1)
        for key, V in (("twin", z["V"]), ("twin0", z["V0"])):
            st = X.twin_choice_stats(U, V, draws=1)
            row[f"{key}_argmax_pct"] = st["twin_argmax_value_pct"]; row[f"{key}_argmax_ties"] = st["twin_argmax_ties"]
            row[f"{key}_argmax_min_pct"] = st["twin_argmax_value_min_pct"]
    json.dump(S, open("results/zoo_stats.json", "w"), indent=1)
    for name, smp, V, f in (("small", W.testbed_samples(), W.testbed_twin(), "results/landscape.json"),
                            ("geant", W.big_samples(), np.load("data/twin_big.npy"), "results/big_landscape.json")):
        U = np.array([s.mean() for s in smp]); L = json.load(open(f))
        for k in ("twin_argmax_pct", "twin_argmax_rank", "twin_top10_pct", "twin_argmax_rank"): L.pop(k, None)
        L.update(X.twin_choice_stats(U, V)); json.dump(L, open(f, "w"), indent=1)
        print(name, {k: v for k, v in L.items() if k.startswith("twin_")}, flush=True)

if __name__ == "__main__":
    main()
