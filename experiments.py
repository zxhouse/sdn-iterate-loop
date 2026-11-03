"""All replay experiments. Usage: python3 experiments.py <section> ...
Results are written to results/<section>.json."""
import sys, json, os, time, itertools
import numpy as np
import sdnloop as S
import worlds as W

os.makedirs("results", exist_ok=True)
NOISE = {"tm": None, "mid": 0.6, "high": 1.2}   # total noise / sd(U) in the topology tier


def ci(x):
    x = np.asarray(x, float)
    return float(x.mean()), float(1.96 * x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0


def save(name, obj):
    json.dump(obj, open(f"results/{name}.json", "w"), indent=1)


def twin_choice_stats(U, V, k=10, draws=2000, seed=0):
    """Value of the twin's best configuration and of the best of its top k, with the twin's
    ties broken at random: expectation (and range) over the tie-breaking."""
    top = np.flatnonzero(np.isclose(V, V.max(), rtol=0, atol=1e-12)); us = U.max()
    ranks = [int((U > U[c]).sum()) + 1 for c in top]
    rng = np.random.default_rng(seed); best = []
    for _ in range(draws):
        o = np.lexsort((rng.random(V.size), -V))[:k]; best.append(U[o].max())
    return {"twin_argmax_value_pct": float(100 * U[top].mean() / us), "twin_argmax_ties": int(top.size),
            "twin_argmax_value_min_pct": float(100 * U[top].min() / us), "twin_argmax_value_max_pct": float(100 * U[top].max() / us),
            "twin_argmax_rank_median": float(np.median(ranks)), "twin_argmax_rank_min": int(min(ranks)), "twin_argmax_rank_max": int(max(ranks)),
            "twin_top10_best_pct": float(100 * np.mean(best) / us)}


def local_opt_mask(U, n, r):
    offs = [sum(1 << i for i in comb) for k in range(1, r + 1) for comb in itertools.combinations(range(n), k)]
    codes = np.arange(U.size)
    m = np.ones(U.size, bool)
    for o in offs:
        m &= U >= U[codes ^ o]
    return m


POLICIES = {
    "random": (S.p_random, None),
    "hill": (S.p_greedy, "reps"),
    "eps": (S.p_eps, "eps"),
    "ucb": (S.p_ucb, "ucb"),
    "ucb_screen": (lambda w, b, s, p: S.p_ucb(w, b, s, p, screen=True), "ucb"),
    "thompson": (S.p_thompson, "ts"),
    "measure_once": (lambda w, b, s, p: S.p_measure_once(w, b, s), None),
    "twin_only": (S.p_twin_only, None),
    "twin_validate": (S.p_twin_validate, "k"),
    "loop_blind": (lambda w, b, s, p: S.p_loop(w, b, s, screen=False), None),
    "loop": (lambda w, b, s, p: S.p_loop(w, b, s), None),
    "loop_twinstart": (S.p_loop_twinstart, None),
}
GRID = {"reps": [1, 2, 4], "eps": [0.1, 0.3, 0.5, 0.7],
        "ucb": [(c, a) for c in (0.0125, 0.025, 0.05, 0.1, 0.2) for a in (2, 3, 5)],
        "ts": [2, 3, 5], "k": [3, 5, 10, 20]}


def tune(worlds, budget=300, runs=8):
    best = {}
    for pol, (fn, g) in POLICIES.items():
        if g is None:
            continue
        scores = {}
        for par in GRID[g]:
            v = [fn(w, budget, 7000 + s, par)["shipped"] for w in worlds for s in range(runs)]
            scores[str(par)] = float(np.mean(v))
        bp = max(scores, key=scores.get)
        best[pol] = bp
        print("  tuned", pol, bp, round(scores[bp], 4), flush=True)
    return best


def parse(p):
    return None if p is None else json.loads(p.replace("(", "[").replace(")", "]"))


def run_policies(w, tuned, budget, runs, seed0=0):
    out = {}
    for pol, (fn, g) in POLICIES.items():
        if pol in ("twin_only", "twin_validate", "loop_twinstart", "ucb_screen") and w.screen is None:
            continue
        par = parse(tuned.get(pol)) if g else None
        if isinstance(par, list): par = tuple(par)
        res = [fn(w, budget, seed0 + s, par) for s in range(runs if pol != "twin_only" else 1)]
        out[pol] = {k: [float(r[k]) for r in res] for k in ("shipped", "found", "belief", "cautious")}
    return out


# ------------------------------------------------------------------ sections
def sec_landscape():
    """E1: characterisation of the measured testbed landscape and of the twin."""
    smp = W.testbed_samples(); V = W.testbed_twin()
    n = np.array([len(s) for s in smp]); U = np.array([s.mean() for s in smp])
    sd = np.array([s.std(ddof=1) for s in smp])
    pooled = float(np.sqrt(np.mean(sd ** 2)))
    # spread of the true utility: variance of means minus estimation variance
    var_true = U.var() - np.mean(sd ** 2 / n)
    res = {"n_configs": int(len(U)), "reps_min": int(n.min()), "reps_max": int(n.max()),
           "U_star": float(U.max()), "argmax": int(U.argmax()), "U_mean": float(U.mean()),
           "U_sd_raw": float(U.std()), "U_sd_true": float(np.sqrt(max(var_true, 0))),
           "sigma_pooled": pooled, "sigma_min": float(sd.min()), "sigma_max": float(sd.max()),
           "sigma_q10_q90": [float(np.quantile(sd, .1)), float(np.quantile(sd, .9))],
           "ratio": pooled / float(np.sqrt(max(var_true, 1e-12))),
           "corr_sd_mean": float(np.corrcoef(sd, U)[0, 1]),
           "twin_pearson": float(np.corrcoef(U, V)[0, 1]),
           "twin_spearman": float(np.corrcoef(np.argsort(np.argsort(U)), np.argsort(np.argsort(V)))[0, 1]),
           "twin_bias": float((V - U).mean()), "twin_mae": float(np.abs(V - U).mean()),
           **twin_choice_stats(U, V),
           }
    for r in (1, 2):
        m = local_opt_mask(U, 10, r); res[f"local_optima_r{r}"] = int(m.sum())
    # robustness of the optimum: bootstrap the bank means
    rng = np.random.default_rng(0); wins = np.zeros(len(U))
    for _ in range(500):
        Ub = np.array([s[rng.integers(len(s), size=len(s))].mean() for s in smp])
        wins[Ub.argmax()] += 1
    res["bootstrap_argmax_share"] = float(wins.max() / 500)
    res["bootstrap_n_distinct_argmax"] = int((wins > 0).sum())
    # per-component noise (T, L) from the raw records
    rec = [json.loads(l) for l in open("data/bank.jsonl")]
    by = {}
    for d in rec:
        by.setdefault(d["x"], []).append((d["T"], d["L"], d["rtt"]))
    sT = np.sqrt(np.mean([np.var([a[0] for a in v], ddof=1) for v in by.values() if len(v) > 1]))
    sL = np.sqrt(np.mean([np.var([a[1] for a in v], ddof=1) for v in by.values() if len(v) > 1]))
    res["sigma_T"], res["sigma_L"] = float(sT), float(sL)
    res["wall_seconds_per_test"] = float(np.median(np.diff(sorted(d["t"] for d in rec))))
    # top configurations
    top = np.argsort(-U)[:8]
    res["top"] = [{"code": int(c), "x": "".join(str((int(c) >> i) & 1) for i in range(10)),
                   "U": float(U[c]), "sd": float(sd[c]), "twin": float(V[c])} for c in top]
    save("landscape", res); print(json.dumps({k: v for k, v in res.items() if k != "top"}, indent=1))


def sec_zoo_stats():
    rows = []
    for fn in W.zoo_files():
        z = np.load(fn); M, V, V0 = z["M"], z["V"], z["V0"]; U = M.mean(1)
        rows.append({"world": os.path.basename(fn)[:-4], "nodes": int(z["n"]), "edges": int(z["m"]),
                     "U_star": float(U.max()), "U_sd": float(U.std()),
                     "sd_tm": float(np.sqrt(np.mean(M.var(1, ddof=1)))),
                     "lo_r1": int(local_opt_mask(U, 14, 1).sum()), "lo_r2": int(local_opt_mask(U, 14, 2).sum()),
                     "twin_pearson": float(np.corrcoef(U, V)[0, 1]),
                     "twin_argmax_pct": twin_choice_stats(U, V)["twin_argmax_value_pct"],
                     "twin_argmax_ties": twin_choice_stats(U, V)["twin_argmax_ties"],
                     "twin0_argmax_pct": twin_choice_stats(U, V0)["twin_argmax_value_pct"]})
        print(rows[-1], flush=True)
    save("zoo_stats", rows)


def zoo_split():
    f = W.zoo_files()
    return [x for x in f if x.endswith("_0.npz")], [x for x in f if not x.endswith("_0.npz")]


def sec_tune():
    tr, _ = zoo_split(); out = {}
    for lvl, r in NOISE.items():
        print("noise", lvl, flush=True)
        out[lvl] = tune([W.zoo_world(fn, r) for fn in tr])
    save("tuned", out)


def sec_policies_zoo(runs=30, budget=300):
    tuned = json.load(open("results/tuned.json")); _, ev = zoo_split(); out = {}
    for lvl, r in NOISE.items():
        out[lvl] = {}
        for fn in ev:
            w = W.zoo_world(fn, r)
            out[lvl][w.name] = run_policies(w, tuned[lvl], budget, runs)
            print(lvl, w.name, {p: round(np.mean(v["shipped"]), 4) for p, v in out[lvl][w.name].items()}, flush=True)
    save("policies_zoo", out)


def sec_policies_testbed(runs=400):
    tuned = json.load(open("results/tuned.json"))["high"]   # testbed noise ratio is close to 1
    w = W.testbed_world(); out = {}
    for budget in (100, 200, 300):
        out[budget] = run_policies(w, tuned, budget, runs, seed0=100000)
        print(budget, {p: round(np.mean(v["shipped"]), 4) for p, v in out[budget].items()}, flush=True)
    save("policies_testbed", out)


def sec_policies_testbed_mid(runs=400):
    tuned = json.load(open("results/tuned.json"))["mid"]
    w = W.testbed_world(); out = {}
    for budget in (100, 200, 300):
        out[budget] = run_policies(w, tuned, budget, runs, seed0=100000)
        print(budget, {p: round(np.mean(v["shipped"]), 4) for p, v in out[budget].items()}, flush=True)
    save("policies_testbed_mid", out)


def sec_allocation(runs_tb=400, runs_zoo=16):
    out = {"testbed": {}, "zoo": {}}
    w = W.testbed_world()
    for scale in (0.5, 1.0, 2.0):   # scale the testbed noise: extra Gaussian noise to reach scale*sigma
        ex = 0.0 if scale <= 1 else w.sigma * np.sqrt(scale ** 2 - 1)
        ww = W.testbed_world(extra_sd=ex) if scale >= 1 else None
        if scale < 1:
            # lower noise: average two independent draws per test
            base = W.testbed_world()
            def draw2(c, rng, b=base):
                return 0.5 * (b.draw(c, rng) + b.draw(c, rng))
            ww = S.World(base.U, draw2, base.sigma / np.sqrt(2), base.screen, "testbed-avg2")
        out["testbed"][scale] = {}
        for nn in (0.5, 1, 2, 3, 4, 5):
            v = [S.loop(ww, 300, 200000 + s, n_new=nn, alloc=5)["shipped"] for s in range(runs_tb)]
            out["testbed"][scale][nn] = ci(v)
        print("tb", scale, {k: round(v[0], 4) for k, v in out["testbed"][scale].items()}, flush=True)
    _, ev = zoo_split()
    for lvl, r in NOISE.items():
        out["zoo"][lvl] = {}
        for nn in (0.5, 1, 2, 3, 4, 5):
            per = [np.mean([S.loop(W.zoo_world(fn, r), 300, 300000 + s, n_new=nn, alloc=5)["shipped"]
                            for s in range(runs_zoo)]) for fn in ev]
            out["zoo"][lvl][nn] = ci(per)
        print("zoo", lvl, {k: round(v[0], 4) for k, v in out["zoo"][lvl].items()}, flush=True)
    save("allocation", out)


ABL = {
    "baseline": {},
    "r_max=1": {"r_max": 1}, "r_max=3": {"r_max": 3}, "unbounded": {"r_max": None},
    "ideas=1": {"ideas": 1}, "ideas=15": {"ideas": 15},
    "no_screen": {"screen": False},
    "memory=1": {"memory": 1}, "memory=3": {"memory": 3},
    "all_new": {"n_new": 5, "alloc": 5}, "no_rotation": {"n_rr": 0},
}


def run_abl(w, kw, budget, seed):
    kw = dict(kw)
    if kw.get("r_max", 0) is None:
        kw["r_max"] = w.n
    return S.loop(w, budget, seed, **kw)["shipped"]


def sec_ablation(runs_tb=400, runs_zoo=16):
    out = {"testbed": {}, "zoo": {}}
    w = W.testbed_world()
    for name, kw in ABL.items():
        out["testbed"][name] = ci([run_abl(w, kw, 300, 400000 + s) for s in range(runs_tb)])
    print("tb", {k: round(v[0], 4) for k, v in out["testbed"].items()}, flush=True)
    _, ev = zoo_split()
    ws = [W.zoo_world(fn, NOISE["mid"]) for fn in ev]
    for name, kw in ABL.items():
        per = [np.mean([run_abl(z, kw, 300, 500000 + s) for s in range(runs_zoo)]) for z in ws]
        out["zoo"][name] = ci(per)
    print("zoo", {k: round(v[0], 4) for k, v in out["zoo"].items()}, flush=True)
    save("ablation", out)


def sec_bias(runs_tb=400, runs_zoo=12):
    out = {"testbed": {}, "zoo": {}}
    w = W.testbed_world(); sg = w.sigma
    mult = (0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0)
    for m in mult:
        out["testbed"][m] = ci([S.loop(w, 300, 600000 + s, bias=m * sg)["shipped"] for s in range(runs_tb)])
    print("tb sigma", sg, {k: round(v[0], 4) for k, v in out["testbed"].items()}, flush=True)
    _, ev = zoo_split()
    ws = [W.zoo_world(fn, NOISE["mid"]) for fn in ev]
    for m in mult:
        per = [np.mean([S.loop(z, 300, 700000 + s, bias=m * z.sigma)["shipped"] for s in range(runs_zoo)]) for z in ws]
        out["zoo"][m] = ci(per)
    print("zoo", {k: round(v[0], 4) for k, v in out["zoo"].items()}, flush=True)
    out["sigma_testbed"] = sg
    save("bias", out)


def sec_dynamics(runs=300):
    """Trajectories on the testbed replay: shipped, found, belief, cautious per round; horizon to local optimum."""
    w = W.testbed_world(); U = w.U
    lo1, lo2 = local_opt_mask(U, 10, 1), local_opt_mask(U, 10, 2)
    T = 400  # rounds = 2000 tests
    acc = {k: np.zeros((runs, T)) for k in ("shipped", "found", "belief", "cautious")}
    is_lo = np.zeros((runs, T)); found_lo = np.zeros((runs, T))
    for s in range(runs):
        R = S.loop(w, 5 * T, 800000 + s, trace=True)
        for k_, o in enumerate(R["trace"][:T]):
            for k in acc:
                acc[k][s, k_] = o[k]
    out = {k: [ci(acc[k][:, t]) for t in range(T)] for k in acc}
    # local-optimum check at selected horizons: re-run with explicit incumbents
    hz = {}
    for H in (20, 60, 200, 400, 1000):
        sh, fd, l1, l2, f2 = [], [], [], [], []
        for s in range(150):
            o = S.loop(w, 5 * H, 900000 + s)
            sh.append(o["shipped"]); fd.append(o["found"])
            l1.append(lo1[o["inc"]]); l2.append(lo2[o["inc"]]); f2.append(lo2[o["found_code"]])
        hz[H] = {"shipped": ci(sh), "found": ci(fd), "inc_lo_r1": float(np.mean(l1)),
                 "inc_lo_r2": float(np.mean(l2)), "found_lo_r2": float(np.mean(f2))}
        print("horizon", H, hz[H], flush=True)
    out["horizon"] = hz
    save("dynamics", out)
    print({k: (round(out[k][19][0], 4), round(out[k][59][0], 4), round(out[k][-1][0], 4)) for k in acc}, flush=True)


def sec_twinq(runs=16):
    """Value of testing as a function of twin fidelity (topology tier, mid noise).
    Twin = true utility plus a fixed per-configuration error of sd q*sd(U)."""
    tuned = json.load(open("results/tuned.json"))["mid"]; _, ev = zoo_split(); out = {}
    for q in (0.0, 0.25, 0.5, 1.0, 2.0):
        res = {p: [] for p in ("twin_only", "twin_validate", "loop", "measure_once", "ucb_screen", "loop_blind")}
        for i, fn in enumerate(ev):
            w = W.zoo_world(fn, NOISE["mid"])
            rng = np.random.default_rng(31337 + i)
            w.screen = w.U + rng.normal(0, q * w.U.std(), w.U.size)
            for p in res:
                fnp, g = POLICIES[p]
                par = parse(tuned.get(p)) if g else None
                if isinstance(par, list): par = tuple(par)
                n = 1 if p == "twin_only" else runs
                res[p].append(float(np.mean([fnp(w, 300, 950000 + s, par)["shipped"] for s in range(n)])))
        out[q] = {p: ci(v) for p, v in res.items()}
        print(q, {p: round(v[0], 4) for p, v in out[q].items()}, flush=True)
    save("twinq", out)


def sec_drift(runs=16):
    """Bounded drift of the objective (Corollary): U_k = U + D * s_k * Z, |Z| <= 1, s_k a slow
    sinusoid over the run (two periods in 300 tests). Z is built from the traffic-matrix
    variability of the world: the normalised difference between two halves of the draws."""
    _, ev = zoo_split(); out = {}
    for Dm in (0.0, 0.5, 1.0, 2.0, 4.0):           # D in units of the world's test-noise sd
        sh, gap = [], []
        for i, fn in enumerate(ev):
            z = np.load(fn); M = z["M"]; K = M.shape[1]
            Z = M[:, :K // 2].mean(1) - M[:, K // 2:].mean(1); Z = Z / np.abs(Z).max()
            w = W.zoo_world(fn, NOISE["mid"]); D = Dm * w.sigma; base = w.draw
            state = {"n": 0}
            def draw(c, rng, base=base, Z=Z, D=D, st=state):
                s = np.sin(2 * np.pi * st["n"] / 150.0); st["n"] += 1
                return base(c, rng) + D * s * Z[c]
            ww = S.World(w.U, draw, w.sigma, w.screen, w.name)
            v = []
            for s in range(runs):
                state["n"] = 0
                o = S.loop(ww, 300, 990000 + s)
                c = o["inc"]
                # 2D-local optimality w.r.t. the reference utility: max gain of a 1-2 knob variant
                n = w.n; offs = [1 << a for a in range(n)] + [(1 << a) | (1 << b) for a in range(n) for b in range(a + 1, n)]
                best_var = max(w.U[c ^ o_] for o_ in offs)
                v.append((o["shipped"], (best_var - w.U[c]) / (2 * D) if D > 0 else np.nan))
            sh.append(np.mean([a for a, b in v])); gap.append(np.nanmean([b for a, b in v]) if D > 0 else np.nan)
        out[Dm] = {"shipped": ci(sh), "gap_over_2D": ci(gap) if Dm > 0 else None}
        print(Dm, out[Dm], flush=True)
    save("drift", out)


if __name__ == "__main__":
    for sec in sys.argv[1:]:
        t = time.time(); globals()["sec_" + sec](); print(sec, "done in", round(time.time() - t), "s", flush=True)
