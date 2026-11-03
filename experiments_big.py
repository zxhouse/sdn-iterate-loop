"""Replay experiments on the GEANT testbed bank. Usage: python3 experiments_big.py <section>..."""
import sys, json, time, numpy as np
import sdnloop as S, worlds as W, experiments as X


def sec_landscape():
    smp = W.big_samples(); V = np.load("data/twin_big.npy")
    U = np.array([s.mean() for s in smp]); sd = np.array([s.std(ddof=1) for s in smp]); n = np.array([len(s) for s in smp])
    var_true = U.var() - np.mean(sd ** 2 / n); pooled = float(np.sqrt(np.mean(sd ** 2)))
    rec = [json.loads(l) for l in open("data/big_bank.jsonl")]
    res = {"U_star": float(U.max()), "U_min": float(U.min()), "U_mean": float(U.mean()), "U_sd_true": float(np.sqrt(var_true)),
           "sigma": pooled, "ratio": pooled / float(np.sqrt(var_true)), "sigma_q10_q90": [float(np.quantile(sd, .1)), float(np.quantile(sd, .9))],
           "lo_r1": int(X.local_opt_mask(U, 10, 1).sum()), "lo_r2": int(X.local_opt_mask(U, 10, 2).sum()),
           "within2pct": int((U >= 0.98 * U.max()).sum()),
           "twin_pearson": float(np.corrcoef(U, V)[0, 1]),
           "twin_spearman": float(np.corrcoef(np.argsort(np.argsort(U)), np.argsort(np.argsort(V)))[0, 1]),
           "twin_bias": float((V - U).mean()), "twin_mae": float(np.abs(V - U).mean()),
           **X.twin_choice_stats(U, V),
           "argmax": "".join(str((int(U.argmax()) >> i) & 1) for i in range(10)),
           "pass_means": [float(np.mean([d["u"] for d in rec if d["pass"] == p])) for p in sorted({d["pass"] for d in rec})],
           "retried": int(sum(d.get("retried", False) for d in rec)), "n_tests": len(rec),
           "sec_per_test": float(np.median(np.diff(sorted(d["t"] for d in rec))))}
    rng = np.random.default_rng(0); wins = np.zeros(len(U))
    for _ in range(500):
        wins[np.array([s[rng.integers(len(s), size=len(s))].mean() for s in smp]).argmax()] += 1
    res["bootstrap_same_best"] = float(wins[U.argmax()] / 500)
    json.dump(res, open("results/big_landscape.json", "w"), indent=1); print(json.dumps(res, indent=1))


def sec_policies(runs=400):
    T = json.load(open("results/tuned.json")); w = W.big_world(); out = {}
    for lvl in ("mid", "high"):
        out[lvl] = {}
        for b in (100, 200, 300):
            out[lvl][b] = X.run_policies(w, T[lvl], b, runs, seed0=100000)
            print(lvl, b, {p: round(np.mean(v["shipped"]), 4) for p, v in out[lvl][b].items()}, flush=True)
    json.dump(out, open("results/big_policies.json", "w"))


def sec_allocation(runs=400):
    w = W.big_world(); out = {}
    for nn in (0.5, 1, 2, 3, 4, 5):
        out[nn] = X.ci([S.loop(w, 300, 200000 + s, n_new=nn, alloc=5)["shipped"] for s in range(runs)])
    print(out, flush=True); json.dump(out, open("results/big_allocation.json", "w"))


def sec_ablation(runs=400):
    w = W.big_world(); out = {}
    for name, kw in X.ABL.items():
        out[name] = X.ci([X.run_abl(w, kw, 300, 400000 + s) for s in range(runs)])
    print({k: round(v[0], 4) for k, v in out.items()}, flush=True); json.dump(out, open("results/big_ablation.json", "w"))


def sec_dynamics(runs=200):
    w = W.big_world(); U = w.U; lo2 = X.local_opt_mask(U, 10, 2); out = {}
    for H in (20, 60, 200):
        sh, fd, be, l2 = [], [], [], []
        for s in range(runs):
            o = S.loop(w, 5 * H, 900000 + s); sh.append(o["shipped"]); fd.append(o["found"]); be.append(o["belief"]); l2.append(lo2[o["inc"]])
        out[H] = {"shipped": X.ci(sh), "found": X.ci(fd), "belief": X.ci(be), "inc_lo_r2": float(np.mean(l2))}
        print(H, out[H], flush=True)
    json.dump(out, open("results/big_dynamics.json", "w"))


if __name__ == "__main__":
    for s in sys.argv[1:]:
        t = time.time(); globals()["sec_" + s](); print(s, "done", round(time.time() - t), flush=True)
